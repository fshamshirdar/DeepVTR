import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
from torchvision import models
from loconet import LocoNet
from sklearn.metrics import accuracy_score

import numpy as np
import os
import time
from tqdm import tqdm

from placenavnet import PlaceNavNet
from dataset import RecordedAirSimDataLoader
import dataset
import constants

class PlaceNavigation:
    def __init__(self, placeRecognition, checkpoint=None, use_cuda=True):
        self.model = PlaceNavNet(num_classes=constants.LOCO_NUM_CLASSES)  # (num_classes=constants.LOCO_NUM_CLASSES)
        self.target_model = PlaceNavNet(num_classes=constants.LOCO_NUM_CLASSES)  # (num_classes=constants.LOCO_NUM_CLASSES)
        self.placeRecognition = placeRecognition

        if (checkpoint is not None):
            self.load_weights(checkpoint)

        if (use_cuda):
            self.cuda()

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])

    def cuda(self):
        self.placeRecognition.cuda()
        self.model.cuda()
        self.target_model.cuda()

    def forward(self, current_state, closest_state, future_state):
        current_tensor = self.placeRecognition.forward(current_state, flatten=False)
        # closest_tensor = self.placeRecognition.forward(future_state, flatten=False)
        future_tensor = self.placeRecognition.forward(future_state, flatten=False)

        current_tensor_detached = current_tensor.detach()
        future_tensor_detached = future_tensor.detach()

        # packed_array = np.concatenate([current_tensor, closest_tensor, future_tensor], axis=0)
        packed_array = np.concatenate([current_tensor_detached, future_tensor_detached], axis=0)
        packed_tensor = torch.from_numpy(packed_array)
        packed_tensor.unsqueeze_(0)
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            packed_tensor = packed_tensor.cuda()
        packed_variable = Variable(packed_tensor)
        output = self.model(packed_variable)
        return F.softmax(output)

    def train(self, datapath, checkpoint_path, train_iterations):
        use_gpu = torch.cuda.is_available()

        kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_loader = torch.utils.data.DataLoader(RecordedAirSimDataLoader(datapath, datatype=dataset.TYPE_PLACE_NAVIGATION, transform=self.placeRecognition.preprocess, validation=False), batch_size=constants.TRAINING_LOCO_BATCH, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(RecordedAirSimDataLoader(datapath, datatype=dataset.TYPE_PLACE_NAVIGATION, transform=self.placeRecognition.preprocess, validation=True), batch_size=constants.TRAINING_LOCO_BATCH, shuffle=True, **kwargs)
        data_loaders = { 'train': train_loader, 'val': val_loader }

        since = time.time()

        # PlaceNav
        criterion1 = nn.CrossEntropyLoss()
        optimizer1 = optim.SGD(list(filter(lambda p: p.requires_grad, self.placeRecognition.model.parameters())) + list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=constants.TRAINING_LOCO_LR, momentum=constants.TRAINING_LOCO_MOMENTUM)
        # optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=constants.TRAINING_LOCO_LR, momentum=constants.TRAINING_LOCO_MOMENTUM)
        exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=constants.TRAINING_LOCO_LR_SCHEDULER_SIZE, gamma=constants.TRAINING_LOCO_LR_SCHEDULER_GAMMA)
 
        best_model_wts1 = self.model.state_dict()
        best_acc1 = 0.0

        # Place
        criterion2 = torch.nn.MarginRankingLoss(margin=constants.TRAINING_PLACE_MARGIN)
        optimizer2 = optim.SGD(list(filter(lambda p: p.requires_grad, self.placeRecognition.tripletnet.parameters())), lr=constants.TRAINING_PLACE_LR, momentum=constants.TRAINING_PLACE_MOMENTUM)
        exp_lr_scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=constants.TRAINING_PLACE_LR_SCHEDULER_SIZE, gamma=constants.TRAINING_PLACE_LR_SCHEDULER_GAMMA)
 
        best_model_wts2 = self.placeRecognition.model.state_dict()
        best_acc2 = 0.0

        for epoch in range(train_iterations):
            print('Epoch {}/{}'.format(epoch, train_iterations - 1))
            print('-' * 10)

            self.placeRecognition.model.train(False)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    exp_lr_scheduler1.step()
                    self.model.train(True)  # Set model to training mode
                    exp_lr_scheduler2.step()
                    self.placeRecognition.model.train(True)
                else:
                    self.model.train(False)  # Set model to evaluate mode
                    self.placeRecognition.model.train(False)

                running_loss1 = 0.0
                running_corrects1 = 0

                running_loss2 = 0.0
                running_corrects2 = 0

                # Iterate over data.
                for data in tqdm(data_loaders[phase]):
                    # get the inputs
                    current_states, future_states, negative_states, packed_states, actions = data

                    # wrap them in Variable
                    if use_gpu:
                        current_states1 = Variable(current_states.cuda())
                        future_states1 = Variable(future_states.cuda())
                        negative_states1 = Variable(negative_states.cuda())
                        actions1 = Variable(actions.cuda())
                    else:
                        current_states1 = Variable(current_states)
                        future_states1 = Variable(future_states)
                        negative_states1 = Variable(negative_states)
                        actions1 = Variable(actions)

                    # PLACE_NAV
                    current_reps = self.placeRecognition.model(current_states1, flatten=False)
                    future_reps = self.placeRecognition.model(future_states1, flatten=False)

                    packed_reps = torch.cat([current_reps, future_reps], dim=1)

                    # packed_reps = torch.cat([current_reps.data, future_reps.data], dim=1)
                    # if use_gpu:
                    #     packed_reps = Variable(packed_reps.cuda())
                    # else:
                    #     packed_reps = Variable(packed_reps)

                    # zero the parameter gradients
                    optimizer1.zero_grad()

                    # forward
                    outputs = self.model(packed_reps)
                    if type(outputs) == tuple:
                        outputs, _ = outputs
                    _, preds = torch.max(outputs.data, 1)
                    loss1 = criterion1(outputs, actions1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss1.backward()
                        optimizer1.step()

                    # statistics
                    running_loss1 += loss1.item()
                    running_corrects1 += torch.sum(preds == actions1.data)

                    # PLACE
                    # wrap them in Variable
                    if use_gpu:
                        current_states2 = Variable(current_states.cuda())
                        future_states2 = Variable(future_states.cuda())
                        negative_states2 = Variable(negative_states.cuda())
                        actions2 = Variable(actions.cuda())
                    else:
                        current_states2 = Variable(current_states)
                        future_states2 = Variable(future_states)
                        negative_states2 = Variable(negative_states)
                        actions2 = Variable(actions)

                    # zero the parameter gradients
                    optimizer2.zero_grad()

                    # dist_a, dist_b, embedded_x, embedded_y, embedded_z = self.tripletnet(anchor, positive, negative) # eucludian dist
                    ## 1 means, dist_a should be larger than dist_b
                    # target = torch.FloatTensor(dist_a.size()).fill_(-1)

                    similarity_a, similarity_b, embedded_x, embedded_y, embedded_z = self.placeRecognition.tripletnet(current_states2, future_states2, negative_states2)
                    # 1 means, similarity_a should be larger than similarity_b
                    target = torch.FloatTensor(similarity_a.size()).fill_(1)
                    if use_gpu:
                        target = target.cuda()
                    target = Variable(target)
        
                    # loss_triplet = criterion(dist_a, dist_b, target)
                    loss_triplet = criterion2(similarity_a, similarity_b, target)
                    loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
                    loss2 = loss_triplet + 0.001 * loss_embedd

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss2.backward()
                        optimizer2.step()

                    # statistics
                    running_loss2 += loss2.item()
                    # running_corrects += torch.sum(dist_a < dist_b)
                    running_corrects2 += torch.sum(similarity_a > similarity_b)

                epoch_loss1 = (float(running_loss1) / float(len(data_loaders[phase].dataset))) * 100.0
                epoch_acc1 = (float(running_corrects1) / float(len(data_loaders[phase].dataset))) * 100.0
                epoch_loss2 = (float(running_loss2) / float(len(data_loaders[phase].dataset))) * 100.0
                epoch_acc2 = (float(running_corrects2) / float(len(data_loaders[phase].dataset))) * 100.0

                print('{} Loss1: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss1, epoch_acc1))
                print('{} Loss2: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss2, epoch_acc2))

                # deep copy the model
                if phase == 'val' and epoch_acc1 > best_acc1:
                    best_acc1 = epoch_acc1
                    best_model_wts1 = self.model.state_dict()
                    self.save_model(checkpoint_path, epoch)
                # deep copy the model
                if phase == 'val' and epoch_acc2 > best_acc2:
                    best_acc2 = epoch_acc2
                    best_model_wts2 = self.placeRecognition.model.state_dict()
                    self.placeRecognition.save_model(checkpoint_path, epoch)

            print()
                
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f} {:4f}'.format(best_acc1, best_acc2))

        # load best model weights
        self.model.load_state_dict(best_model_wts1)
        self.placeRecognition.model.load_state_dict(best_model_wts2)
        return self.model


    def save_model(self, checkpoint_path, epoch):
        print ("Saving a new checkpoint on epoch {}".format(epoch+1))
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
        }
        torch.save(state, os.path.join(checkpoint_path, "placenav_{}_checkpoint_{}.pth".format(constants.LOCO_NUM_CLASSES, epoch)))

    def eval(self, datapath):
        use_gpu = torch.cuda.is_available()
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        data_loader = torch.utils.data.DataLoader(RecordedAirSimDataLoader(datapath, datatype=dataset.TYPE_PLACE_NAVIGATION, transform=self.preprocess, validation=True), batch_size=constants.TRAINING_LOCO_BATCH, shuffle=True, **kwargs)

        running_corrects = 0
        for data in tqdm(data_loader):
            # get the inputs
            current_states, future_states, negative_states, packed_states, actions = data

            # wrap them in Variable
            if use_gpu:
                current_states = Variable(current_states.cuda())
                future_states = Variable(future_states.cuda())
                negative_states = Variable(negative_states.cuda())
                actions = Variable(actions.cuda())
            else:
                current_states = Variable(current_states)
                future_states = Variable(future_states)
                negative_states = Variable(negative_states)
                actions = Variable(actions)

            current_reps = self.placeRecognition.model(current_states, flatten=False)
            future_reps = self.placeRecognition.model(future_states, flatten=False)
            current_reps_detached = current_reps.detach()
            future_reps_detached = future_reps.detach()

            packed_reps = torch.cat([current_reps_detached, future_reps_detached], dim=1)

            outputs = self.model(packed_reps)
            if type(outputs) == tuple:
                outputs, _ = outputs
            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == actions.data)

        epoch_acc = (float(running_corrects) / float(len(data_loader.dataset))) * 100.0
        print("Accuracy {} ".format(epoch_acc))
