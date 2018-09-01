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

        # packed_array = np.concatenate([current_tensor, closest_tensor, future_tensor], axis=0)
        packed_array = np.concatenate([current_tensor, future_tensor], axis=0)
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
        # weights = [1.0, 1.5, 1.5, 0.5, 0.5, 0.5]
        # class_weights = torch.FloatTensor(weights).cuda()
        # criterion = nn.CrossEntropyLoss(weight=class_weights)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=constants.TRAINING_LOCO_LR, momentum=constants.TRAINING_LOCO_MOMENTUM)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=constants.TRAINING_LOCO_LR_SCHEDULER_SIZE, gamma=constants.TRAINING_LOCO_LR_SCHEDULER_GAMMA)
 
        kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_loader = torch.utils.data.DataLoader(RecordedAirSimDataLoader(datapath, locomotion=True, transform=self.placeRecognition.preprocess, validation=False), batch_size=constants.TRAINING_LOCO_BATCH, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(RecordedAirSimDataLoader(datapath, locomotion=True, transform=self.placeRecognition.preprocess, validation=True), batch_size=constants.TRAINING_LOCO_BATCH, shuffle=True, **kwargs)
        data_loaders = { 'train': train_loader, 'val': val_loader }

        since = time.time()

        best_model_wts = self.model.state_dict()
        best_acc = 0.0

        for epoch in range(train_iterations):
            print('Epoch {}/{}'.format(epoch, train_iterations - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    exp_lr_scheduler.step()
                    self.model.train(True)  # Set model to training mode
                else:
                    self.model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for data in tqdm(data_loaders[phase]):
                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    if use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = self.model(inputs)
                    if type(outputs) == tuple:
                        outputs, _ = outputs
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = (float(running_loss) / float(len(data_loaders[phase].dataset))) * 100.0
                epoch_acc = (float(running_corrects) / float(len(data_loaders[phase].dataset))) * 100.0

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = self.model.state_dict()
                    self.save_model(checkpoint_path, epoch)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model


    def save_model(self, checkpoint_path, epoch):
        print ("Saving a new checkpoint on epoch {}".format(epoch+1))
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
        }
        torch.save(state, os.path.join(checkpoint_path, "nav_{}_checkpoint_{}.pth".format(constants.LOCO_NUM_CLASSES, epoch)))

    def eval(self, datapath):
        use_gpu = torch.cuda.is_available()
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        data_loader = torch.utils.data.DataLoader(RecordedAirSimDataLoader(datapath, locomotion=True, transform=self.preprocess, validation=True), batch_size=constants.TRAINING_LOCO_BATCH, shuffle=True, **kwargs)

        running_corrects = 0
        for data in tqdm(data_loader):
            inputs, labels = data

            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = self.model(inputs)
            if type(outputs) == tuple:
                outputs, _ = outputs
            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_acc = (float(running_corrects) / float(len(data_loader.dataset))) * 100.0
        print("Accuracy {} ".format(epoch_acc))
