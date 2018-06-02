import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from placenet import PlaceNet
from tripletnet import TripletNet
from dataset import RecordedAirSimDataLoader
import constants

class PlaceRecognition:
    def __init__(self):
        self.model = PlaceNet()
        self.tripletnet = TripletNet(self.model)
        self.normalize = transforms.Normalize(
            #mean=[121.50361069 / 127., 122.37611083 / 127., 121.25987563 / 127.],
            mean=[127. / 255., 127. / 255., 127. / 255.],
            std=[1 / 255., 1 / 255., 1 / 255.]
        )

        self.preprocess = transforms.Compose([
            transforms.Resize(constants.TRAINING_PLACE_IMAGE_SCALE),
            transforms.CenterCrop(constants.PLACE_IMAGE_SIZE),
            transforms.ToTensor(),
            self.normalize
        ])

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])

    def cuda(self):
        self.model.cuda()

    def forward(self, input):
        image_tensor = self.preprocess(input)
        image_tensor.unsqueeze_(0)
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            image_tensor = image_tensor.cuda()
        image_variable = Variable(image_tensor)
        return self.model(image_variable) # get representation

    def compute_similarity_score(self, rep1, rep2):
        similarity_score = F.cosine_similarity(rep1, rep2)
        return similarity_score[0]

    def train(self, datapath, checkpoint_path, train_iterations):
        use_gpu = torch.cuda.is_available()
        # criterion = nn.CrossEntropyLoss()
        criterion = torch.nn.MarginRankingLoss(margin=constants.TRAINING_PLACE_MARGIN)
        optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, self.tripletnet.parameters())), lr=constants.TRAINING_PLACE_LR, momentum=constants.TRAINING_PLACE_MOMENTUM)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=constants.TRAINING_PLACE_LR_SCHEDULER_SIZE, gamma=constants.TRAINING_PLACE_LR_SCHEDULER_GAMMA)
 
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_loader = torch.utils.data.DataLoader(RecordedAirSimDataLoader(datapath, locomotion=False, transform=self.preprocess), batch_size=constants.TRAINING_PLACE_BATCH, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(RecordedAirSimDataLoader(datapath, locomotion=False, transform=self.preprocess, validation=True), batch_size=constants.TRAINING_PLACE_BATCH, shuffle=True, **kwargs)
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
                    anchor, positive, negative = data

                    # wrap them in Variable
                    if use_gpu:
                        anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
                    anchor, positive, negative = Variable(anchor), Variable(positive), Variable(negative)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # dist_a, dist_b, embedded_x, embedded_y, embedded_z = self.tripletnet(anchor, positive, negative) eucludian dist
                    ## 1 means, dist_a should be larger than dist_b
                    # target = torch.FloatTensor(dist_a.size()).fill_(-1)

                    similarity_a, similarity_b, embedded_x, embedded_y, embedded_z = self.tripletnet(anchor, positive, negative)
                    # 1 means, similarity_a should be larger than similarity_b
                    target = torch.FloatTensor(similarity_a.size()).fill_(1)
                    if use_gpu:
                        target = target.cuda()
                    target = Variable(target)
        
                    # loss_triplet = criterion(dist_a, dist_b, target)
                    loss_triplet = criterion(similarity_a, similarity_b, target)
                    loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
                    loss = loss_triplet + 0.001 * loss_embedd

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item()
                    # running_corrects += torch.sum(dist_a < dist_b)
                    running_corrects += torch.sum(similarity_a > similarity_b)

                epoch_loss = (float(running_loss) / float(len(data_loaders[phase].dataset))) * 100.0
                epoch_acc = (float(running_corrects) / float(len(data_loaders[phase].dataset))) * 100.0

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = self.model.state_dict()
                    print (checkpoint_path, epoch)
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
        torch.save(state, os.path.join(checkpoint_path, "place_checkpoint_{}.pth".format(epoch)))

    def eval(self, datapath):
        use_gpu = torch.cuda.is_available()
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        data_loader = torch.utils.data.DataLoader(RecordedAirSimDataLoader(datapath, locomotion=False, transform=self.preprocess, validation=True), batch_size=constants.TRAINING_PLACE_BATCH, shuffle=True, **kwargs)
        criterion = torch.nn.MarginRankingLoss(margin=constants.TRAINING_PLACE_MARGIN)

        running_loss = 0.0
        running_corrects = 0
        for data in tqdm(data_loader):
            anchor, positive, negative = data

            # wrap them in Variable
            if use_gpu:
                anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
            anchor, positive, negative = Variable(anchor), Variable(positive), Variable(negative)

            # dist_a, dist_b, embedded_x, embedded_y, embedded_z = self.tripletnet(anchor, positive, negative)
            ## 1 means, dist_a should be larger than dist_b
            # target = torch.FloatTensor(dist_a.size()).fill_(-1)

            similarity_a, similarity_b, embedded_x, embedded_y, embedded_z = self.tripletnet(anchor, positive, negative)
            # 1 means, similarity_a should be larger than similarity_b
            target = torch.FloatTensor(similarity_a.size()).fill_(1)
            if use_gpu:
                target = target.cuda()
            target = Variable(target)
        
            # loss_triplet = criterion(dist_a, dist_b, target)
            loss_triplet = criterion(similarity_a, similarity_b, target)
            loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
            loss = loss_triplet + 0.001 * loss_embedd

            running_loss += loss.item()
            running_corrects += torch.sum(similarity_a < similarity_b)

        epoch_loss = (float(running_loss) / float(len(data_loaders[phase].dataset))) * 100.0
        epoch_acc = (float(running_corrects) / float(len(data_loaders[phase].dataset))) * 100.0
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
