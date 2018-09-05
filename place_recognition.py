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
from siamesenet import SiameseNet
from dataset import RecordedAirSimDataLoader
from dataset import OnlineVizDoomDataLoader
import constants

class Replicate(nn.Module):
    def __init__(self):
        super(Replicate, self).__init__()

    def forward(self, x):
       return x

class PlaceRecognition:
    def __init__(self, checkpoint=None, use_cuda=True):
        if (constants.PLACE_NETWORK == constants.PLACE_NETWORK_PLACENET):
            self.model = PlaceNet()
            self.normalize = transforms.Normalize(
                mean=[127. / 255., 127. / 255., 127. / 255.],
                std=[1 / 255., 1 / 255., 1 / 255.]
            )
        elif (constants.PLACE_NETWORK == constants.PLACE_NETWORK_RESNET18):
            kwargs = {'num_classes': 512}
            # kwargs = {}
            self.model = models.resnet18(**kwargs)
            # self.model.fc = Replicate()
            self.normalize = transforms.Normalize(
                mean = [0.5, 0.5, 0.5],
                std = [0.5, 0.5, 0.5]
            )
        else:
            print ("Place network is not valid!")

        if (constants.PLACE_TOP_SIAMESE):
            self.siamesenet = SiameseNet(self.model)
        else:
            self.tripletnet = TripletNet(self.model)

        self.preprocess = transforms.Compose([
            transforms.Resize(constants.TRAINING_PLACE_IMAGE_SCALE),
            transforms.CenterCrop(constants.PLACE_IMAGE_SIZE),
            transforms.ToTensor(),
            self.normalize
        ])

        self.array_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(constants.TRAINING_PLACE_IMAGE_SCALE),
            transforms.CenterCrop(constants.PLACE_IMAGE_SIZE),
            transforms.ToTensor(),
            self.normalize
        ])

        if (checkpoint is not None):
            self.load_weights(checkpoint)

        self.use_cuda = use_cuda
        if (use_cuda):
            self.cuda()

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if (constants.PLACE_TOP_SIAMESE):
            self.siamesetnet.load_state_dict(checkpoint['state_dict'])
        else: # triplet
            self.model.load_state_dict(checkpoint['state_dict'])

    def cuda(self):
        self.model.cuda()
        if (constants.PLACE_TOP_SIAMESE):
            self.siamesenet.cuda()
        else:
            self.tripletnet.cuda()

    def forward(self, input, flatten=True):
        if (isinstance(input, (np.ndarray, np.generic))):
            image_tensor = self.array_preprocess(input)
        else:
            image_tensor = self.preprocess(input)
        image_tensor.unsqueeze_(0)
        if self.use_cuda:
            image_tensor = image_tensor.cuda()
        image_variable = Variable(image_tensor)
        return self.model(image_variable, flatten) # get representation

    def compute_similarity_score(self, rep1, rep2):
        if (constants.PLACE_TOP_MODEL):
           similarity_score = self.siamesenet(rep1, rep2, embedding_required=False)
           similarity_score = similarity_score[0][1] # positive similarity
        else: # triplet
           similarity_score = F.cosine_similarity(rep1, rep2)
           similarity_score = similarity_score[0]
        return similarity_score

    def train(self, datapath, checkpoint_path, train_iterations, online_learning=False):
        if (constants.PLACE_TOP_SIAMESE):
            if (online_learning):
                return self.train_siamese_online(constants.VIZDOOM_DEFAULT_WAD, checkpoint_path, train_iterations)
            else:
                return self.train_siamese(datapath, checkpoint_path, train_iterations)
        else:
            return self.train_triplet(datapath, checkpoint_path, train_iterations)

    def eval(self, datapath):
        if (constants.PLACE_TOP_SIAMESE):
            return self.eval_siamese(datapath, checkpoint_path, train_iterations)
        else:
            return self.eval_triplet(datapath, checkpoint_path, train_iterations)

    def train_triplet(self, datapath, checkpoint_path, train_iterations):
        # criterion = nn.CrossEntropyLoss()
        criterion = torch.nn.MarginRankingLoss(margin=constants.TRAINING_PLACE_MARGIN)
        optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, self.tripletnet.parameters())), lr=constants.TRAINING_PLACE_LR, momentum=constants.TRAINING_PLACE_MOMENTUM)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=constants.TRAINING_PLACE_LR_SCHEDULER_SIZE, gamma=constants.TRAINING_PLACE_LR_SCHEDULER_GAMMA)
 
        kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
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
                    if self.use_cuda:
                        anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
                    anchor, positive, negative = Variable(anchor), Variable(positive), Variable(negative)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # dist_a, dist_b, embedded_x, embedded_y, embedded_z = self.tripletnet(anchor, positive, negative) # eucludian dist
                    ## 1 means, dist_a should be larger than dist_b
                    # target = torch.FloatTensor(dist_a.size()).fill_(-1)

                    similarity_a, similarity_b, embedded_x, embedded_y, embedded_z = self.tripletnet(anchor, positive, negative)
                    # 1 means, similarity_a should be larger than similarity_b
                    target = torch.FloatTensor(similarity_a.size()).fill_(1)
                    if self.use_cuda:
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
                    running_corrects += torch.sum(similarity_a > similarity_b + constants.TRAINING_PLACE_MARGIN)

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

    def eval_triplet(self, datapath):
        kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
        data_loader = torch.utils.data.DataLoader(RecordedAirSimDataLoader(datapath, locomotion=False, transform=self.preprocess, validation=True), batch_size=constants.TRAINING_PLACE_BATCH, shuffle=True, **kwargs)
        criterion = torch.nn.MarginRankingLoss(margin=constants.TRAINING_PLACE_MARGIN)

        running_loss = 0.0
        running_corrects = 0
        for data in tqdm(data_loader):
            anchor, positive, negative = data

            # wrap them in Variable
            if self.use_cuda:
                anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
            anchor, positive, negative = Variable(anchor), Variable(positive), Variable(negative)

            # dist_a, dist_b, embedded_x, embedded_y, embedded_z = self.tripletnet(anchor, positive, negative)
            ## 1 means, dist_a should be larger than dist_b
            # target = torch.FloatTensor(dist_a.size()).fill_(-1)

            similarity_a, similarity_b, embedded_x, embedded_y, embedded_z = self.tripletnet(anchor, positive, negative)
            print (similarity_a - similarity_b)
            # 1 means, similarity_a should be larger than similarity_b
            target = torch.FloatTensor(similarity_a.size()).fill_(1)
            if self.use_cuda:
                target = target.cuda()
            target = Variable(target)
        
            # loss_triplet = criterion(dist_a, dist_b, target)
            loss_triplet = criterion(similarity_a, similarity_b, target)
            loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
            loss = loss_triplet + 0.001 * loss_embedd

            running_loss += loss.item()
            running_corrects += torch.sum(similarity_a > similarity_b)

        epoch_loss = (float(running_loss) / float(len(data_loaders[phase].dataset))) * 100.0
        epoch_acc = (float(running_corrects) / float(len(data_loaders[phase].dataset))) * 100.0
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    def train_siamese(self, datapath, checkpoint_path, train_iterations):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, self.siamesenet.parameters())), lr=constants.TRAINING_PLACE_LR, momentum=constants.TRAINING_PLACE_MOMENTUM)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=constants.TRAINING_PLACE_LR_SCHEDULER_SIZE, gamma=constants.TRAINING_PLACE_LR_SCHEDULER_GAMMA)
 
        kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_loader = torch.utils.data.DataLoader(RecordedAirSimDataLoader(datapath, locomotion=False, transform=self.preprocess), batch_size=constants.TRAINING_PLACE_BATCH, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(RecordedAirSimDataLoader(datapath, locomotion=False, transform=self.preprocess, validation=True), batch_size=constants.TRAINING_PLACE_BATCH, shuffle=True, **kwargs)
        data_loaders = { 'train': train_loader, 'val': val_loader }

        since = time.time()

        best_model_wts = self.siamesenet.state_dict()
        best_acc = 0.0

        for epoch in range(train_iterations):
            print('Epoch {}/{}'.format(epoch, train_iterations - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    exp_lr_scheduler.step()
                    self.siamesenet.train(True)  # Set model to training mode
                else:
                    self.siamesenet.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for data in tqdm(data_loaders[phase]):
                    # get the inputs
                    anchors, pairs, labels = data

                    # wrap them in Variable
                    if self.use_cuda:
                        anchors = Variable(anchors.cuda())
                        pairs = Variable(pairs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        anchors, pairs, labels = Variable(anchors), Variable(pairs), Variable(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = self.siamesenet(anchors, pairs)
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
                    best_model_wts = self.siamesenet.state_dict()
                    print (checkpoint_path, epoch)
                    self.save_model(checkpoint_path, epoch)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.siamesenet.load_state_dict(best_model_wts)
        return self.siamesenet

    def eval_siamese(self, datapath):
        kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
        data_loader = torch.utils.data.DataLoader(RecordedAirSimDataLoader(datapath, locomotion=False, transform=self.preprocess, validation=True), batch_size=constants.TRAINING_PLACE_BATCH, shuffle=True, **kwargs)

        running_corrects = 0
        for data in tqdm(data_loader):
            anchors, pairs, labels = data

            if self.use_cuda:
                anchors = Variable(anchors.cuda())
                pairs = Variable(pairs.cuda())
                labels = Variable(labels.cuda())
            else:
                anchors, pairs, labels = Variable(anchors), Variable(pairs), Variable(labels)

            outputs = self.siamesenet(anchors, pairs)
            if type(outputs) == tuple:
                outputs, _ = outputs
            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_acc = (float(running_corrects) / float(len(data_loaders[phase].dataset))) * 100.0
        print('Accuracy: {}'.format(epoch_acc))

    def train_siamese_online(self, wad, checkpoint_path, train_iterations):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, self.siamesenet.parameters())), lr=constants.TRAINING_PLACE_LR, momentum=constants.TRAINING_PLACE_MOMENTUM)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=constants.TRAINING_PLACE_LR_SCHEDULER_SIZE, gamma=constants.TRAINING_PLACE_LR_SCHEDULER_GAMMA)
 
        kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_dataset = OnlineVizDoomDataLoader(wad, locomotion=False, transform=self.array_preprocess)
        val_dataset = OnlineVizDoomDataLoader(wad, locomotion=False, transform=self.array_preprocess)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=constants.TRAINING_PLACE_BATCH, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=constants.TRAINING_PLACE_BATCH, shuffle=True, **kwargs)
        data_loaders = { 'train': train_loader, 'val': val_loader }

        since = time.time()

        best_model_wts = self.siamesenet.state_dict()
        best_acc = 0.0

        for epoch in range(train_iterations):
            train_dataset.collect()
            val_dataset.collect()

            print('Epoch {}/{}'.format(epoch, train_iterations - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    exp_lr_scheduler.step()
                    self.siamesenet.train(True)  # Set model to training mode
                else:
                    self.siamesenet.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for data in tqdm(data_loaders[phase]):
                    # get the inputs
                    anchors, pairs, labels = data

                    # wrap them in Variable
                    if self.use_cuda:
                        anchors = Variable(anchors.cuda())
                        pairs = Variable(pairs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        anchors, pairs, labels = Variable(anchors), Variable(pairs), Variable(labels)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = self.siamesenet(anchors, pairs)
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
                    best_model_wts = self.siamesenet.state_dict()
                    print (checkpoint_path, epoch)
                    self.save_model(checkpoint_path, epoch)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.siamesenet.load_state_dict(best_model_wts)
        return self.siamesenet

    def save_model(self, checkpoint_path, epoch):
        print ("Saving a new checkpoint on epoch {}".format(epoch+1))
        state = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
        }
        if (constants.PLACE_TOP_MODEL == constants.PLACE_TOP_SIAMESE):
            top_network_name = 'siamese'
        else: # triplet
            top_network_name = 'triplet' 
        model_name = 'place_{}_checkpoint_{}.pth'.format(top_network_name, epoch)
        torch.save(state, os.path.join(checkpoint_path, model_name))
