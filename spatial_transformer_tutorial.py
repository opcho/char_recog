# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
import custom_dataset
import time
import copy
import string
import utils
import argparse
from PIL import Image
import os
import csv
import torchvision.utils as v_utils

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--test-batch-size', type = int, default = 16)
parser.add_argument('--epochs', type = int, default = 60)
parser.add_argument('--input_channel', type = int, default = 60)
parser.add_argument('--lr', type = float, default = 0.001)

parser.add_argument('--model', type = str, default = "stn_aug2_full")
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
parser.add_argument('--feature_extract', type = bool, default = True)
parser.add_argument('--num_classes', type = int, default = len(string.digits + string.ascii_uppercase + string.ascii_lowercase))
parser.add_argument('--img_height', type = int, default = 224)
parser.add_argument('--img_width', type = int, default = 224)
parser.add_argument('--CLASSES', type = str, default =string.digits + string.ascii_uppercase + string.ascii_lowercase)
args = parser.parse_args()

# origin
# train_transform=transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     torchvision.transforms.Resize((args.img_height, args.img_width)),
#     # torchvision.transforms.RandomRotation(degrees=(-30, 30)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])
# test_transform=transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     torchvision.transforms.Resize((args.img_height, args.img_width)),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])
train_transform=transforms.Compose([
    torchvision.transforms.Resize((args.img_height, args.img_width)),
    # torchvision.transforms.RandomRotation(degrees=(-20, 20)),
transforms.RandomAffine((-20,20), translate=(0.1,0.1),  shear=(-20,20),  fillcolor=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
test_transform=transforms.Compose([
    torchvision.transforms.Resize((args.img_height, args.img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
gtrain_transform=transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Resize((64,64)),
    # torchvision.transforms.RandomRotation(degrees=(-30, 30)),
    transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,)),
])
gtest_transform=transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Resize((64,64)),
    transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,)),
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_datasets=torch.utils.data.ConcatDataset([
    # witout msk
    custom_dataset.Chars74k(csv_file='good_train.csv', root_dir='data', CLASSES=args.CLASSES,
                             target_transform=utils.get_index_of_label, transform=train_transform,transform2=gtrain_transform, ),
    custom_dataset.Chars74k(csv_file='good_validation.csv', root_dir='data', CLASSES=args.CLASSES,
                             target_transform=utils.get_index_of_label, transform=train_transform,transform2=gtrain_transform, ),
    custom_dataset.Chars74k(csv_file='good_train.csv', root_dir='data', CLASSES=args.CLASSES,
                             target_transform=utils.get_index_of_label, transform=train_transform,transform2=gtrain_transform, ),
    custom_dataset.Chars74k(csv_file='good_validation.csv', root_dir='data', CLASSES=args.CLASSES,
                             target_transform=utils.get_index_of_label, transform=train_transform,transform2=gtrain_transform, ),

    # with msk
    # custom_dataset.Chars74k(csv_file='data/mskc_train/mskc_train.csv',root_dir='data/mskc_train',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=train_transform,),
    # custom_dataset.Chars74k(csv_file='data/mskc_validation/mskc_validation.csv',root_dir='data/mskc_validation',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=train_transform,),
    # custom_dataset.Chars74k(csv_file='data/augment_-10/augment_-10.csv',root_dir='data/augment_-10',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=train_transform,),
    # custom_dataset.Chars74k(csv_file='data/augment_-20/augment_-20.csv',root_dir='data/augment_-20',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=train_transform,),
    # custom_dataset.Chars74k(csv_file='data/augment_-30/augment_-30.csv',root_dir='data/augment_-30',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=train_transform,),
    # custom_dataset.Chars74k(csv_file='data/augment_10/augment_10.csv',root_dir='data/augment_10',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=train_transform,),
    # custom_dataset.Chars74k(csv_file='data/augment_20/augment_20.csv',root_dir='data/augment_20',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=train_transform,),
    # custom_dataset.Chars74k(csv_file='data/augment_30/augment_30.csv',root_dir='data/augment_30',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=train_transform,),
])
# Training dataset
train_loader = torch.utils.data.DataLoader(
    train_datasets,
    batch_size=args.batch_size, shuffle=True, num_workers=4)


# Test dataset
test_loader = torch.utils.data.DataLoader(
    # witout msk
    custom_dataset.Chars74k(csv_file='good_test.csv', root_dir='data', CLASSES=args.CLASSES,
                             target_transform=utils.get_index_of_label, transform=test_transform, transform2=gtest_transform, ),
    # with msk
    # custom_dataset.Chars74k(csv_file='data/mskc_test/mskc_test.csv',root_dir='data/mskc_test',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=test_transform,),
    batch_size=args.batch_size, shuffle=True, num_workers=4)


# val_datasets=torch.utils.data.ConcatDataset([
#     custom_dataset.Chars74k(csv_file='good_validation.csv', root_dir='data', CLASSES=args.CLASSES,
#                              target_transform=utils.get_index_of_label, transform=train_transform,transform2=gtrain_transform, ),
#     ])
# val_loader = torch.utils.data.DataLoader(
#     val_datasets,
#     batch_size=args.batch_size, shuffle=True, num_workers=4)


def convert_alphabet_index(index,CLASSES = args.CLASSES):
    if CLASSES[index].isupper():
        return CLASSES.index(CLASSES[index].lower())
    elif CLASSES[index].islower():
        return CLASSES.index(CLASSES[index].upper())
    else:
        return index


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def adjust_lr(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Net(nn.Module):
    def __init__(self, num_classes, feature_extract,input_channel=3, use_pretrained=True):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(num_classes, num_classes)
        self.bn1=nn.BatchNorm1d(num_classes)
        self.rl1=nn.ReLU(True)
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            # origin
            # nn.Conv2d(1, 4, kernel_size=7),
            # nn.MaxPool2d(2, stride=2),
            # nn.BatchNorm2d(4),
            # nn.ReLU(True),
            # nn.Conv2d(4, 8, kernel_size=5),
            # nn.MaxPool2d(2, stride=2),
            # nn.BatchNorm2d(8),
            # nn.ReLU(True),

            nn.Conv2d(input_channel, 8, kernel_size=5, stride=2,),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size=5, stride=2, ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 24, kernel_size=3, stride=2,),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            # nn.Linear(8 *144, 16),
            nn.Linear(24 *9, 16),
            nn.BatchNorm1d(16),
            # nn.ReLU(True),
            nn.Linear(16, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.pretrain_model=models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(self.pretrain_model, feature_extract)
        self.pretrain_model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.pretrain_model.num_classes = num_classes

    # Spatial transformer network forward function
    def stn(self, x,x2):
        xs = self.localization(x)
        # xs = xs.view(-1, 8 *144)
        xs = xs.view(-1, 24 *9)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x,x2):
        # transform the input
        x = self.stn(x,x2)

        # Perform the usual forward pass
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)

        x=self.pretrain_model(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.rl1(x)
        # return x
        return F.log_softmax(x, dim=1)

def train(model,train_loader, test_loader, criterion, optimizer,  num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    val_acc_history = []
    train_acc_history = []

    for epoch in range(num_epochs):
        adjust_lr(optimizer, epoch)
        running_loss = 0.0
        running_corrects = 0
        insensitive=0
        val_acc=0
        val_loss = 0
        val_coorect = 0
        val_insensitive=0
        val_top5=0
        top5=0
        validation_idx=0.2
        for batch_idx, (inputs, inputs2,labels) in enumerate(train_loader):
            inputs, inputs2, labels = inputs.to(device), inputs2.to(device), labels.to(device)
            if validation_idx*len(train_loader)> batch_idx and (validation_idx-0.2)*len(train_loader)<= batch_idx:
                phase='validation'
            else:
                phase = 'train'

            if epoch%5==0:
                validation_idx= 0.2 if validation_idx==1 else validation_idx+0.2

            if phase=='train':
                model.train()
                # zero the parameter gradients
                optimizer.zero_grad()

                output = model(inputs,inputs2)
                loss = criterion(output, labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

                # statistics
                pred = output.max(1, keepdim=True)[1]
                running_loss += loss.item() * inputs.size(0)
                # case sensitive top1 count
                running_corrects += pred.eq(labels.view_as(pred)).sum().item()
                # case insensitive top1 count
                for t, p in zip(labels, output.max(1)[1]):
                    if p.item() == t.item() or convert_alphabet_index(p.item()) == t.item():
                        insensitive += 1

                for t, p in zip(labels, output.topk(5)[1]):
                    for pch in p:
                        if pch.item() == t.item() :
                            top5 += 1
                            break
            else:
                # test

                model.eval()
                output = model(inputs,inputs2)
                val_loss += criterion(output, labels, ).item()
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                val_coorect += pred.eq(labels.view_as(pred)).sum().item()
                # case insensitive top1 count
                for t, p in zip(labels, output.max(1)[1]):
                    if p.item() == t.item() or convert_alphabet_index(p.item()) == t.item():
                        val_insensitive += 1

                for t, p in zip(labels, output.topk(3)[1]):
                    for pch in p:
                        if pch.item() == t.item():
                            val_top5 += 1
                            break
        train_batch=len(train_loader.dataset)*0.8
        val_batch = len(train_loader.dataset) * 0.2

        epoch_loss = running_loss / train_batch
        epoch_sensitive_acc = running_corrects / train_batch
        epoch_insensitive_acc = insensitive / train_batch
        epoch_top3_insensitive_acc = top5 / train_batch

        print('Epoch [{}/{}] '.format(epoch + 1, num_epochs),end='')
        print('{}\tLoss: {:.2f}\tAccT1: {:.2f}\tin_AccT1: {:.2f}\tAccT5: {:.2f}'.format('Train', epoch_loss, epoch_sensitive_acc,epoch_insensitive_acc,epoch_top3_insensitive_acc))
        train_acc_history.append(epoch_sensitive_acc)

        val_acc=val_coorect/val_batch
        val_acc_history.append(val_acc)
        print('Epoch [{}/{}] '.format(epoch + 1, num_epochs), end='')
        print('{}\tLoss: {:.2f}\tAccT1: {:.2f}\tin_AccT1: {:.2f}\tAccT5: {:.2f}'.format('Val', val_loss,
                                                                                        val_acc,
                                                                                        val_insensitive / val_batch,
                                                                                        val_top5 / val_batch))
        # deep copy the model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), args.model)
    print(args.model,'model saved')

    return model, train_acc_history,val_acc_history

def test(model,dataloaders, criterion, save_failed_img=False):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        insensitive=0
        top5=0
        failed_img_count=0

        for data,inputs2, target in dataloaders:
            data,inputs2, target = data.to(device), inputs2.to(device), target.to(device)
            output = model(data,inputs2)

            # sum up batch loss
            # test_loss +=criterion(output, target, size_average=False).item()
            test_loss +=criterion(output, target, ).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            # save fail image
            if save_failed_img and failed_img_count<=10:
                for d,t,p in zip(data,target, output.max(1)[1]):
                    if p.item()!=t.item():
                        failed_img_count+=1
                        # torchvision.transforms.ToPILImage(d).save('gt_{}_pred_{}.png'.format(args.CLASSES[t.item()],args.CLASSES[p.item()]))

                        v_utils.save_image(d.cpu().data, "./failed_{}_gt_{}_pred_{}.png".format(failed_img_count,args.CLASSES[p.item()],args.CLASSES[t.item()]))
            print('failed img saved')

            # case insensitive top1 count
            for t, p in zip(target, output.max(1)[1]):
                if p.item() == t.item() or convert_alphabet_index(p.item()) == t.item():
                    insensitive += 1

            for t, p in zip(target, output.topk(3)[1]):
                for pch in p:
                    if pch.item() == t.item() :
                        top5 += 1
                        break

        test_loss /= len(dataloaders.dataset)
        print('{}\tLoss: {:.2f}\tAccT1: {:.2f}\tin_AccT1: {:.2f}\tAccT5: {:.2f}'.format('Test', test_loss, correct/len(dataloaders.dataset),insensitive/len(dataloaders.dataset),top5/len(dataloaders.dataset)))
    return correct/len(dataloaders.dataset)


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def visualize_stn(model,dataloaders):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(dataloaders))[0].to(device)
        data2 = next(iter(dataloaders))[1].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data,data2).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')



if __name__ == "__main__":
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    model = Net(args.num_classes,args.feature_extract,).to(device)

    print('total parameters ',sum(p.numel() for p in model.parameters() if p.requires_grad))
    # for param in model.parameters():
    #     print(param.data)
    #     if param.requires_grad:
    #         print('params',param.numel())

    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()

    # origin
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    # criterion=F.nll_loss

    best_model, train_hist, tets_hist=train(model,train_loader, test_loader, criterion, optimizer,  args.epochs)
    # print(tets_hist)
    # with open( args.model+'_acc_record.csv', 'w') as writeFile:
    #     writer = csv.writer(writeFile)
    #     writer.writerows( tets_hist)
    #     print(args.model+'_acc_record.csv','saved')

    plt.ion()  # interactive mode
    # Visualize the STN transformation on some input batch
    visualize_stn(best_model,test_loader)

    plt.ioff()
    plt.savefig(args.model+'before&after.png')
    plt.show()

    ohist = [h for h in train_hist]
    shist = [h for h in tets_hist]
    plt.title("Train Accuracy vs. Test Accuracy")
    plt.xlabel(" Epochs")
    plt.ylabel(" Accuracy")
    plt.plot(range(1, args.epochs + 1), ohist, label="Train")
    plt.plot(range(1, args.epochs + 1), shist, label="Test")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(0, args.epochs + 1, 5))
    plt.legend()
    plt.savefig(args.model+'.png')
    plt.show()

    # test model
    print('test start')
    model = Net(args.num_classes,args.feature_extract,).to(device)
    model.load_state_dict(torch.load(args.model))
    test(model,test_loader,criterion,save_failed_img=True)

