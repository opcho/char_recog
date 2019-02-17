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
import custom_datasets
import time
import copy
import string
import utils
import argparse

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--test-batch-size', type = int, default = 16)
parser.add_argument('--epochs', type = int, default = 10)
parser.add_argument('--input_channel', type = int, default = 3)
parser.add_argument('--lr', type = float, default = 0.001)

parser.add_argument('--model', type = str, default = "stn_squeezenet_aug")
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
    # torchvision.transforms.RandomRotation(degrees=(-30, 30)),
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_datasets=torch.utils.data.ConcatDataset([
    custom_datasets.Chars74k(csv_file='data/train.csv',root_dir='data',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=train_transform,),
    custom_datasets.Chars74k(csv_file='data/validation.csv',root_dir='data',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=train_transform,),
    # custom_datasets.Chars74k(csv_file='data/augment_-10/augment_-10.csv',root_dir='data/augment_-10',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=train_transform,),
    # custom_datasets.Chars74k(csv_file='data/augment_-20/augment_-20.csv',root_dir='data/augment_-20',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=train_transform,),
    # custom_datasets.Chars74k(csv_file='data/augment_-30/augment_-30.csv',root_dir='data/augment_-30',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=train_transform,),
    # custom_datasets.Chars74k(csv_file='data/augment_10/augment_10.csv',root_dir='data/augment_10',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=train_transform,),
    # custom_datasets.Chars74k(csv_file='data/augment_20/augment_20.csv',root_dir='data/augment_20',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=train_transform,),
    # custom_datasets.Chars74k(csv_file='data/augment_30/augment_30.csv',root_dir='data/augment_30',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=train_transform,),
])

# Training dataset
train_loader = torch.utils.data.DataLoader(
    train_datasets,
    batch_size=args.batch_size, shuffle=True, num_workers=4)
# Test dataset
test_loader = torch.utils.data.DataLoader(
    custom_datasets.Chars74k(csv_file='data/train.csv',root_dir='data',CLASSES=args.CLASSES, target_transform=utils.get_index_of_label, transform=test_transform,),
    batch_size=args.batch_size, shuffle=True, num_workers=4)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class Net(nn.Module):
    def __init__(self, num_classes, feature_extract,input_channel=3, use_pretrained=True):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(320, 50)
        # self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            # origin
            # nn.Conv2d(input_channel, 8, kernel_size=7),
            # nn.MaxPool2d(2, stride=2),
            # nn.ReLU(True),
            # nn.Conv2d(8, 10, kernel_size=5),
            # nn.MaxPool2d(2, stride=2),
            # nn.ReLU(True),
            nn.Conv2d(input_channel, 8, kernel_size=7, stride=2,),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=2,),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(8, 16, kernel_size=5, stride=2, ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2,),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 *5*5, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.pretrain_model=models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(self.pretrain_model, feature_extract)
        self.pretrain_model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.pretrain_model.num_classes = num_classes

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 *5*5)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        x=self.pretrain_model(x)
        return x
        # return F.log_softmax(x, dim=1)

def train(model,train_loader, test_loader, criterion, optimizer,  num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    val_acc_history = []
    train_acc_history = []

    for epoch in range(num_epochs):
        # train
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(output, labels)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # statistics
            pred = output.max(1, keepdim=True)[1]
            running_corrects += pred.eq(labels.view_as(pred)).sum().item()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)\

        print('Epoch [{}/{}] '.format(epoch + 1, num_epochs),end='')
        print('{}\tLoss: {:.2f}\tAcc: {:.2f}'.format('Train', epoch_loss, epoch_acc))
        train_acc_history.append(epoch_acc)

        # test
        print('Epoch [{}/{}] '.format(epoch + 1, num_epochs),end='')
        val_acc=test(model,test_loader,criterion)
        val_acc_history.append(val_acc)

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

    return model, train_acc_history,val_acc_history

def test(model,dataloaders, criterion, ):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in dataloaders:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            # test_loss +=criterion(output, target, size_average=False).item()
            test_loss +=criterion(output, target, ).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(dataloaders.dataset)
        print('{}\tLoss: {:.2f}\tAcc: {:.2f}'.format('Test', test_loss, correct/len(dataloaders.dataset)))
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

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

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

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # origin
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    # criterion=F.nll_loss

    best_model, train_hist, tets_hist=train(model,train_loader, test_loader, criterion, optimizer,  args.epochs)

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
    plt.xticks(np.arange(1, args.epochs + 1, 5))
    plt.legend()
    plt.savefig(args.model+'.png')
    plt.show()


    # model = Net(args.num_classes,args.feature_extract,).to(device)
    # model.load_state_dict(torch.load(args.model))
    # test(model,test_loader,criterion)