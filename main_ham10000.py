from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
#from torchvision.models import resnet101, ResNet101_Weights
import numpy as np
import pandas as pd

from HAM_preprocess import HAM10000
from projected_gradient_descent import projected_gradient_descent

from llr import locally_linearity_regularization
from tulip import tulip_loss
from utils import get_optimizer, get_loss, get_scheduler, CustomTensorDataset

#used batch=64 lr=0.00001 for llr
parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-HAM-VGG',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--loss', default='advbeta',
                    help='[standard | llr | tulip]')

args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

data_dir = '/content/data/HAM10000/'
df_train = pd.read_csv(data_dir + 'train_data.csv')
df_val = pd.read_csv(data_dir + 'val_data.csv')


input_size = 224

norm_mean = [0.7630392, 0.5456477, 0.57004845]
norm_std = [0.1409286, 0.15261266, 0.16997074]
# define the transformation of the train images.
train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                        transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
# define the transformation of the val images.
val_transform = transforms.Compose([transforms.Resize((input_size,input_size)), transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)])

# Define the training set using the table train_df and using our defined transitions (train_transform)
training_set = HAM10000(df_train, transform=train_transform)
train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
# Same for the validation set:
test_set = HAM10000(df_val, transform=val_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

def train(args, model, device, train_loader, optimizer, epoch):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # calculate robust loss
        if args.loss == "standard":

            loss_fn = nn.CrossEntropyLoss(reduction="sum")

            loss = loss_fn(model(data), target)


        elif 'llr' in args.loss:
            if 'llr65' in args.loss:
                lambd, mu = 6.0, 5.0
            elif 'llr36' in args.loss:
                lambd, mu = 3.0, 6.0
            else:
                #lambd, mu = 4.0, 3.0
                lambd, mu = 3.0, 6.0

            if 'sllr' in args.loss:
                version = "sum"
            else:
                version = None

            epsilon = 0.031

            loss_fn = nn.CrossEntropyLoss(reduction="sum")

            norm = np.inf
            #norm = 2

            outputs, loss = locally_linearity_regularization(
                model, loss_fn, data, target, norm=norm, optimizer=optimizer,
                step_size=epsilon/2, epsilon=epsilon, perturb_steps=2,
                lambd=lambd, mu=mu, version=version
            )

        elif 'tulip' in args.loss:

            if 'tulipem1' in args.loss:
                lambd = 1e-1
            elif 'tulipem2' in args.loss:
                lambd = 1e-2
            elif 'tulip0' in args.loss:
                lambd = 0
            else:
                lambd = 1

            if 'ssem1' in args.loss:
                step_size = 1e-1
            elif 'ssem2' in args.loss:
                step_size = 1e-2
            elif 'ssem3' in args.loss:
                step_size = 1e-3
            else:
                step_size = 1e-0

            loss_fn = nn.CrossEntropyLoss(reduction="none")
            
            outputs, loss = tulip_loss(model, loss_fn, data, target,
                    step_size=step_size, lambd=lambd)

        elif 'advbeta' in args.loss:

            epsilon = 0.031

            norm = np.inf

            loss_fn = nn.CrossEntropyLoss(reduction="sum")

            advx = projected_gradient_descent(model, data, y=target,
                    clip_min=0, clip_max=1,
                    eps_iter=epsilon/5,
                    eps=epsilon, norm=norm, nb_iter=10)

            if 'beta.5' in args.loss:
                beta = 0.5
            elif 'beta8' in args.loss:
                beta = 8.
            elif 'beta4' in args.loss:
                beta = 4.
            elif 'beta2' in args.loss:
                beta = 2.
            else:
                beta = 1.

            outputs = model(advx)
            adv_loss = loss_fn(outputs, target)
            loss = loss_fn(model(data), target) + beta * adv_loss

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 5:
        lr = args.lr * 0.1
    if epoch >= 10:
        lr = args.lr * 0.01
    if epoch >= 15:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model, ResNet101() can be also used here for training
    # model = resnet101(weights=ResNet101_Weights.DEFAULT)
    model = models.vgg16(pretrained=True)
    #model = models.vgg16()
    #model.fc = nn.Linear(2048, 43)
    model.classifier[6] = nn.Linear(4096, 7)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        train_loss, train_accuracy = eval_train(model, device, train_loader)
        test_loss, test_accuracy = eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-vgg-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-vgg-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()

