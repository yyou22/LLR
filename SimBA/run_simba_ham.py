import torch
import torchvision.datasets as dset
from torchvision import datasets
import torchvision.transforms as transforms
import utils
import math
import random
import argparse
import os
import sys
#sys.path.append('pytorch-cifar')
#import models
import torch.nn as nn
import torch.nn.functional as F
from simba import SimBA
import torchvision.models as models

import numpy as np
import pandas as pd

from HAM_preprocess import HAM10000
import torchvision.transforms as trans

parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
#parser.add_argument('--data_root', type=str, required=True, help='root directory of imagenet data')
parser.add_argument('--result_dir', type=str, default='save_ham', help='directory for saving results')
parser.add_argument('--sampled_image_dir', type=str, default='save_ham', help='directory to cache sampled images')
parser.add_argument('--model', type=str, default='vgg', help='type of base model to use')
#parser.add_argument('--model_ckpt', type=str, required=True, help='model checkpoint location')
parser.add_argument('--num_runs', type=int, default=1103, help='number of image samples')
parser.add_argument('--batch_size', type=int, default=50, help='batch size for parallel runs')
parser.add_argument('--num_iters', type=int, default=5000, help='maximum number of iterations, 0 for unlimited')
parser.add_argument('--log_every', type=int, default=10, help='log every n iterations')
parser.add_argument('--epsilon', type=float, default=0.2, help='step size per iteration')
parser.add_argument('--linf_bound', type=float, default=0.0, help='L_inf bound for frequency space attack')
parser.add_argument('--freq_dims', type=int, default=96, help='dimensionality of 2D frequency space')
parser.add_argument('--order', type=str, default='rand', help='(random) order of coordinate selection')
parser.add_argument('--stride', type=int, default=7, help='stride for block order')
parser.add_argument('--targeted', action='store_true', help='perform targeted attack')
parser.add_argument('--pixel_attack', action='store_true', help='attack in pixel space')
parser.add_argument('--save_suffix', type=str, default='', help='suffix appended to save file')
parser.add_argument('--model-checkpoint', default='../Checkpoints/beta.5.pt',
                    help='directory of model for saving checkpoint')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')


args = parser.parse_args()

if not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)
if not os.path.exists(args.sampled_image_dir):
    os.mkdir(args.sampled_image_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

data_dir = '/content/data/HAM10000/'
df_train = pd.read_csv(data_dir + 'train_data.csv')
df_val = pd.read_csv(data_dir + 'val_data.csv')

# load model and dataset
#model = getattr(models, args.model)().cuda()
#model = torch.nn.DataParallel(model)
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, 7)
model = model.to(device)
model.load_state_dict(torch.load(os.path.join(args.model_checkpoint)))
#checkpoint = torch.load(args.model_ckpt)
#model.load_state_dict(checkpoint['net'])
model.eval()

#image_size = 32
image_size = 96
#testset = dset.CIFAR10(root=args.data_root, train=False, download=True, transform=utils.CIFAR_TRANSFORM)

testset = HAM10000(df_val, transform=trans.Compose([
    trans.Resize((96, 96)),
    trans.ToTensor()]))
#test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

#attacker = SimBA(model, 'cifar', image_size)
attacker = SimBA(model, 'HAM', image_size)

# load sampled images or sample new ones
# this is to ensure all attacks are run on the same set of correctly classified images
batchfile = '%s/images_%s_%d_96.pth' % (args.sampled_image_dir, args.model, args.num_runs)
if os.path.isfile(batchfile):
    checkpoint = torch.load(batchfile)
    images = checkpoint['images']
    labels = checkpoint['labels']
else:
    #images = torch.zeros(args.num_runs, 3, image_size, image_size)
    #labels = torch.zeros(args.num_runs).long()
    #preds = labels + 1
    #while preds.ne(labels).sum() > 0:
        #idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)]
        #for i in list(idx):
            #images[i], labels[i] = testset[random.randint(0, len(testset) - 1)]
        #preds[idx], _ = utils.get_preds(model, images[idx], 'cifar', batch_size=args.batch_size)
    #torch.save({'images': images, 'labels': labels}, batchfile)

    # Allocate lists to store images and labels
    images = torch.zeros(args.num_runs, 3, image_size, image_size)
    labels = torch.zeros(args.num_runs).long()

    # Iterate through the testset and save images and labels
    for i in range(0, args.num_runs):
        images[i], labels[i] = testset[i]

    # Save the images and labels
    torch.save({'images': images, 'labels': labels}, batchfile)

if args.order == 'rand':
    n_dims = 3 * args.freq_dims * args.freq_dims
else:
    n_dims = 3 * image_size * image_size
if args.num_iters > 0:
    max_iters = int(min(n_dims, args.num_iters))
else:
    max_iters = int(n_dims)
N = int(math.floor(float(args.num_runs) / float(args.batch_size)))
for i in range(N):
    upper = min((i + 1) * args.batch_size, args.num_runs)
    images_batch = images[(i * args.batch_size):upper]
    labels_batch = labels[(i * args.batch_size):upper]
    # replace true label with random target labels in case of targeted attack
    if args.targeted:
        labels_targeted = labels_batch.clone()
        while labels_targeted.eq(labels_batch).sum() > 0:
            labels_targeted = torch.floor(10 * torch.rand(labels_batch.size())).long()
        labels_batch = labels_targeted
    adv, probs, succs, queries, l2_norms, linf_norms = attacker.simba_batch(
        images_batch, labels_batch, max_iters, args.freq_dims, args.stride, args.epsilon, linf_bound=args.linf_bound,
        order=args.order, targeted=args.targeted, pixel_attack=args.pixel_attack, log_every=args.log_every)
    if i == 0:
        all_adv = adv
        all_probs = probs
        all_succs = succs
        all_queries = queries
        all_l2_norms = l2_norms
        all_linf_norms = linf_norms
    else:
        all_adv = torch.cat([all_adv, adv], dim=0)
        all_probs = torch.cat([all_probs, probs], dim=0)
        all_succs = torch.cat([all_succs, succs], dim=0)
        all_queries = torch.cat([all_queries, queries], dim=0)
        all_l2_norms = torch.cat([all_l2_norms, l2_norms], dim=0)
        all_linf_norms = torch.cat([all_linf_norms, linf_norms], dim=0)
    if args.pixel_attack:
        prefix = 'pixel'
    else:
        prefix = 'dct'
    if args.targeted:
        prefix += '_targeted'
    savefile = '%s/%s_%s_%d_%d_%d_%.4f_%s%s.pth' % (
        args.result_dir, prefix, args.model, args.num_runs, args.num_iters, args.freq_dims, args.epsilon, args.order, args.save_suffix)
    torch.save({'adv': all_adv, 'probs': all_probs, 'succs': all_succs, 'queries': all_queries,
                'l2_norms': all_l2_norms, 'linf_norms': all_linf_norms}, savefile)