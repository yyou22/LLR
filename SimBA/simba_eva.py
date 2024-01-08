import torch
import torchvision.models as models
import torch.nn as nn
import argparse

import torch
import torchvision.models as models
import torch.nn as nn
import argparse

from HAM_preprocess import HAM10000
import utils

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Evaluate model accuracy on adversarial images')
parser.add_argument('--saved_file_path', type=str, default='./save_ham/pixel_vgg_1000_10000_32_0.2000_rand.pth', help='Path to the saved adversarial images and labels')
parser.add_argument('--model_checkpoint', type=str, default='../Checkpoints/standard.pt', help='Path to the model checkpoint')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size for evaluation')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA')

args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

data_dir = '/content/data/HAM10000/'
df_train = pd.read_csv(data_dir + 'train_data.csv')
df_val = pd.read_csv(data_dir + 'val_data.csv')

testset = HAM10000(df_val, transform=utils.HAM_TRANSFORM)

def calculate_accuracy(model, adv_images, true_labels, batch_size, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(adv_images), batch_size):
            images_batch = adv_images[i:i + batch_size].to(device)
            labels_batch = true_labels[i:i + batch_size].to(device)
            outputs = model(images_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def main():

    # Load the saved data
    saved_data = torch.load(args.saved_file_path, map_location=device)
    adv_images = saved_data['adv']  # Adversarial images
    #true_labels = saved_data['labels']  # True labels

    succs = saved_data['succs']
    #print(succs)

    print(adv_images.shape)

    plt.imshow(adv_images.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()

    plt.imshow(adv_images.cpu().detach().numpy()[1].transpose(1, 2, 0))
    plt.show()

    plt.imshow(adv_images.cpu().detach().numpy()[2].transpose(1, 2, 0))
    plt.show()

    # Load the model
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 7)
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # Calculate accuracy
    #accuracy = calculate_accuracy(model, adv_images, true_labels, args.batch_size, device)
    #print(f'Accuracy on adversarial images: {accuracy}%')

if __name__ == "__main__":
    main()

