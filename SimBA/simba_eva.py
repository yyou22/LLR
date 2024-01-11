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

from torch.autograd import Variable

import torchvision.transforms as trans
from torchvision import transforms as T

parser = argparse.ArgumentParser(description='Evaluate model accuracy on adversarial images')
parser.add_argument('--saved_file_path', type=str, default='./save_ham/pixel_vgg_1103_5000_96_0.2000_rand.pth', help='Path to the saved adversarial images')
parser.add_argument('--image_file_path', type=str, default='./save_ham/images_vgg_1103_96.pth', help='Path to the sampled images and labels')
parser.add_argument('--model_checkpoint', type=str, default='../Checkpoints/beta.5.pt', help='Path to the model checkpoint')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA')

args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

data_dir = '/content/data/HAM10000/'
df_train = pd.read_csv(data_dir + 'train_data.csv')
df_val = pd.read_csv(data_dir + 'val_data.csv')


norm_mean = [0.7630392, 0.5456477, 0.57004845]
norm_std = [0.1409286, 0.15261266, 0.16997074]

normalize = T.Normalize(norm_mean, norm_std)

normalize_ = trans.Compose([
	trans.Resize((224, 224)),
	trans.Normalize(norm_mean, norm_std)])

#the standard model uses 224 by 224 for prediction, the adv model uses 96 by 96
HAM_TRANSFORM = trans.Compose([
	trans.Resize((224, 224)),
	trans.ToTensor(),
	trans.Normalize(norm_mean, norm_std)])

testset = HAM10000(df_val, transform=HAM_TRANSFORM)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

def calculate_accuracy2(model, device, test_loader):

	model.eval()
	natural_err_total = 0

	for data, target in test_loader:
		data, target = data.to(device), target.to(device)
		# pgd attack
		X, y = Variable(data, requires_grad=True), Variable(target)
		out = model(X)
		err_natural = (out.data.max(1)[1] != y.data).float().sum()
		natural_err_total += err_natural
	print('natural_err_total: ', natural_err_total)

def calculate_accuracy(model, adv_images, true_labels, batch_size, device):
	#correct = 0
	#total = 0
	#with torch.no_grad():
		#for i in range(0, len(adv_images), batch_size):
			#images_batch = adv_images[i:i + batch_size].to(device)
			#labels_batch = true_labels[i:i + batch_size].to(device)
			#outputs = model(normalize(images_batch))
			#_, predicted = torch.max(outputs.data, 1)
			#total += labels_batch.size(0)
			#correct += (predicted == labels_batch).sum().item()
	#accuracy = 100 * correct / total
	#return accuracy

	accu_total = 0
	
	with torch.no_grad():
		for i in range(0, len(adv_images), batch_size):
			images_batch = adv_images[i:i + batch_size].to(device)
			labels_batch = true_labels[i:i + batch_size].to(device)
			#print(images_batch.shape)
			#print(labels_batch.shape)
			out = model(normalize(images_batch))
			accu = (out.data.max(1)[1] == labels_batch.to(device)).float().sum()

			#print(accu)
			
			accu_total += accu

	return accu_total / len(adv_images)

def main():

	# Load the saved data
	saved_data = torch.load(args.saved_file_path, map_location=device)

	saved_image = torch.load(args.image_file_path, map_location=device)

	adv_images = saved_data['adv']  # Adversarial images

	nat_images = saved_image['images']
	true_labels = saved_image['labels']  # True labels
	true_labels_adv = true_labels[:-3]

	print(nat_images.shape)

	#print(true_labels)

	labels_numpy = true_labels.cpu().detach().numpy()

	# Use numpy's unique function to count occurrences
	unique, counts = np.unique(labels_numpy, return_counts=True)

	# Print the counts for each unique value
	for label, count in zip(unique, counts):
		print(f"Label {label}: {count} occurrences")

	succs = saved_data['succs']
	print(succs.shape)

	print(adv_images.shape)

	plt.imshow(adv_images.cpu().detach().numpy()[0].transpose(1, 2, 0))
	plt.show()

	plt.imshow(adv_images.cpu().detach().numpy()[1].transpose(1, 2, 0))
	plt.show()

	plt.imshow(adv_images.cpu().detach().numpy()[2].transpose(1, 2, 0))
	plt.show()

	plt.imshow(nat_images.cpu().detach().numpy()[0].transpose(1, 2, 0))
	plt.show()

	plt.imshow(nat_images.cpu().detach().numpy()[1].transpose(1, 2, 0))
	plt.show()

	plt.imshow(nat_images.cpu().detach().numpy()[2].transpose(1, 2, 0))
	plt.show()

	# Load the model
	model = models.vgg16()
	model.classifier[6] = nn.Linear(4096, 7)

	model.to(device)
	model.load_state_dict(torch.load(args.model_checkpoint))
	model.eval()

	# Calculate accuracy
	accuracy = calculate_accuracy(model, adv_images, true_labels_adv, args.batch_size, device)
	print(f'Accuracy on adversarial images: {accuracy}%')

	#calculate_accuracy2(model, device, test_loader)

if __name__ == "__main__":
	main()

