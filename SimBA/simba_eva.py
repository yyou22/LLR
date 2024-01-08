import torch
import torchvision.models as models
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser(description='Evaluate model accuracy on adversarial images')
parser.add_argument('--saved_file_path', type=str, default='./save_ham/pixel_vgg_1000_10000_32_0.2000_rand.pth', help='Path to the saved adversarial images and labels')
parser.add_argument('--model_checkpoint', type=str, default='../Checkpoints/standard.pt', help='Path to the model checkpoint')
args = parser.parse_args()

def main(args):
    # Load the saved data
    saved_data = torch.load(args.saved_file_path)
    adv_images = saved_data['adv']  # Adversarial images
    true_labels = saved_data['labels']  # True labels

    # Load your model
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 7)
    model.load_state_dict(torch.load(args.model_checkpoint))
    model.eval()

    # Evaluate accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(adv_images)):
            outputs = model(adv_images[i].unsqueeze(0))
            _, predicted = torch.max(outputs.data, 1)
            total += 1
            correct += (predicted == true_labels[i]).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on adversarial images: {accuracy}%')

if __name__ == "__main__":

    main(args)
