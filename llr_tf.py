import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def pgd_attack(model, x, y, epsilon, alpha, num_steps, random_start):
    """ Implements the PGD attack """
    if random_start:
        x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    for _ in range(num_steps):
        x.requires_grad_()
        with torch.enable_grad():
            loss = F.cross_entropy(model(x), y)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + alpha * torch.sign(grad.detach())
        x = torch.min(torch.max(x, original_x - epsilon), original_x + epsilon)
        x = torch.clamp(x, 0, 1)
    return x

def locally_linearity_regularization(model, x, y, epsilon=0.031, alpha=0.007, num_steps=10, lambd=4.0, mu=3.0, smoothing_factor=2.0):
    model.eval()
    batch_size = x.size(0)

    # Convert labels to one-hot
    y_onehot = torch.eye(10)[y].to(x.device)

    # Calculate gradient of natural loss
    x.requires_grad_(True)
    natural_output = model(x)
    natural_loss = F.cross_entropy(natural_output, y_onehot, reduction='sum')
    grad_natural_loss = torch.autograd.grad(natural_loss, [x])[0]
    
    # Perform PGD attack to find adversarial examples
    adv_x = pgd_attack(model, x, y, epsilon, alpha, num_steps, random_start=True)

    # Calculate loss at adversarial examples
    adv_output = model(adv_x)
    adv_loss = F.cross_entropy(adv_output, y_onehot, reduction='sum')

    # Calculate the difference in losses and gradients
    diff_loss = adv_loss - natural_loss
    dot_product = torch.sum((adv_x - x) * grad_natural_loss.detach(), dim=(1,2,3))
    attack_loss = torch.sign(diff_loss) * diff_loss + smoothing_factor * torch.sign(dot_product) * dot_product

    # Final loss calculation
    final_loss = natural_loss + lambd * torch.mean(attack_loss)

    return final_loss
