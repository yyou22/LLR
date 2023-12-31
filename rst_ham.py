import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms as T

import matplotlib.pyplot as plt
import torchvision.models as models
import pandas as pd
import numpy as np

norm_mean = [0.7630392, 0.5456477, 0.57004845]
norm_std = [0.1409286, 0.15261266, 0.16997074]

inv_mean = [-0.7630392/0.1409286, -0.5456477/0.15261266, -0.57004845/0.16997074]
inv_std = [1/0.1409286, 1/0.15261266, 1/0.16997074]

normalize = T.Normalize(norm_mean, norm_std)
inv_normalize = T.Normalize(inv_mean, inv_std)

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def rst_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example

    x_nat_inv = inv_normalize(x_natural)

    x_adv = x_nat_inv.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(normalize(x_adv)), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_nat_inv - epsilon), x_nat_inv + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    #print(x_adv.shape)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    #showing image
    #img = x_adv.cpu().detach().numpy()[0].transpose(1, 2, 0)
    # Undo the normalization
    #img = (img * norm_std) + norm_mean  # Revert the normalization
    # Clip the values to be between 0 and 1
    #img = np.clip(img, 0, 1)
    # Display the image
    #plt.imshow(img)
    #plt.savefig('./x_adv.png')
    #plt.show()

    #img = inv_normalize(x_natural).cpu().detach().numpy()[0].transpose(1, 2, 0)
    # Undo the normalization
    #img = (img * norm_std) + norm_mean  # Revert the normalization
    # Clip the values to be between 0 and 1
    #img = np.clip(img, 0, 1)
    # Display the image
    #plt.imshow(img)
    #plt.savefig('./x_nat.png')
    #plt.show()

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y, reduction='sum')
    #loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    #F.softmax(model(x_natural), dim=1))
    loss_robust = F.cross_entropy(model(normalize(x_adv)), y, reduction='sum')
    loss = loss_natural + beta * loss_robust

    #showing image
    #img = normalize(x_adv).cpu().detach().numpy()[0].transpose(1, 2, 0)
    # Undo the normalization
    #img = (img * norm_std) + norm_mean  # Revert the normalization
    # Clip the values to be between 0 and 1
    #img = np.clip(img, 0, 1)
    # Display the image
    #plt.imshow(img)
    #plt.savefig('./x_adv.png')
    #plt.show()

    #img = x_natural.cpu().detach().numpy()[0].transpose(1, 2, 0)
    # Undo the normalization
    #img = (img * norm_std) + norm_mean  # Revert the normalization
    # Clip the values to be between 0 and 1
    #img = np.clip(img, 0, 1)
    # Display the image
    #plt.imshow(img)
    #plt.savefig('./x_nat.png')
    #plt.show()

    #print('loss_natural', loss_natural)
    #print('loss_robust', loss_robust)

    return loss