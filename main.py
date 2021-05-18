import os
import sys
import copy
import random
import argparse
import datetime
import numpy as np
import scipy.stats as stats
import torch
import torch.utils as utils
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

from tqdm import tqdm
from torchvision import transforms, datasets
from torchvision.utils import save_image
from sklearn.metrics import confusion_matrix, accuracy_score

from vae import VAE
from iicnet import IICNet

import utils as myutils



parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=64)
parser.add_argument('-e', '--epochs', type=int, default=900)
parser.add_argument('-j', '--n-workers', type=int, default=2)
parser.add_argument('-a', '--alpha', type=float, default=1.0)
parser.add_argument('--n-heads', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('-z', '--z-dim', type=int, default=64)
parser.add_argument('--out', type=str, default='latest')
parser.add_argument('--interval', type=int, default=10)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

ds = 'ham'
if ds == 'ham':
    in_channels = 3
    n_classes = 2
    img_width = 256
    img_crop_width = 224
    norm_mean = (0.7385317, 0.5376959, 0.56187296)
    norm_std = (0.15286525, 0.1630482, 0.18261696)
elif ds == 'mnist':
    in_channels = 1
    n_classes = 10
    img_width = 28
    img_crop_width = 26
    norm_mean = (0.5,)
    norm_std = (0.5,)
    args.interval = 1

normalize = transforms.Normalize(norm_mean, norm_std)

train_transforms = transforms.Compose([
    transforms.Resize(img_width),
    #transforms.CenterCrop(img_crop_width),
    transforms.RandomCrop(img_crop_width),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    #transforms.Grayscale(),
    transforms.ToTensor(),
    #transforms.Lambda(myutils.custom_greyscale_to_tensor),
    normalize
])

test_transforms = transforms.Compose([
    transforms.Resize(img_width),
    transforms.CenterCrop(img_crop_width),
    #transforms.Grayscale(),
    transforms.ToTensor(),
    #transforms.Lambda(myutils.custom_greyscale_to_tensor),
    normalize
])

if ds == 'ham':
    train_ds = datasets.ImageFolder(
        './data/train',
        train_transforms
    )
    test_ds = datasets.ImageFolder(
    './data/test',
    test_transforms
    )
elif ds == 'mnist':
    train_ds = datasets.MNIST('./data', train=True, transform=train_transforms, download=True)
    test_ds = datasets.MNIST('./data', train=False, transform=test_transforms, download=True)

kwargs = {
    'num_workers': args.n_workers,
    'pin_memory': True
} if use_cuda else {}

train_loader = data.DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    **kwargs
)

test_loader = data.DataLoader(
    test_ds,
    batch_size=256,

)

model = IICNet(in_channels=in_channels, n_classes=n_classes, n_heads=args.n_heads, pretrained=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5), int(args.epochs*0.8)], gamma=0.1) # 10 < 20 < 40
#scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=args.lr*1e-3)


def compute_joint(x_out, x_tf_out):
    p_ij = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)
    p_ij = p_ij.sum(dim=0)
    p_ij = (p_ij + p_ij.t()) / 2.
    p_ij = p_ij / p_ij.sum()
    return p_ij


def IID_loss(x_out, x_tf_out, EPS=sys.float_info.epsilon):
    _, n_classes = x_out.size()
    p_ij = compute_joint(x_out, x_tf_out)

    p_i = p_ij.sum(dim=1).view(n_classes, 1).expand(n_classes, n_classes)
    p_j = p_ij.sum(dim=0).view(1, n_classes).expand(n_classes, n_classes)

    p_ij = torch.where(p_ij < EPS, torch.tensor([EPS], device=p_ij.device), p_ij)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)

    alpha = args.alpha
    loss = -1 * (p_ij * (torch.log(p_ij) - alpha * torch.log(p_i) - alpha * torch.log(p_j))).sum()
    return loss


def train(epoch):
    model.train()

    train_loss = 0
    train_loss_oc = 0
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data_tf = myutils.my_transforms(data, img_crop_width)

        data = data.to(device)
        data_tf = data_tf.to(device)

        optimizer.zero_grad()

        output, output_overclustering = model(data)
        output_tf, output_tf_overclustering = model(data_tf)

        loss1 = 0
        loss2 = 0
        for i in range(args.n_heads):
            loss1 += IID_loss(output[i], output_tf[i])
            loss2 += IID_loss(output_overclustering[i], output_tf_overclustering[i])
        
        ### loss2 range
        loss1 /= args.n_heads
        loss2 /= args.n_heads
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        train_loss += loss1
        train_loss_oc += loss2
        total_loss += loss
    
    # total
    print('Train ==> Loss: {:.4f}'.format(total_loss))
    return train_loss, train_loss_oc


def test(epoch):
    model.eval()
    
    out_targs = [[] for _ in range(args.n_heads)]
    ref_targs = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            
            output, _ = model(data)

            for i in range(args.n_heads):
                out_targs[i].extend(output[i].argmax(dim=1).cpu().detach().clone())
            ref_targs.extend(target.cpu().detach().clone())

    accs = np.array([accuracy_score(ref_targs, out) for out in out_targs])
    accs = np.where(accs > 1-accs, accs, 1-accs) * 100.
    best_acc =  accs.max()
    best_output_index = accs.argmax()
   
    print('Test ==> Accuray: {}'.format(accs))
    
    if epoch % args.interval == 0:
        np.set_printoptions(suppress=True)
        print(confusion_matrix(ref_targs, out_targs[best_output_index]))

    return best_acc


if __name__ == '__main__':
    os.makedirs('./result/', exist_ok=True)

    best_acc = 0
    best_model = None
    for epoch in range(1, args.epochs + 1):
        print('\nEpoch: {} | best_acc = {:.3f}'.format(epoch, best_acc))

        train(epoch)
        acc = test(epoch)

        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model)

    torch.save(best_model, './result/acc_{}.pth'.format(best_acc))
