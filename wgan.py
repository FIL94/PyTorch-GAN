import argparse
import os
import numpy as np
import sys
from datetime import datetime

from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from torch.autograd import Variable

import torch.nn as nn
import torch


parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="path to folder containing images")
parser.add_argument("--log_dir", type=str, default=os.getcwd(), help="directory to save results")
parser.add_argument("--pretrained_models", type=int, default=None, help="number epoch for loading models")
parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image samples")
opt = parser.parse_args()
print(opt)

os.makedirs(os.path.join(opt.log_dir, "trainig", "wgan", 'images'), exist_ok=True)
os.makedirs(os.path.join(opt.log_dir, "trainig", "wgan", 'checkpoints', 'generators'), exist_ok=True)
os.makedirs(os.path.join(opt.log_dir, "trainig", "wgan", 'checkpoints', 'discriminators'), exist_ok=True)
os.makedirs(os.path.join(opt.log_dir, "trainig", "wgan", 'metrics'), exist_ok=True)

loger = SummaryWriter(os.path.join(opt.log_dir, "trainig", "wgan", 'metrics',
                                   datetime.now().strftime('%Y_%m_%d %H_%M_%S')))

if os.path.exists(os.path.join(opt.log_dir, "trainig", "wgan", 'metrics', 'logs.txt')):
    if opt.pretrained_models is not None:
        with open(os.path.join(opt.log_dir, "trainig", "wgan", 'metrics', 'logs.txt'), 'r') as f:
            f.seek(0)
            lines = f.readlines()
            lines = lines[:int(opt.pretrained_models)]
            for line in lines:
                epoch, g_loss, d_loss = line.split(' ')
                loger.add_scalar("Generator loss", float(g_loss), int(epoch))
                loger.add_scalar("Discriminator loss", float(d_loss), int(epoch))
        with open(os.path.join(opt.log_dir, "trainig", "wgan", 'metrics', 'logs.txt'), 'a') as f:
            f.seek(0)
            f.truncate()
            f.writelines(lines)
    else:
        os.remove(os.path.join(opt.log_dir, "trainig", "wgan", 'metrics', 'logs.txt'))

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
print(f"Device: {'cuda' if cuda else 'cpu'}")


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()


# Configure data loader
class GirlsDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.images = os.listdir(path)
        self.transform = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.transform(read_image(os.path.join(self.path, self.images[item])).div(torch.tensor([255])))


girlsDataset = GirlsDataset(opt.path)
dataloader = torch.utils.data.DataLoader(
    girlsDataset,
    batch_size=opt.batch_size,
    shuffle=True
)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
if opt.pretrained_models is not None:
    try:
        generator.load_state_dict(torch.load(os.path.join(opt.log_dir, "trainig", "wgan", 'checkpoints', 'generators',
                                                          f'model_{opt.pretrained_models}.pt')))
        generator.train()
        optimizer_G.load_state_dict(torch.load(os.path.join(opt.log_dir, "trainig", "wgan", 'checkpoints', 'generators',
                                                            f'optimizer_{opt.pretrained_models}.pt')))
        discriminator.load_state_dict(torch.load(os.path.join(opt.log_dir, "trainig", "wgan", 'checkpoints', 'discriminators',
                                                              f'model_{opt.pretrained_models}.pt')))
        discriminator.train()
        optimizer_D.load_state_dict(torch.load(os.path.join(opt.log_dir, "trainig", "wgan", 'checkpoints', 'discriminators',
                                                            f'optimizer_{opt.pretrained_models}.pt')))
        start_point = opt.pretrained_models
    except:
        print("Pretrained models was not found!")
        sys.exit()
else:
    start_point = 0


for epoch in range(start_point + 1, start_point + opt.n_epochs + 1):
    for i, imgs in enumerate(dataloader):
        if imgs.shape[0] == 1:
            continue

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        loss_D.backward()
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            loss_G.backward()
            optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, start_point + opt.n_epochs, i+1, len(dataloader), loss_D.item(), loss_G.item())
        )

    loger.add_scalar("Generator loss", loss_G.item(), epoch)
    loger.add_scalar("Discriminator loss", loss_D.item(), epoch)
    loger.flush()

    with open(os.path.join(opt.log_dir, "trainig", "wgan", 'metrics', 'logs.txt'), 'a') as f:
        f.write(f"{epoch} {loss_G.item()} {loss_D.item()}\n")

    torch.save(generator.state_dict(),
               os.path.join(opt.log_dir, "trainig", "wgan", 'checkpoints', 'generators', f'model_{epoch}.pt'))
    torch.save(optimizer_G.state_dict(),
               os.path.join(opt.log_dir, "trainig", "wgan", 'checkpoints', 'generators', f'optimizer_{epoch}.pt'))
    torch.save(discriminator.state_dict(),
               os.path.join(opt.log_dir, "trainig", "wgan", 'checkpoints', 'discriminators', f'model_{epoch}.pt'))
    torch.save(optimizer_D.state_dict(),
               os.path.join(opt.log_dir, "trainig", "wgan", 'checkpoints', 'discriminators',
                            f'optimizer_{epoch}.pt'))
    if epoch % opt.sample_interval == 0:
        save_image(gen_imgs.data[:25],
                   os.path.join(opt.log_dir, "trainig", "wgan", 'images', f"epoch_{epoch}.png"), nrow=5,
                   normalize=True)

print("Training was finished!")
loger.close()
