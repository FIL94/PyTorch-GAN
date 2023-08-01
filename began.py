import argparse
import os
import sys
from datetime import datetime
import numpy as np

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
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="number of epoch for generate images")
opt = parser.parse_args()
print(opt)

os.makedirs(os.path.join(opt.log_dir, "trainig", "began", 'images'), exist_ok=True)
os.makedirs(os.path.join(opt.log_dir, "trainig", "began", 'checkpoints', 'generators'), exist_ok=True)
os.makedirs(os.path.join(opt.log_dir, "trainig", "began", 'checkpoints', 'discriminators'), exist_ok=True)
os.makedirs(os.path.join(opt.log_dir, "trainig", "began", 'metrics'), exist_ok=True)

loger = SummaryWriter(os.path.join(opt.log_dir, "trainig", "began", 'metrics',
                                   datetime.now().strftime('%Y_%m_%d %H_%M_%S')))

if os.path.exists(os.path.join(opt.log_dir, "trainig", "began", 'metrics', 'logs.txt')):
    if opt.pretrained_models is not None:
        with open(os.path.join(opt.log_dir, "trainig", "began", 'metrics', 'logs.txt'), 'r') as f:
            f.seek(0)
            lines = f.readlines()
            lines = lines[:int(opt.pretrained_models)]
            for line in lines:
                epoch, g_loss, d_loss, M = line.split(' ')
                loger.add_scalar("Generator loss", float(g_loss), int(epoch))
                loger.add_scalar("Discriminator loss", float(d_loss), int(epoch))
                loger.add_scalar("Convergence metric", float(M), int(epoch))
        with open(os.path.join(opt.log_dir, "trainig", "began", 'metrics', 'logs.txt'), 'a') as f:
            f.seek(0)
            f.truncate()
            f.writelines(lines)
    else:
        os.remove(os.path.join(opt.log_dir, "trainig", "began", 'metrics', 'logs.txt'))

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
print(f"Device: {'cuda' if cuda else 'cpu'}")


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Upsampling
        self.down = nn.Sequential(nn.Conv2d(opt.channels, 64, 3, 2, 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = opt.img_size // 2
        down_dim = 64 * (opt.img_size // 2) ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, opt.channels, 3, 1, 1))

    def forward(self, img):
        out = self.down(img)
        out = self.fc(out.view(out.size(0), -1))
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        return out


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)


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
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

# BEGAN hyper parameters
gamma = 0.75
lambda_k = 0.001
k = 0.0

if opt.pretrained_models is not None:
    try:
        generator.load_state_dict(torch.load(os.path.join(opt.log_dir, "trainig", "began", 'checkpoints', 'generators',
                                                          f'model_{opt.pretrained_models}.pt')))
        generator.train()
        optimizer_G.load_state_dict(torch.load(os.path.join(opt.log_dir, "trainig", "began", 'checkpoints', 'generators',
                                                            f'optimizer_{opt.pretrained_models}.pt')))
        discriminator.load_state_dict(torch.load(os.path.join(opt.log_dir, "trainig", "began", 'checkpoints', 'discriminators',
                                                              f'model_{opt.pretrained_models}.pt')))
        discriminator.train()
        optimizer_D.load_state_dict(torch.load(os.path.join(opt.log_dir, "trainig", "began", 'checkpoints', 'discriminators',
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

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = torch.mean(torch.abs(discriminator(gen_imgs) - gen_imgs))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        d_real = discriminator(real_imgs)
        d_fake = discriminator(gen_imgs.detach())

        d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
        d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
        d_loss = d_loss_real - k * d_loss_fake

        d_loss.backward()
        optimizer_D.step()

        # ----------------
        # Update weights
        # ----------------

        diff = torch.mean(gamma * d_loss_real - d_loss_fake)

        # Update weight term for fake samples
        k = k + lambda_k * diff.item()
        k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

        # Update convergence metric
        M = (d_loss_real + torch.abs(diff)).data.item()

        # --------------
        # Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f"
            % (epoch, start_point + opt.n_epochs, i+1, len(dataloader), d_loss.item(), g_loss.item(), M, k)
        )

    loger.add_scalar("Generator loss", g_loss.item(), epoch)
    loger.add_scalar("Discriminator loss", d_loss.item(), epoch)
    loger.add_scalar("Convergence metric", M, epoch)
    loger.flush()

    with open(os.path.join(opt.log_dir, "trainig", "began", 'metrics', 'logs.txt'), 'a') as f:
        f.write(f"{epoch} {g_loss.item()} {d_loss.item()} {M}\n")

    torch.save(generator.state_dict(), os.path.join(opt.log_dir, "trainig", "began", 'checkpoints', 'generators', f'model_{epoch}.pt'))
    torch.save(optimizer_G.state_dict(), os.path.join(opt.log_dir, "trainig", "began", 'checkpoints', 'generators', f'optimizer_{epoch}.pt'))
    torch.save(discriminator.state_dict(), os.path.join(opt.log_dir, "trainig", "began", 'checkpoints', 'discriminators', f'model_{epoch}.pt'))
    torch.save(optimizer_D.state_dict(), os.path.join(opt.log_dir, "trainig", "began", 'checkpoints', 'discriminators', f'optimizer_{epoch}.pt'))
    if epoch % opt.sample_interval == 0:
        save_image(gen_imgs.data[:25], os.path.join(opt.log_dir, "trainig", "began", 'images', f"epoch_{epoch}.png"), nrow=5, normalize=True)

print("Training was finished!")
loger.close()
