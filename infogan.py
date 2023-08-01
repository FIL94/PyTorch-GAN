import argparse
import os
import sys
from datetime import datetime
import numpy as np
import itertools

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
parser.add_argument("--code_dim", type=int, default=2, help="latent code")
parser.add_argument("--n_classes", type=int, default=1, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

os.makedirs(os.path.join(opt.log_dir, "trainig", "infogan", 'images'), exist_ok=True)
os.makedirs(os.path.join(opt.log_dir, "trainig", "infogan", 'checkpoints', 'generators'), exist_ok=True)
os.makedirs(os.path.join(opt.log_dir, "trainig", "infogan", 'checkpoints', 'discriminators'), exist_ok=True)
os.makedirs(os.path.join(opt.log_dir, "trainig", "infogan", 'metrics'), exist_ok=True)

loger = SummaryWriter(os.path.join(opt.log_dir, "trainig", "infogan", 'metrics',
                                   datetime.now().strftime('%Y_%m_%d %H_%M_%S')))

if os.path.exists(os.path.join(opt.log_dir, "trainig", "infogan", 'metrics', 'logs.txt')):
    if opt.pretrained_models is not None:
        with open(os.path.join(opt.log_dir, "trainig", "infogan", 'metrics', 'logs.txt'), 'r') as f:
            f.seek(0)
            lines = f.readlines()
            lines = lines[:int(opt.pretrained_models)]
            for line in lines:
                epoch, g_loss, d_loss, i_loss = line.split(' ')
                loger.add_scalar("Generator loss", float(g_loss), int(epoch))
                loger.add_scalar("Discriminator loss", float(d_loss), int(epoch))
                loger.add_scalar("Info loss", float(i_loss), int(epoch))
        with open(os.path.join(opt.log_dir, "trainig", "infogan", 'metrics', 'logs.txt'), 'a') as f:
            f.seek(0)
            f.truncate()
            f.writelines(lines)
    else:
        os.remove(os.path.join(opt.log_dir, "trainig", "infogan", 'metrics', 'logs.txt'))

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
print(f"Device: {'cuda' if cuda else 'cpu'}")


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.n_classes + opt.code_dim

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

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

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code


# Loss functions
adversarial_loss = torch.nn.MSELoss()
categorical_loss = torch.nn.CrossEntropyLoss()
continuous_loss = torch.nn.MSELoss()

# Loss weights
lambda_cat = 1
lambda_con = 0.1

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    categorical_loss.cuda()
    continuous_loss.cuda()

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
        return [self.transform(read_image(os.path.join(self.path,
                self.images[item])).div(torch.tensor([255]))), torch.tensor([0])]


girlsDataset = GirlsDataset(opt.path)
dataloader = torch.utils.data.DataLoader(
    girlsDataset,
    batch_size=opt.batch_size,
    shuffle=True
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

if opt.pretrained_models is not None:
    try:
        generator.load_state_dict(torch.load(os.path.join(opt.log_dir, "trainig", "infogan", 'checkpoints', 'generators',
                                                          f'model_{opt.pretrained_models}.pt')))
        generator.train()
        optimizer_G.load_state_dict(torch.load(os.path.join(opt.log_dir, "trainig", "infogan", 'checkpoints', 'generators',
                                                            f'optimizer_{opt.pretrained_models}.pt')))
        discriminator.load_state_dict(torch.load(os.path.join(opt.log_dir, "trainig", "infogan", 'checkpoints', 'discriminators',
                                                              f'model_{opt.pretrained_models}.pt')))
        discriminator.train()
        optimizer_D.load_state_dict(torch.load(os.path.join(opt.log_dir, "trainig", "infogan", 'checkpoints', 'discriminators',
                                                            f'optimizer_{opt.pretrained_models}.pt')))
        start_point = opt.pretrained_models
    except:
        print("Pretrained models was not found!")
        sys.exit()
else:
    start_point = 0

# Static generator inputs for sampling
static_z = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.latent_dim))))
static_label = to_categorical(
    np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes
)
static_code = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.code_dim))))


# ----------
#  Training
# ----------

for epoch in range(start_point + 1, start_point + opt.n_epochs + 1):
    for i, (imgs, labels) in enumerate(dataloader):
        if imgs.shape[0] == 1:
            continue

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

        # Generate a batch of images
        gen_imgs = generator(z, label_input, code_input)

        # Loss measures generator's ability to fool the discriminator
        validity, _, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, _, _ = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, valid)

        # Loss for fake images
        fake_pred, _, _ = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # ------------------
        # Information Loss
        # ------------------

        optimizer_info.zero_grad()

        # Sample labels
        sampled_labels = np.random.randint(0, opt.n_classes, batch_size)

        # Ground truth labels
        gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

        # Sample noise, labels and code as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

        gen_imgs = generator(z, label_input, code_input)
        _, pred_label, pred_code = discriminator(gen_imgs)

        info_loss = lambda_cat * categorical_loss(pred_label, gt_labels) + lambda_con * continuous_loss(
            pred_code, code_input
        )

        info_loss.backward()
        optimizer_info.step()

        # --------------
        # Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
            % (epoch, start_point + opt.n_epochs, i+1, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
        )

    loger.add_scalar("Generator loss", g_loss.item(), epoch)
    loger.add_scalar("Discriminator loss", d_loss.item(), epoch)
    loger.add_scalar("Info loss", info_loss.item(), epoch)
    loger.flush()

    with open(os.path.join(opt.log_dir, "trainig", "infogan", 'metrics', 'logs.txt'), 'a') as f:
        f.write(f"{epoch} {g_loss.item()} {d_loss.item()} {info_loss.item()}\n")

    torch.save(generator.state_dict(), os.path.join(opt.log_dir, "trainig", "infogan", 'checkpoints', 'generators', f'model_{epoch}.pt'))
    torch.save(optimizer_G.state_dict(), os.path.join(opt.log_dir, "trainig", "infogan", 'checkpoints', 'generators', f'optimizer_{epoch}.pt'))
    torch.save(discriminator.state_dict(), os.path.join(opt.log_dir, "trainig", "infogan", 'checkpoints', 'discriminators', f'model_{epoch}.pt'))
    torch.save(optimizer_D.state_dict(), os.path.join(opt.log_dir, "trainig", "infogan", 'checkpoints', 'discriminators', f'optimizer_{epoch}.pt'))
    if epoch % opt.sample_interval == 0:
        save_image(gen_imgs.data[:25], os.path.join(opt.log_dir, "trainig", "infogan", 'images', f"epoch_{epoch}.png"), nrow=5, normalize=True)

print("Training was finished!")
loger.close()
