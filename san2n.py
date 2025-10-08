from __future__ import division
import os
import time
import datetime
import argparse
import numpy as np
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import restore_img
from s3dCNN import SNet

parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default="gauss20")
parser.add_argument('--train_sim1_dirs', type=str, default='', help='data y file name')
parser.add_argument('--train_sim2_dirs', type=str, default='', help='data y` file name')
parser.add_argument('--val_dirs', type=str, default='')
parser.add_argument('--mask_path', type=str, default='', help='data mask file name')
parser.add_argument('--gt_path', type=str, default='', help='data gt file name')
parser.add_argument('--is_generate_image', type=bool, default=False)
parser.add_argument('--save_model_path', type=str, default='./results')
parser.add_argument('--pad_patch_size', default=[0, 0, 0], type=list)
parser.add_argument('--log_name', type=str, default='log')  #
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--in_channel', type=int, default=60)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.00001)
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--n_snapshot', type=int, default=15)
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument("--Lambda1", type=float, default=1.0)
parser.add_argument("--Lambda2", type=float, default=1.0)
parser.add_argument("--sigma_path", type=str, default='', help='data sigma file name')
parser.add_argument("--increase_ratio1", type=float, default=10)
parser.add_argument("--increase_ratio2", type=float, default=1)

opt, _ = parser.parse_known_args()
operation_seed_counter = 0
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices

def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    #g_cuda_generator = torch.Generator(device="cpu")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def validation_data(dataset_dir):
    im1 = np.load(dataset_dir)
    im1 = np.array(im1, dtype=np.float32)

    im = np.expand_dims(im1, axis=0)
    im = np.transpose(im, [0, 4, 1, 2, 3])

    return im


def space_to_depth(x, block_size):
    n, c, h, w, l = x.size()
    unfolded_x = torch.nn.functional.unfold(x.squeeze(1), 2, stride=2)
    unfolded_x = torch.nn.functional.unfold(unfolded_x.unsqueeze(1), [8, 1], stride=[8, 1])
    return unfolded_x.view(n, c * block_size**3, h // block_size,
                           w // block_size, l // block_size)


def generate_mask_pair(img):
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w, l = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * l // 2 * 8,),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * l // 2 * 8, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 3], [0, 4],
         [1, 0], [1, 2], [1, 5],
         [2, 1], [2, 3], [2, 6],
         [3, 0], [3, 2], [3, 7],
         [4, 5], [4, 7], [4, 0],
         [5, 4], [5, 6], [5, 1],
         [6, 5], [6, 7], [6, 2],
         [7, 4], [7, 6], [7, 3]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2 * l // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=24,
                  size=(n * h // 2 * w // 2 * l // 2, ),
                  generator=get_generator(),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * l // 2 * 8,
                                step=8,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1) #需要检查
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


class RicianLoss(nn.Module):
    def __init__(self, sigma_path):
        super(RicianLoss, self).__init__()
        self.sigma =  torch.from_numpy(np.load(sigma_path))
    #
    def forward(self, predictions, inputs):
        # Rician loss
        sigma = self.sigma.view(1, -1, 1, 1, 1).to(predictions.device)
        n_batch, gds = inputs.shape[0], inputs.shape[1]


        predictions = predictions.view(n_batch, gds, -1)
        inputs = inputs.view(n_batch, gds, -1)
        sigma_expanded = sigma.view(1, gds, 1).expand(n_batch, gds, inputs.shape[-1])

        term1 = torch.log(inputs / (sigma_expanded ** 2))
        term1 = torch.where(torch.isinf(term1), torch.zeros_like(term1), term1)

        term2 = -(inputs ** 2 + predictions ** 2) / (2 * (sigma_expanded ** 2))
        z = (inputs * predictions) / (sigma_expanded ** 2)
        I0e = torch.special.i0e(z)
        lI0e = torch.log(I0e)
        term3 = lI0e + z

        log_pdf = term1 + term2 + term3

        loss = -torch.sum(log_pdf) / n_batch

        return loss


def sobel_3d(data):
    sobel_x = torch.tensor([[[[[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]],

                              [[2, 4, 2],
                               [0, 0, 0],
                               [-2, -4, -2]],

                              [[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]]]]], dtype=torch.float32).cuda() / 32

    sobel_y = torch.tensor([[[[[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]],

                              [[2, 0, -2],
                               [4, 0, -4],
                               [2, 0, -2]],

                              [[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]]]]], dtype=torch.float32).cuda() / 32

    sobel_z = torch.tensor([[[[[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]],

                              [[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]],

                              [[-1, -2, -1],
                               [-2, -4, -2],
                               [-1, -2, -1]]]]], dtype=torch.float32).cuda() / 32
    grad_x = torch.stack(
        [F.conv3d(data[:, i:i + 1, :, :, :], sobel_x, padding=1, stride=1) for i in range(data.size(1))], dim=1)[:, :,
             0, :, :, :]
    grad_y = torch.stack(
        [F.conv3d(data[:, i:i + 1, :, :, :], sobel_y, padding=1, stride=1) for i in range(data.size(1))], dim=1)[:, :,
             0, :, :, :]
    grad_z = torch.stack(
        [F.conv3d(data[:, i:i + 1, :, :, :], sobel_z, padding=1, stride=1) for i in range(data.size(1))], dim=1)[:, :,
             0, :, :, :]
    # 计算梯度的模长
    gradient_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + grad_z.pow(2))
    return F.softmax(gradient_magnitude)

def generate_subimages(img, mask):
    n, c, h, w, l = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           l // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 4, 1).reshape(-1)
        subimage[:, i:i + 1, :, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, l // 2, 1).permute(0, 4, 1, 2, 3)
    return subimage


class DataLoader_Imagenet_val(Dataset):
    def __init__(self, data_sim1_dir, data_sim2_dir):
        super(DataLoader_Imagenet_val, self).__init__()
        self.data_sim1 = np.load(data_sim1_dir, allow_pickle=True)
        self.data_sim2 = np.load(data_sim2_dir, allow_pickle=True)
        print('fetch {} samples for training'.format(len(self.data_sim1)))

    def __getitem__(self, index):
        # fetch image
        train_data_sim1 = self.data_sim1[index]
        train_data_sim1 = np.array(train_data_sim1, dtype=np.float32)
        train_data_sim1 = torch.from_numpy(train_data_sim1)
        train_data_sim1 = train_data_sim1.permute(3, 0, 1, 2)

        train_data_sim2 = self.data_sim2[index]
        train_data_sim2 = np.array(train_data_sim2, dtype=np.float32)
        train_data_sim2 = torch.from_numpy(train_data_sim2)
        train_data_sim2 = train_data_sim2.permute(3, 0, 1, 2)
        return train_data_sim1, train_data_sim2

    def __len__(self):
        return len(self.data_sim1)



# Training Set
TrainingDataset = DataLoader_Imagenet_val(opt.train_sim1_dirs, opt.train_sim2_dirs)
TrainingLoader = DataLoader(dataset=TrainingDataset,
                            num_workers=1,
                            batch_size=opt.batchsize,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)


# Validation Set
valid_dict = {}

val_data= validation_data(opt.val_dirs)

valid_dict.update({'simulation': val_data})
network = SNet(opt.in_channel, opt.in_channel)

s = sum([np.prod(list(p.size())) for p in network.parameters()])
print('Number of params: %d' % s)


if opt.parallel:
    network = torch.nn.DataParallel(network)
network = network.cuda()

# about training scheme
num_epoch = opt.n_epoch
ratio = num_epoch / 100
optimizer = optim.Adam(network.parameters(), lr=opt.lr)

scheduler = lr_scheduler.MultiStepLR(optimizer,
                                     milestones=[
                                         int(50 * ratio) - 1,
                                         int(70 * ratio) - 1,
                                         int(90 * ratio) - 1,
                                     ],
                                     gamma=opt.gamma)

print("Batchsize={}, number of epoch={}".format(opt.batchsize, opt.n_epoch))

print('init finish')

loss_rician = RicianLoss(opt.sigma_path)

val_save_img_number = 0
for epoch in range(1, opt.n_epoch + 1):
    cnt = 0
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    print("LearningRate of Epoch {} = {}".format(epoch, current_lr))

    epoch_loss = 0
    epoch_loss1 = 0
    epoch_lossr = 0
    epoch_loss2 = 0
    epoch_lossq = 0
    epoch_all_loss = 0
    epoch_lossq2 = 0
    epoch_losssvl = 0
    number = 0
    network.train()
    st = time.time()
    for iteration, data in tqdm(enumerate(TrainingLoader), total=len(TrainingLoader), desc='train'):
        number += 1
        noisy_sim1, noisy_sim2 = data[0], data[1]
        noisy_sim1, noisy_sim2 = noisy_sim1.cuda(), noisy_sim2.cuda()

        optimizer.zero_grad()

        mask1, mask2 = generate_mask_pair(noisy_sim1)

        noisy_sub1 = generate_subimages(noisy_sim1, mask1)
        noisy_sub2 = generate_subimages(noisy_sim1, mask2)

        noisy_suba = generate_subimages(noisy_sim2, mask1)

        with torch.no_grad():
            noisy_denoised1 = network(noisy_sim1)
            noisy_denoised2 = network(noisy_sim2)

        noisy_sub1_denoised = generate_subimages(noisy_denoised1, mask1)
        noisy_sub2_denoised = generate_subimages(noisy_denoised1, mask2)

        noisy_suba_denoised = generate_subimages(noisy_denoised2, mask1)


        noisy_output1 = network(noisy_sub1)
        noisy_target1 = noisy_sub2

        noisy_outputa = noisy_output1
        noisy_targeta = noisy_suba

        loss_r = loss_rician(noisy_output1, noisy_target1)
        if math.isnan(loss_r.item()):
            loss_r = 0
        epoch_lossr += opt.beta * loss_r.item()
        Lambda1 = epoch / opt.n_epoch * opt.increase_ratio1
        Lambda2 = epoch / opt.n_epoch * opt.increase_ratio2

        diff1 = noisy_output1 - noisy_target1
        diffq = noisy_outputa - noisy_targeta
        exp_diff1 = noisy_sub1_denoised - noisy_sub2_denoised
        exp_diffq = noisy_sub1_denoised - noisy_suba_denoised

        loss1 = torch.mean(diff1**2)
        epoch_loss1 += loss1.item()

        loss_attn = sobel_3d(noisy_sub1_denoised)
        lossq = Lambda2 * torch.mean(torch.mul(diffq**2, loss_attn))
        epoch_lossq += lossq.item()

        loss2 = Lambda1 * (torch.mean((diff1 - exp_diff1)**2))
        epoch_loss2 += loss2.item()

        lossq2 = Lambda2 * (torch.mean(torch.mul((diffq - exp_diffq)**2, loss_attn)))
        epoch_lossq2 += lossq2.item()


        loss_all = opt.Lambda1 * loss1 + opt.Lambda2 * loss2 + lossq + lossq2 + opt.beta * loss_r
        epoch_all_loss += loss_all.item()

        loss_all.backward()
        optimizer.step()

    print('{:04d} , Lambda2={},Lossq={:.6f}, Loss1={:.6f}, Lambda1={}, Loss2={:.6f} , Lossq2={:.6f} ,Lossr={:.6f} ,Loss_Full={:.6f}, Time={:.4f}'
    .format(epoch, Lambda2, epoch_lossq / number, epoch_loss1 / number, Lambda1, epoch_loss2 / number, epoch_lossq2 / number, epoch_lossr / number, epoch_all_loss / number,
    time.time() - st))

    scheduler.step()

    if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch:
        network.eval()
        # save checkpoint
        checkpoint(network, epoch, "model")
        # validation
        save_model_path = os.path.join(opt.save_model_path, opt.log_name,
                                       systime)
        print('save path is {}'.format(save_model_path))
        validation_path = os.path.join(save_model_path, "validation")
        os.makedirs(validation_path, exist_ok=True)
        np.random.seed(21)
        psnr_result = []
        ssim_result = []
        rmse_result = []
        for valid_name, valid_images in valid_dict.items():
            val_st = time.time()
            repeat_times = 1
            val_im = valid_images
            for i in range(repeat_times):
                for idx, noisy_im in enumerate(val_im):
                    val_noisy_im = torch.Tensor(noisy_im)
                    val_noisy_im = torch.unsqueeze(val_noisy_im, 0)
                    val_noisy_im = val_noisy_im.cuda()
                    # padding to square
                    _, _, H, W, L = val_noisy_im.shape
                    L_size = max(opt.pad_patch_size[2], L)
                    W_size = max(opt.pad_patch_size[1], W)
                    H_size = max(opt.pad_patch_size[0], H)
                    val_noisy_im = F.pad(val_noisy_im, [0, L_size - L, 0, W_size - W, 0, H_size - H])
                    with torch.no_grad():
                        pre = network(val_noisy_im)
                        pre = pre[:, :, :H, :W, :L]
                    pre = pre.permute(0, 2, 3, 4, 1).cpu().numpy()
                noisy_im = np.transpose(noisy_im, [1, 2, 3, 0])
                psnr_value_all, ssim_value_all, rmse_value_all = restore_img(valid_name, pre, noisy_im, validation_path, opt.mask_path, opt.gt_path, opt.is_generate_image)
                print('Overall epoch is {}, time is {:.4f}, psnr is {:.3f}, ssim is {:.3f}, rmse is {:.3f}'.format(epoch, time.time() - val_st, psnr_value_all, ssim_value_all, rmse_value_all))

                log_path = os.path.join(validation_path,"log.csv")
                with open(log_path, "a") as f:
                    f.writelines(valid_name + ", epoch{}, psnr is {}, ssim is {}, rmse is {}\n".format(epoch, psnr_value_all, ssim_value_all, rmse_value_all))

