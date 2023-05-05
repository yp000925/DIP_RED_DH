import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from threading import  Thread
import queue

import numpy as np
import torch
from models.network_utilis import get_network_and_input,non_local_means
from utils.basic_utilis import psnr,estimate_variance
from utils.data import Data
from utils.dh_utils import *

import time
from scipy.signal import convolve2d
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from DIP_RED_Net import train_via_admm


GRAY_SCALE = True
NOISE_SIGMA = 5
if torch.cuda.is_available():
    device = 'cuda'
    dtype = torch.cuda.FloatTensor
else:
    device = 'cpu'
    dtype = torch.FloatTensor

# ---- define propagation kernel -----
w = 632e-9
deltax = 3.45e-6
deltay = 3.45e-6
distance = 0.02
nx = 512
ny = 512
model_name = "AutoDHc/"
timestr = time.strftime("sample3_%Y-%m-%d-%H_%M_%S/", time.localtime())
out_dir = 'output/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_dir = out_dir + model_name
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
writer = SummaryWriter(out_dir + timestr)

""" Load the GT intensity map and get the diffraction pattern"""
img = Image.open('test_image.png').resize([512, 512]).convert('L')
# img = Image.open('test_image2.jpg').resize([512, 512]).convert('L')
# img = Image.open('USAF1951.jpg').resize([512, 512]).convert('L')
gt_intensity = torch.from_numpy(np.array(img))
gt_intensity = gt_intensity / torch.max(gt_intensity)

# ---- forward and backward propagation -----
A = generate_otf_torch(w, nx, ny, deltax, deltay, distance)
AT = generate_otf_torch(w, nx, ny, deltax, deltay, -distance)
holo = ifft2(torch.multiply(A, fft2(gt_intensity)))  # 此处应该是gt_intensity才对
holo = holo.abs()**2
holo = holo / torch.max(holo)
rec = ifft2(torch.multiply(AT, fft2(holo)))
rec = torch.abs(rec)
rec = norm_tensor(rec)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(holo, cmap='gray')
ax[1].imshow(rec, cmap='gray')
ax[1].set_title(('BP PSNR{:.2f}').format(psnr(rec, gt_intensity)))
# fig.show()


# ---- Define the network and data structure -----
y  = np.array(holo.unsqueeze(0)) # numpy array with shape [ch,h,w]
clean_img = np.array(gt_intensity.unsqueeze(0))
print(y.shape)
NOISE_SIGMA = estimate_variance(y[0,:,:])*255
print(NOISE_SIGMA)
net, net_input = get_network_and_input(y.shape)
net = net.to(device)
plot_array = {1, 10, 20, 30}
clean,list_psnr,list_stopping = train_via_admm(net.to(device), net_input.to(device), non_local_means, A.to(device),y, dtype=dtype,
                                               device = device,clean_img=clean_img,
                                               admm_iter=30)

