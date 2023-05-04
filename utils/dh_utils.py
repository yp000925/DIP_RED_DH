import numpy as np
import matplotlib.pyplot as plt
import os
from math import pi, sqrt
from torch.fft import fft2,ifft2,fftshift,ifftshift
import torch
import torchvision
import PIL.Image as Image

def generate_otf_torch(wavelength, nx, ny, deltax,deltay, distance):
    """
    Generate the otf from [0,pi] not [-pi/2,pi/2] using torch
    :param wavelength:
    :param nx:
    :param ny:
    :param deltax:
    :param deltay:
    :param distance:
    :return:
    """
    r1 = torch.linspace(-nx/2,nx/2-1,nx)
    c1 = torch.linspace(-ny/2,ny/2-1,ny)
    deltaFx = 1/(nx*deltax)*r1
    deltaFy = 1/(nx*deltay)*c1
    mesh_qx, mesh_qy = torch.meshgrid(deltaFx,deltaFy)
    k = 2*torch.pi/wavelength
    otf = np.exp(1j*k*distance*torch.sqrt(1-wavelength**2*(mesh_qx**2
                                                           +mesh_qy**2)))
    otf = torch.fft.ifftshift(otf)
    return otf


def forward_propagation(x, A):
    out = torch.fft.ifft2(torch.multiply(torch.fft.fft2(x), A))
    return out

def prepross_bg(img,bg):
    temp = img/bg
    out = (temp-np.min(temp))/(1-np.min(temp))
    return out

def norm_tensor(x):
    return (x-torch.min(x))/(torch.max(x)-torch.min(x))

def psnr(x,im_orig):
    x = norm_tensor(x)
    # im_orig = norm_tensor(im_orig)
    mse = torch.mean(torch.square(im_orig - x))
    psnr = torch.tensor(10.0)* torch.log10(1/ mse)
    return psnr



