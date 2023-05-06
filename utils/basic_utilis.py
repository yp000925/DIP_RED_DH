import torch
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


def psnr(x, ref, is_tensor=False):
    x = (x - x.min()) / (x.max() - x.min())
    ref = (ref - ref.min()) / (ref.max() - ref.min())
    if is_tensor:
        mse = torch.mean(torch.square(ref - x))
        psnr = torch.tensor(10.0) * torch.log10(1 / mse)
    else:
        mse = np.mean(np.square(ref - x))
        psnr = 10.0 * np.log10(1 / mse)
    return psnr


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for filling tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplied by. Basically it is standard deviation scalar.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False
    return net_input

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False

def torch_to_np(img_var):
    """Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    """
    return img_var.detach().cpu().numpy()[0]

def np_to_torch(img_np):
    """Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    """
    return torch.from_numpy(img_np)[None, :]

def estimate_variance(img):
    h,w = img.shape[-2::]
    lap_kernel = np.array([[1,-2,1], [-2, 4, -2], [1,-2,1]])
    out = convolve2d(img, lap_kernel, mode='valid')
    out = np.sum(np.abs(out))
    out = (out*np.sqrt(0.5*np.pi)/(6*(h-2)*(w-2)))
    return out

# ---------- plot functions ----------
def plot_dict(data_dict):
    i, columns = 0, len(data_dict)
    scale = columns * 10  # you can play with it
    plt.figure(figsize=(scale, scale))
    for key, data in data_dict.items():
        i, ax = i + 1, plt.subplot(1, columns, i + 1)
        plt.imshow(data.img[0,:,:], cmap='gray')
        ax.text(0.5, -0.15, key + (" psnr: %.2f" % (data.psnr) if data.psnr is not None else ""),
                size=36, ha="center", transform=ax.transAxes)
    plt.show()

def plot_and_save_dict(data_dict, save_path =''):
    i, columns = 0, len(data_dict)
    scale = columns * 10  # you can play with it
    plt.figure(figsize=(scale, scale))
    for key, data in data_dict.items():
        i, ax = i + 1, plt.subplot(1, columns, i + 1)
        plt.imshow(data.img[0,:,:], cmap='gray')
        ax.text(0.5, -0.15, key + (" psnr: %.2f" % (data.psnr) if data.psnr is not None else ""),
                size=36, ha="center", transform=ax.transAxes)
    plt.savefig(save_path)
    plt.show()
