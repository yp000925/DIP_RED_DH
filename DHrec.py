import os
from threading import Thread  # needed since the denoiser is running in parallel
import queue

import numpy as np
import torch
import torch.optim
from utils.utils_mine import get_network_and_input
from utils.utils import *  # auxiliary functions
from utils.utils_mine import psnr
from utils.mine_blur_utils2 import *  # blur functions
from utils.data import Data  # class that holds img, psnr, time
from utils.dh_utils import *
from skimage.restoration import denoise_nl_means
import time
from scipy.signal import convolve2d
from torch.utils.tensorboard import SummaryWriter

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
holo = ifft2(torch.multiply(A, fft2(gt_intensity)))  # 此处应该是gt_intensity才对
holo = holo.abs()**2
holo = holo / torch.max(holo)
y =  np.array(holo.unsqueeze(0))

AT = generate_otf_torch(w, nx, ny, deltax, deltay, -distance)
rec = ifft2(torch.multiply(AT, fft2(holo)))
rec = torch.abs(rec)
rec = norm_tensor(rec)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(holo, cmap='gray')
ax[1].imshow(rec, cmap='gray')
ax[1].set_title(('BP PSNR{:.2f}').format(psnr(rec, gt_intensity)))
# fig.show()




# ---- estimate the noise  -----
lap_kernel = np.array([[1,-2,1], [-2, 4, -2], [1,-2,1]])
h=nx
w=ny
def estimate_variance(img):
    out = convolve2d(img, lap_kernel, mode='valid')
    out = np.sum(np.abs(out))
    out = (out*np.sqrt(0.5*np.pi)/(6*(h-2)*(w-2)))
    return out

print(holo.shape)
NOISE_SIGMA = estimate_variance(holo)*255
print(NOISE_SIGMA)

def non_local_means(noisy_np_img, sigma, fast_mode=True):
    """ get a numpy noisy image
        returns a denoised numpy image using Non-Local-Means
    """
    sigma = sigma / 255.
    h = 0.6 * sigma if fast_mode else 0.8 * sigma
    patch_kw = dict(h=h,                   # Cut-off distance, a higher h results in a smoother image
                    sigma=sigma,           # sigma provided
                    fast_mode=fast_mode,   # If True, a fast version is used. If False, the original version is used.
                    patch_size=5,          # 5x5 patches (Size of patches used for denoising.)
                    patch_distance=6,      # 13x13 search area
                    multichannel=False)
    denoised_img = []
    n_channels = noisy_np_img.shape[0]
    for c in range(n_channels):
        denoise_fast = denoise_nl_means(noisy_np_img[c, :, :], **patch_kw)
        denoised_img += [denoise_fast]
    return np.array(denoised_img)


def train_via_admm(net, net_input, denoiser_function, A, y, tau, noise_lev,            # H is the kernel, y is the blurred image
                   clean_img=None, plot_array={}, algorithm_name="",             # clean_img for psnr to be shown
                   gamma=.9, step_size=1000, save_path="",         # scheduler parameters and path to save params
                   admm_iter=5000, LR=0.004,                                          # admm_iter is step_2_iter
                   sigma_f=3, update_iter=10, method='fixed_point',  # method: 'fixed_point' or 'grad' or 'mixed'
                   beta=0.1, mu=0.1, LR_x=None, noise_factor=0.01):  # LR_x needed only if method!=fixed_point
    """ training the network using
        # mu=0.04
        ## Must Params ##
        net                 - the network to be trained
        net_input           - the network input
        denoiser_function   - an external denoiser function, used as black box, this function
                              must get numpy noisy image, and return numpy denoised image
        H                   - the blur kernel
        y                   - the blurred image [C,H,W]

        # optional params #
        clean_img           - the original image if exist for psnr compare only, or None (default)
        plot_array          - prints params at the begging of the training and plot images at the required indices
        algorithm_name      - the name that would show up while running, just to know what we are running ;)
        admm_iter           - total number of admm epoch
        LR                  - the lr of the network
        sigma_f             - the sigma to send the denoiser function
        update_iter         - denoised image updated every 'update_iter' iteration
        method              - 'fixed_point' or 'grad' or 'mixed'

        # equation params #
        beta                - regularization parameter (lambda in the article)
        mu                  - ADMM parameter
        LR_x                - learning rate of the parameter x, needed only if method!=fixed point
        # more
        noise_factor       - the amount of noise added to the input of the network
    """
    # get optimizer and loss function:
    mse = torch.nn.MSELoss().type(dtype)  # using MSE loss
    # additional noise added to the input:
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    # x update method:
    if method == 'fixed_point':
        swap_iter = admm_iter + 1
        LR_x = None
    elif method == 'grad':
        swap_iter = -1
    elif method == 'mixed':
        swap_iter = admm_iter // 2
    else:
        assert False, "method can be 'fixed_point' or 'grad' or 'mixed' only "

    # run RED via ADMM, initialize:
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)  # using ADAM opt
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma, step_size=step_size)

    # initialization
    y_torch = np_to_torch(y)
    x = y.copy()
    avg = np.rint(y)
    f_x, u = x.copy(), np.zeros_like(x)
    img_queue = queue.Queue()
    # The denoiser thread that runs in parallel:
    denoiser_thread = Thread(target=lambda q, f, f_args: q.put(f(*f_args)),
                             args=(img_queue, denoiser_function, [x.copy(), sigma_f]))
    denoiser_thread.start()

    list_psnr=[]
    list_stopping=[]

    if clean_img is not None:
        psnr_y = compare_PSNR(clean_img, y,on_y=(not GRAY_SCALE), gray_scale=GRAY_SCALE)  # get the noisy image psnr

    # ADMM:
    for i in range(1, 1 + admm_iter):

        rho = tau*noise_lev*np.sqrt(y.shape[0]*y.shape[1]*y.shape[2] - 1)

        # step 1, update network:
        optimizer.zero_grad()
        net_input = net_input_saved + (noise.normal_() * noise_factor)
        out = net(net_input)
        out_np = torch_to_np(out)

        pred_y = forward_propagation(out[0,0,:,:],A).abs()**2
        pred_y = pred_y/torch.max(pred_y)
        # loss:
        loss_y = mse(pred_y[None,None,:,:],y_torch)
        loss_x = mse(out, np_to_torch(x - u))

        total_loss = loss_y + mu * loss_x
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        # step 2, update x using a denoiser and result from step 1
        if i % update_iter == 0:  # the denoiser work in parallel
            denoiser_thread.join()
            f_x = img_queue.get()
            denoiser_thread = Thread(target=lambda q, f, f_args: q.put(f(*f_args)),
                                     args=(img_queue, denoiser_function, [x.copy(), sigma_f]))
            denoiser_thread.start()

        if i < swap_iter:
            x = 1 / (beta + mu) * (beta * f_x + mu * (out_np + u))
        else:
            x = x - LR_x * (beta * (x - f_x) + mu * (x - out_np - u))
        np.clip(x, 0, 1, out=x)  # making sure that image is in bounds

        # step 3, update u
        u = u + out_np - x

        # Averaging:
        avg = avg * .99 + out_np * .01

        stopping = loss_y.detach().numpy()/ rho
        list_stopping.append(stopping)

        # show psnrs:
        if clean_img is not None:
            psnr_net = compare_PSNR(clean_img, out_np, on_y=(not GRAY_SCALE), gray_scale=GRAY_SCALE)
            psnr_x_u = compare_PSNR(clean_img, x - u, on_y=(not GRAY_SCALE), gray_scale=GRAY_SCALE)
            psnr_avg = compare_PSNR(clean_img, avg, on_y=(not GRAY_SCALE), gray_scale=GRAY_SCALE)
            psnr_noisy = compare_PSNR(clean_img, y,on_y=(not GRAY_SCALE), gray_scale=GRAY_SCALE)
            list_psnr.append(psnr_avg)
            print('\r', algorithm_name, '%04d/%04d Loss %f' % (i, admm_iter, total_loss.item()),
                  'psnrs: noisy: %.2f net: %.2f x-u: %.2f avg: %.2f' % (psnr_noisy,psnr_net, psnr_x_u,psnr_avg),
                  'stopping: %.2f' %(stopping), end='')
            if i in plot_array:
                tmp_dict = {'Clean': Data(clean_img),
                            'Blurred': Data(y,psnr_y),
                            'Net': Data(psnr, psnr_net),
                            'x-u': Data(x - u, psnr_x_u),
                            'avg': Data(avg, psnr_avg),
                            'u': Data((u - np.min(u)) / (np.max(u) - np.min(u)))
                            }
                plot_dict(tmp_dict)
        else:
            print('\r', algorithm_name, 'iteration %04d/%04d Loss %f' % (i, admm_iter, total_loss.item()), end='')

    # join the thread:
    if denoiser_thread.is_alive():
        denoiser_thread.join()  # joining the thread
    return avg, list_psnr,list_stopping


tau = 1
img_shape = [1,nx,ny]
noise_lev = NOISE_SIGMA/255
net, net_input = get_network_and_input(img_shape)
plot_checkpoints = {1, 10, 100, 1000, 2000, 5000, 10000, 20000}
clean,list_psnr,list_stopping = train_via_admm(net, net_input, non_local_means, A,y, tau, noise_lev,admm_iter=1000,
                                                                               algorithm_name='DIP_RED',
                                                                               clean_img=np.array(gt_intensity.unsqueeze(0)))
