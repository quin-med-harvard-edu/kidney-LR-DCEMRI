import torch
import numpy as np
import torchkbnufft as tkbn
import torch.optim as optim
import itertools
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import random
import hdf5storage as hdfs

from helpers.downsample import down_sample_freq
from helpers.partition_data import partition_volume
from model.network import GenConvNet, DenseMapNet
from model.loss import data_l2_loss
from model.z_generator import gen_line_segment

root_dat = './data/'
root_model = './torch_model/'
root_tensorboard = './tensor_b/'
subject_id = 'sub-9'
single_slice_idx = 1

# Insert data here / dat_slice is data dictionary
dat_slice = {}
dat_slice['coilprofile'] = hdfs.loadmat(
    os.path.join(os.path.join(os.path.join(root_dat, subject_id)),
                 'coilprofile_slice_1.mat'))['coilprofile_slice_1']
dat_slice['dcf'] = hdfs.loadmat(
    os.path.join(os.path.join(os.path.join(root_dat, subject_id)),
                 'dcf_slice_1.mat'))['dcf_slice_1']
dat_slice['k3n'] = hdfs.loadmat(
    os.path.join(os.path.join(os.path.join(root_dat, subject_id)),
                 'k3n_slice_1.mat'))['k3n_slice_1']
dat_slice['k_samples'] = hdfs.loadmat(
    os.path.join(os.path.join(os.path.join(root_dat, subject_id)),
                 'k_samples_slice_1.mat'))['k_samples_slice_1']

# For reproducibility
torch.manual_seed(123)
random.seed(123)
np.random.seed(123)

# Learning parameters
learning_rate = np.power(.1, 4)
fov_factor = 2  # frequency downsampling factor
n_z_dim = 56  # dimensionality of lower dim representation
bias_z = 2 * np.ones((1, n_z_dim))  # add bias, origin is not preferred
var_noise = 0.0001  # noise variance on z
max_r = 2  # maximum length of line segment

s_batch = 48  # batch size
max_num_iter = 12000  # maximum number of iterations
save_hop = 200  # save after save_hop iter

# Data parameters
start_idx = 0  # starting spoke index
stop_idx = int(34 * 4 * 60 / 3.3)  # final spoke index
spv = 34  # spoke per volume (spokes are assumed to be pseudo instant)

now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
model_path = os.path.join(os.path.join(root_model, subject_id), date_time)
path_z = os.path.join(model_path, 'z_vals')

if not os.path.exists(model_path):
    os.makedirs(model_path)
    os.makedirs(path_z)

writer = SummaryWriter(os.path.join(root_tensorboard, date_time))
gpu = torch.device("cuda")

map_net = DenseMapNet(n_z_dim)
gen_conv = GenConvNet()
nd = (224, 224)
kd = (int(224 * 1.5), int(224 * 1.5))
jd = (6, 6)

operator = tkbn.KbNufft(im_size=nd, grid_size=kd, numpoints=jd).to(device=gpu)

mat_net = map_net.to(device=gpu)
gen_conv = gen_conv.to(device=gpu)

optimizer = optim.Adam(itertools.chain(map_net.parameters(), gen_conv.parameters()),
                       lr=learning_rate)

div_k_space, div_k_samples, coil_p, div_sqrt_dcf = partition_volume(dat_slice,
                                                                    spv,
                                                                    start_idx,
                                                                    stop_idx)

div_k_space, div_k_samples, div_sqrt_dcf, coil_p = down_sample_freq(div_k_space,
                                                                    div_k_samples,
                                                                    div_sqrt_dcf, coil_p,
                                                                    fov_factor)

len_dat = len(div_k_samples)
idx_c_p = np.array([0 for _ in range(len_dat)])
coil_p = [coil_p]

low_dim_r = gen_line_segment(max_r, len_dat, n_z_dim, bias_z)
np.savetxt(os.path.join(path_z, 'z_slice_{}.txt'.format(single_slice_idx)),
           low_dim_r)

len_dat = len(div_k_space)
div_k_space = div_k_space * np.expand_dims(div_sqrt_dcf, axis=2)

div_k_space = torch.tensor(div_k_space).to(torch.complex64)
k_traj = np.array([np.array([
    np.real(el.ravel()) * 2 * np.pi,
    np.imag(el.ravel()) * 2 * np.pi]) for el in div_k_samples])
k_traj = torch.tensor(k_traj).to(torch.float32)

coil_p = np.transpose(coil_p, (0, 3, 1, 2))
[_, num_coil, dim_x_im, dim_y_im] = coil_p.shape

div_sqrt_dcf = torch.tensor(np.expand_dims(div_sqrt_dcf, 1)
                            / np.sqrt(dim_x_im * dim_y_im)).to(torch.float32)

noise = np.random.randn(len_dat, n_z_dim)
low_z = low_dim_r + noise * var_noise
low_z = torch.tensor(low_z).to(torch.float32)

plot_el = low_z[-1].unsqueeze(0)  # fix what to plot on tensorboard

tqdm_bar = tqdm(range(max_num_iter), desc='Iterating:', leave=True)
i, ii = 0, 0

list_idx_ = list(range(len(low_z)))
for idx_iter in tqdm_bar:

    # randomly shuffle training samples in the same order
    random.shuffle(list_idx_)
    div_k_space = div_k_space[list_idx_]
    k_traj = k_traj[list_idx_]
    div_sqrt_dcf = div_sqrt_dcf[list_idx_]
    low_z = low_z[list_idx_]
    div_coil_p = idx_c_p[list_idx_]

    mv_loss = 0
    for idx_b in range(int(np.ceil(len_dat / s_batch))):
        optimizer.zero_grad()
        tmp_ = map_net.forward(
            low_z[idx_b * s_batch:np.minimum(
                len_dat, (idx_b + 1) * s_batch)].to(device=gpu).unsqueeze(1))
        tmp__ = gen_conv.forward(torch.reshape(tmp_, (tmp_.shape[0], tmp_.shape[1],
                                                      int(np.sqrt(tmp_.shape[2])),
                                                      int(np.sqrt(tmp_.shape[2])))))

        idx_coils = div_coil_p[idx_b * s_batch:
                               np.minimum(len_dat, (idx_b + 1) * s_batch)]
        c_profile = torch.tensor([coil_p[_idx, :, :, :] for _idx in idx_coils]).to(
            torch.complex64)
        k_spc = div_k_space[idx_b * s_batch:
                            np.minimum(len_dat, (idx_b + 1) * s_batch)]
        k_trj = k_traj[idx_b * s_batch:
                       np.minimum(len_dat, (idx_b + 1) * s_batch)]
        sqrt_dcf = div_sqrt_dcf[idx_b * s_batch:
                                np.minimum(len_dat, (idx_b + 1) * s_batch)]

        smap_sz_b = int(np.abs(idx_b * s_batch - np.minimum(
            len_dat, (idx_b + 1) * s_batch)))
        smap_sz = (smap_sz_b, num_coil, dim_x_im, dim_y_im)
        smap = torch.ones(*smap_sz, dtype=torch.complex64)
        loss = data_l2_loss(
            tmp__,
            k_spc.to(device=gpu),
            k_trj.to(device=gpu),
            sqrt_dcf.to(device=gpu),
            c_profile.to(device=gpu),
            smap.to(device=gpu),
            operator)

        loss.backward()
        writer.add_scalar('training loss', loss.item(), i)
        i += 1

        mv_loss += loss.item()
        optimizer.step()

    if ((idx_iter + 1) % save_hop) == 0:
        ckpt_path = os.path.join(model_path, 'checkpoint_' + str(ii) + '.pt')
        torch.save({
            'map_net_dict': map_net.state_dict(),
            'gen_conv_dict': gen_conv.state_dict(),
            'optimizer': optimizer.state_dict()}, ckpt_path)
        with torch.no_grad():
            # generate image to plot in tensorbard
            tmp_ = map_net.forward(plot_el.to(device=gpu).unsqueeze(1))
            tmp__ = gen_conv.forward(
                torch.reshape(tmp_, (tmp_.shape[0], tmp_.shape[1],
                                     int(np.sqrt(tmp_.shape[2])),
                                     int(np.sqrt(tmp_.shape[2])))))
            writer_img = torch.abs(torch.complex(
                tmp__[-1, 0, :, :], tmp__[-1, 1, :, :])).unsqueeze(0)
        writer.add_image('image rec', writer_img, ii)
        ii += 1
    tqdm_bar.set_description("loss:{}".format(mv_loss / int(len_dat / s_batch)))
