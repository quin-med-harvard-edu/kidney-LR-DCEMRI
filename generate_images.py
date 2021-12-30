import torch
import numpy as np
import torchkbnufft as tkbn
import torch.optim as optim
import itertools
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import nibabel as nib

from model.network import DenseMapNet, GenConvNet
from model.z_generator import gen_line_segment

root_train = './torch_model'

# TODO: change dummy names here
subject_id = 'sub-9'
name_pre_trained = 'dummy'
checkpoint = 'dummy'
single_slice_idx = 1

root_rec_folder = os.path.join(os.path.join(os.path.join(root_train, 'gens'),
                                            subject_id), name_pre_trained)
if not os.path.exists(root_rec_folder):
    os.makedirs(root_rec_folder)

n_el = None
root_tensorboard = os.path.join(root_train, 'tensor_b')
root_pre_trained = os.path.join(root_train, os.path.join(subject_id, name_pre_trained))
path_model = os.path.join(root_pre_trained, checkpoint + '.pt')
path_z = os.path.join(root_pre_trained, 'z_vals/z_slice_' + str(single_slice_idx) +
                      '.txt')

# root_pre_trained = None
if path_model:
    assert subject_id in path_model, 'this model is not trained for {}'.format(
        subject_id)

s_batch = 60
map_net = DenseMapNet(56)
gen_conv = GenConvNet()

gpu = torch.device("cuda")
mat_net = map_net.to(device=gpu)
gen_conv = gen_conv.to(device=gpu)

optimizer = optim.Adam(itertools.chain(map_net.parameters(), gen_conv.parameters()),
                       lr=np.power(.1, 3))

writer = SummaryWriter(os.path.join(root_tensorboard,
                                    name_pre_trained + '-rec'))

if path_model:
    print('loading pre-trained')
    pre_train_dict = torch.load(path_model)
    map_net.load_state_dict(pre_train_dict['map_net_dict'])
    gen_conv.load_state_dict(pre_train_dict['gen_conv_dict'])
    optimizer.load_state_dict(pre_train_dict['optimizer'])
    tmp_z = np.loadtxt(path_z)
    if n_el:
        low_z = tmp_z[0] + np.outer(np.arange(n_el), (tmp_z[-1] - tmp_z[0]) / n_el)
    else:
        low_z = tmp_z
    low_z = torch.tensor(low_z).to(torch.float32)

else:
    low_z = torch.tensor(gen_line_segment(30, 56)).to(torch.float32)

nd = (448, 448)
kd = (int(448 * 1.5), int(448 * 1.5))
jd = (2, 2)
operator = tkbn.KbNufft(im_size=nd, grid_size=kd, numpoints=jd).to(device=gpu)

ii = 0
with torch.no_grad():
    rec_volume = []
    for idx_b in tqdm(range(int(np.ceil(low_z.shape[0] / s_batch)))):
        # generate image to plot in tensorbard
        tmp_ = map_net.forward(low_z[idx_b * s_batch:np.minimum(
            low_z.shape[0], (idx_b + 1) * s_batch)].to(device=gpu).unsqueeze(1))
        tmp__ = gen_conv.forward(
            torch.reshape(tmp_, (tmp_.shape[0], tmp_.shape[1],
                                 int(np.sqrt(tmp_.shape[2])),
                                 int(np.sqrt(tmp_.shape[2])))))
        writer_img = torch.abs(torch.complex(
            tmp__[:, 0, :, :], tmp__[:, 1, :, :])).unsqueeze(1)
        rec_volume.extend(np.array(writer_img.cpu()))
        for im_idx in range(len(writer_img)):
            writer.add_image('image rec', writer_img[im_idx, :, :, :], ii)
            ii += 1

    rec_volume = np.array(rec_volume)
    max_p = np.max(np.abs(rec_volume))
    min_p = np.min(np.abs(rec_volume))
    rec_vol_16 = np.array(np.round((np.abs(rec_volume) - min_p) /
                                   (max_p - min_p) * 2 ** 15), dtype=np.uint16)
    # save 4D volume
    # Change dimensions from NSl x Nv x Ns x Ns to Ns x Ns x NSl x Nv
    rec_vol_16 = np.transpose(rec_vol_16, [2, 3, 0, 1])
    file_name = os.path.join(root_rec_folder,
                             checkpoint + '_4D.nii.gz')
    nii_dat = nib.Nifti1Image(rec_vol_16, np.eye(4))
    nii_dat.to_filename(file_name)
