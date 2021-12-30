import numpy as np


def partition_volume(dat_slice: dict,
                     spv: int,
                     start_smp: int = 0,
                     cutoff_smp: int = None):
    """ Partition input data into volumes
        Args:
            dat_slice(dict): data dictionary
            spv(int): spokes per each volume
            start_smp(int): starting sample for acquisition
            cutoff_smp(int): cutoff sample value. E.g. 1326 for 2 minute data
        Return
            div_k_space(ndarray[Nv x Ns x Nc x spv]): k space data
            div_k_samples(ndarray[Nv x Ns x spv]): sampling trajectories
            coil_p(ndarray(Ns x Ns x Nc)): coil profiles
            div_sqrt_dcf(ndarray[Nv x Ns x spv]):
            where [Nv x Ns x Nc x spv] are number of volumes, number of samples, number
            of coils and spoke per volume respectively
            """
    [num_sample, num_coil, num_spoke] = dat_slice['k3n'].shape
    if cutoff_smp:
        num_vol = int(np.floor((cutoff_smp - start_smp) / spv))
    else:
        num_vol = int(np.floor((num_spoke - start_smp) / spv))
    k_3 = (dat_slice['k3n'] * np.power(10, 2))
    coil_p = dat_slice['coilprofile'] / np.max(np.abs(dat_slice['coilprofile']))

    div_k_space = np.array(
        [k_3[:, :, start_smp + el * spv:start_smp + (el + 1) * spv]
         for el in range(num_vol)])
    div_k_samples = np.array(
        [dat_slice['k_samples'][:, start_smp + el * spv:start_smp + (el + 1) * spv]
         for el in range(num_vol)])
    div_sqrt_dcf = np.array(
        [np.sqrt(dat_slice['dcf'])[:, start_smp + el * spv:start_smp + (el + 1) * spv]
         for el in range(num_vol)])

    return div_k_space, div_k_samples, coil_p, div_sqrt_dcf


def partition_sliding(dat_slice,
                      spv,
                      start_smp: int = 0,
                      cutoff_smp: int = None):
    """ Partition input data into volumes
        Args:
            dat_slice(dict): data dictionary
            spv(int): spokes per each volume
            start_smp(int): starting sample for acquisition
            cutoff_smp(int): cutoff sample value. E.g. 1326 for 2 minute data
        Return
            div_k_space(ndarray[Nv x Ns x Nc x spv]): k space data
            div_k_samples(ndarray[Nv x Ns x spv]): sampling trajectories
            coil_p(ndarray(Ns x Ns x Nc)): coil profiles
            div_sqrt_dcf(ndarray[Nv x Ns x spv]):
            where [Nv x Ns x Nc x spv] are number of volumes, number of samples, number
            of coils and spoke per volume respectively
            """

    [_, _, num_spoke] = dat_slice['k3n'].shape

    if cutoff_smp:
        num_vol = int(np.floor((cutoff_smp - start_smp) - spv))
    else:
        num_vol = int(np.floor(num_spoke - start_smp - spv))

    k_3 = (dat_slice['k3n'] * np.power(10, 2))
    coil_p = dat_slice['coilprofile'] / np.max(np.abs(dat_slice['coilprofile']))

    div_k_space = np.array(
        [k_3[:, :, start_smp + el:start_smp + el + spv] for el in range(num_vol)])
    div_k_samples = np.array(
        [dat_slice['k_samples'][:, start_smp + el:start_smp + el + spv] for el in
         range(num_vol)])
    div_sqrt_dcf = np.array(
        [np.sqrt(dat_slice['dcf'])[:, start_smp + el:start_smp + el + spv] for el in
         range(num_vol)])
    return div_k_space, div_k_samples, coil_p, div_sqrt_dcf
