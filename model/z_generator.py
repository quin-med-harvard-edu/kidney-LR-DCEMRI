import numpy as np
from model.pca_based_z.divide_phase import divide_phase_frame_ma


def gen_with_pca(k_space: np.ndarray,
                 spv: int,
                 len_segment: float,
                 n_dim: int,
                 start_bias: np.ndarray,
                 max_num_phase: int = 10,
                 k_center_radius: int = 0,
                 phase_th: float = 0.01):
    """ Generate equi-length sub line segments with pseudo random angles where each
        line segment is annotated with PCA based clustering.
            Args:
                k_space(ndarray[complex]): num_sample x num_channel x num_spoke data
                spv(int): number of spokes to generate a reconstructed image
                len_segment(float): l2 norm of each subsegment in the line segment
                n_dim(int):dimensionality of the segmented lines
                start_bias(ndarray(float)): starting point of the segmented lines
                max_num_phase(int): number of phases
                k_center_radius(int): number of k space samples used for phase division
                phase_th(float): phase threshold level to determine a new phase
            Return:
                samples(ndarray(float)):
                """
    labels, mean_ = divide_phase_frame_ma(k3n=np.array(k_space),
                                          spv=spv,
                                          max_num_phase=max_num_phase,
                                          k_center_radius=k_center_radius,
                                          phase_th=phase_th)
    labels = labels.astype(np.int32)

    n_sample = len(labels)
    samples = []
    for idx_label in list(np.unique(labels)):
        el_seg = len(np.where(labels == idx_label)[0])
        angle = np.random.uniform(-5, 5, n_dim)
        angle = angle / np.linalg.norm(angle)
        tmp_ = start_bias + np.outer(np.arange(el_seg) / n_sample * len_segment,
                                     angle)
        start_bias = tmp_[-1]
        samples.append(tmp_)

    samples = np.concatenate(samples, axis=0)
    samples = samples[:n_sample, :]
    return samples


def gen_segmented_line_segment(len_segment: float,
                               el_seg: int,
                               n_sample: int,
                               n_dim: int,
                               start_bias: np.ndarray):
    """ Generate equi-length sub line segments with pseudo random angles
        Args:
            len_segment(float): l2 norm of each subsegment in the line segment
            el_seg(int): number of elements in each segment
            n_sample(int): number of samples to be generated
            n_dim(int):dimensionality of the segmented lines
            start_bias(ndarray(float)): starting point of the segmented lines
        Return:
            samples(ndarray(float)):
            """
    n_seg = int(np.ceil(n_sample / el_seg))
    samples = []

    for idx_ in range(n_seg):
        angle = np.random.uniform(-5, 5, n_dim)
        angle = angle / np.linalg.norm(angle)
        tmp_ = start_bias + np.outer(np.arange(el_seg) / n_sample * len_segment,
                                     angle)
        start_bias = tmp_[-1]
        samples.append(tmp_)

    samples = np.concatenate(samples, axis=0)
    samples = samples[:n_sample, :]

    return samples


def gen_line_segment(len_segment: float,
                     n_sample: int,
                     n_dim: int,
                     start_bias: float):
    angle = np.random.randint(1, 10, n_dim)
    angle = angle / np.linalg.norm(angle)
    samples = start_bias + np.outer(np.arange(n_sample) / n_sample * len_segment, angle)

    return samples


def divide_line_segment(min_p: float,
                        max_p: float,
                        n_sample: int):
    samples = []
    for idx in range(n_sample):
        samples.append(min_p + (max_p - min_p) / n_sample * idx)

    return np.array(samples)
