import numpy as np


def info_c_threshold(score: np.ndarray,
                     lam: float = 3.0):
    """ Computes information criterion based score threshold
        Args:
            score(ndarray[float]): Real valued score vector of size K x None
            lam(float): inlier weight for the computation, the higher the more inliers
                are enforced
        Return:
            score_threshold(float): anomaly detection threshold for the vector score """

    tmp_val = np.sort(score)
    tmp_sum = np.cumsum(tmp_val)
    tmp_sum /= tmp_sum[-1]
    selection_score = tmp_sum - lam * np.log2(1 + np.arange(tmp_val.shape[0]))
    if not np.where(np.ediff1d(selection_score) > 0)[0].size == 0:
        score_threshold = tmp_val[np.where(np.ediff1d(selection_score) > 0)[0][0]] - 1
    else:
        score_threshold = tmp_val[-1] - 0.1
    return score_threshold


# TODO: the radius is not an actual radius. In our application the center is shifted
def k_spc_center_fid(k_space: np.ndarray,
                     radius: int = 1):
    """ Extracts mean extracted center of k_space for the data around a radius
        Args:
            k_space(ndarray[complex]): complex valued k-space data for a single slice
                of size num_sample x num_coil x num_spoke
            radius(int): wrapping index
        Return:
            zm_cntr_x(ndarray[complex]): center of k-space of size num_spoke x
                num_coil x num_feature where num_feature is determined by radius """
    [num_sample, num_coil, num_spoke] = k_space.shape
    # Get center of k-space with the interval
    if radius == 0:
        coil_ksp = np.expand_dims(k_space[int(num_sample / 2), :, :], 0)
    else:
        coil_ksp = k_space[int(num_sample / 2) - radius:
                           int(num_sample / 2) + radius, :, :]

    cntr_x = np.transpose(np.abs(coil_ksp))
    zm_cntr_x = cntr_x - np.mean(cntr_x, 0)
    return zm_cntr_x
