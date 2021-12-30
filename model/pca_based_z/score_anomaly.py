import numpy as np
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, ConstantKernel as C
from scipy.ndimage import gaussian_filter1d
from copy import copy

from model.pca_based_z.helpers import k_spc_center_fid


def moving_average(a: np.ndarray,
                   n: int = 3):
    """ Args:
            a(ndarray[]): a 2 dimensional array, 1st dimension represents time serie dim.
                2nd dimension represents the number of time series
            n(int): length of the window
        Return:
            a_ma(ndarray[]): a 2 dimensional array in the form of input
            """
    ret = np.concatenate([np.zeros((int(n / 2), a.shape[1])), a,
                          np.zeros((int(n / 2), a.shape[1]))], axis=0)
    ret = np.cumsum(ret, dtype=float, axis=0)
    ret[int(n / 2):, :] = ret[int(n / 2):, :] - ret[:-int(n / 2), :]
    return ret[int(n / 2):-int(n / 2), :] / n


class GaussianProcessFID:
    """ FID based outlier detector using Gaussian processes
        Attr:
            list_gp(list[GaussianProcessRegressor]): a list of sklearn Gaussian
                process regressor functions
            num_gp(int): number of channels = number of processes
            list_domain_x(list[ndarray(complex)])
            list_domain_y(list[ndarray(complex)])
            feature(ndarray(float)): score that is used for outlier rejection
            kernel(sklearnKernel): a sklearn kernel for the Gaussian process.
                Processes share the same kernel. """

    def __init__(self,
                 kernel=C(0.8, (1e-3, 1e3)) * RBF(1.5, (1e-2, 1e5))):

        self.list_gp = []
        self.num_gp = 0
        self.list_domain_x = []
        self.list_domain_y = []

        self.feature = None
        self.kernel = kernel

    # TODO: This is a static method, maybe generalize it

    def fit(self, k3n, radius_kspc_c=1):
        """ Extracts the k space center and fits the Gaussian processes to each channel
            Args:
                 k3n(ndarray[complex])
        """
        self.feature = k_spc_center_fid(k3n, radius_kspc_c)
        [num_sample, num_coil, num_spoke] = k3n.shape
        feature = k_spc_center_fid(k3n)

        for idx_c in tqdm(range(num_coil)):
            domain_x = np.atleast_2d(np.arange(feature.shape[0])).T
            domain_y = np.atleast_2d(feature[:, idx_c, :])
            self.list_domain_x.append(domain_x)
            self.list_domain_y.append(domain_y)

            self.list_gp.append(GaussianProcessRegressor(kernel=self.kernel,
                                                         alpha=np.var(domain_y),
                                                         n_restarts_optimizer=5))
            self.list_gp[idx_c].fit(domain_x, domain_y)

        self.num_gp = len(self.list_gp)

    def _predict(self, list_domain_x):
        list_y_hat, list_sigma_hat = [None for idx in range(self.num_gp)], \
                                     [None for idx in range(self.num_gp)]
        for idx_c in tqdm(range(self.num_gp)):
            list_y_hat[idx_c], list_sigma_hat[idx_c] = self.list_gp[idx_c].predict(
                list_domain_x[idx_c], return_std=True)

        return np.array(list_y_hat), np.array(list_sigma_hat)

    def score(self, list_domain_y=None, list_domain_x=None):
        if not (list_domain_y or list_domain_x):
            list_domain_y = copy(self.list_domain_y)
            list_domain_x = copy(self.list_domain_x)

        list_y_hat, list_sigma_hat = self._predict(list_domain_x)

        list_like_c = [None for idx in range(self.num_gp)]
        for idx_c in tqdm(range(self.num_gp)):
            list_like_c[idx_c] = -0.5 * (np.linalg.norm(
                np.squeeze(list_domain_y[idx_c]) - np.squeeze(list_y_hat[idx_c]),
                axis=1) / list_sigma_hat[idx_c]) ** 2

        list_like_c = np.array(list_like_c)
        score = 0
        for idx_c in tqdm(range(self.num_gp)):
            score += -list_like_c[idx_c]

        return score


def fid_center_corr(k3n: np.ndarray,
                    step: int):
    """ FID based outlier detector using cross correlation within a window
        Args:
            k3n(ndarray[complex]): k space data
            step(int): radius of the window
        Return:
            score(ndarray[float]): outlier score for each sample """
    k_center = np.squeeze(k_spc_center_fid(k3n, 0))
    feature = moving_average(np.abs(k_center - np.min(k_center)) /
                             (np.max(k_center) - np.min(k_center)), 34)
    score = []
    for idx in tqdm(range(feature.shape[0])):
        _corr = -1
        for idx_w in range(np.maximum(0, idx - step),
                           np.minimum(feature.shape[0], idx + step)):
            _tmp = \
                np.corrcoef(np.squeeze(feature[idx, :]), np.squeeze(feature[idx_w, :]))[
                    0, 1]
            _corr += _tmp
        score.append(1 - (_corr / (np.minimum(feature.shape[0], idx + step)
                                   - np.maximum(0, idx - step))))

    return np.array(score)


def fid_center_ma(k3n: np.ndarray,
                  sig: int = 17,
                  radius: int = 2):
    """ FID based outlier detector using a simple moving average model
            Args:
                k3n(ndarray[complex]): k space data
                sig(int): radius of the moving average kernel
                radius(int): k space center radius
            Return:
                score(ndarray[float]): outlier score for each sample
                mean_(ndarray[complex]): model mean """
    k_center = k_spc_center_fid(k3n, radius=radius)
    x_ = np.abs(k_center - np.min(k_center)) / (np.max(k_center) - np.min(k_center))
    mean_ = gaussian_filter1d(x_, sigma=sig, axis=0)
    std_ = np.std(x_)

    score = np.sum(0.5 * np.linalg.norm(np.power((x_ - mean_) / std_, 2),
                                        axis=-1), axis=-1)

    return score, mean_
