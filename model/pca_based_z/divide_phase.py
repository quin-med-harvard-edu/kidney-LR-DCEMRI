import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from model.pca_based_z.score_anomaly import fid_center_ma

from sklearn.decomposition import PCA


def divide_phase_frame_ma(k3n: np.ndarray,
                          spv: int = 34,
                          max_num_phase: int = 9,
                          k_center_radius: int = 2,
                          phase_th: float = 0.02):
    """ Divides k space trajectory into phases that are possibly effected by contrast
        and/or major motion.
        Args:
            k3n(ndarray[complex]): num_sample x num_channel x num_spoke data
            max_num_phase(int): a limit on number of phases to be return
            spv(int): radius of the moving average kernel
            k_center_radius(int): radius around k space center
            outlier_th(float): thresholding weight
            phase_th(float):  threshold to detect number of phases using k-means score
        Return:
            phase_labels(ndarray[int]): phase labels for each k space line. If valued
                np.nan, it is an outlier and rejected in the model.
            score(ndarray[float]):
         """

    num_vol = int(np.floor(k3n.shape[-1] / spv))
    score, mean_ = fid_center_ma(k3n=k3n, sig=spv, radius=k_center_radius)
    mean_ = np.reshape(mean_, (mean_.shape[0], -1))

    pca = PCA(n_components=1)
    ft_ = np.squeeze(pca.fit_transform(np.squeeze(mean_)))
    ft_ = (ft_ - np.min(ft_)) / (np.max(ft_) - np.min(ft_))

    score_km = []
    list_num_phase = list(np.arange(1, max_num_phase))
    for num_phase in list_num_phase:
        k_means = KMeans(n_clusters=num_phase, random_state=5).fit(
            np.expand_dims(ft_, axis=-1))
        score_km.append(k_means.score(np.expand_dims(ft_, axis=-1)))

    score_km = np.array(score_km)
    num_phase = list_num_phase[
        int(np.sum(np.around(score_km / score_km[0], 3) >= phase_th)) - 1]
    k_means = KMeans(num_phase, random_state=5).fit(np.expand_dims(ft_, axis=-1))

    n_lim = np.minimum(int(np.min([np.sum(k_means.labels_ == __tmp) for __tmp in
                                   list(np.unique(k_means.labels_))])), spv * 6)
    neigh = KNeighborsClassifier(n_neighbors=n_lim)
    neigh.fit(np.expand_dims(np.arange(len(ft_)) + 0.1, axis=-1), k_means.labels_)

    label_phase = neigh.predict(np.expand_dims(np.arange(len(ft_)), axis=-1))
    label_phase = np.array([np.bincount(label_phase[el * spv:(el + 1) * spv]).argmax()
                            for el in range(num_vol)])
    idx_change = np.where((label_phase[1:] > label_phase[:-1]) +
                          (label_phase[1:] < label_phase[:-1]))[0]
    label_phase = np.zeros(label_phase.shape)
    for dd_ in list(idx_change):
        label_phase[dd_:] += 1

    return label_phase, ft_
