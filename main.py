import matplotlib.pyplot as plt
import numpy as np
from denclue import Denclue_Algorithm
import warnings
warnings.filterwarnings('ignore')


def scatter(x, y, cluster_label, save_name, number_cluster):
    n = len(cluster_label)
    plt.figure(figsize=(10, 5))
    sct = plt.scatter(x, y, c=[cluster_label[i] for i in range(n)], cmap='turbo')
    plt.legend(*sct.legend_elements(), loc='best')
    plt.title(f"Number of cluster: {number_cluster}")
    plt.savefig(save_name)
    plt.show()


def read_data(filename):
    with open(filename) as fh:
        lines = fh.readlines()
        x1, x2 = [], []
        for line in lines:
            a, b = map(float, line.strip('\n').split())
            x1.append(a)
            x2.append(b)
        return np.array(x1), np.array(x2)


if __name__ == '__main__':
    hypers = {"H": 0.1,  # Smoothing parameter
              "CLUSTER_MERGE_DISTANCE": 0.5,  # Cluster linkage distance
              "CONSIDER_CLOSE": 0.2,  # Neighbor point consider same cluster
              "DELTA": 0.2,  # Learning rate
              "XI": 0.01,   # Denoising parameter
              "MAX_ITERATIONS": 50}

    # diff_h = [0.1, 0.2, 0.5, 1.5]
    diff_h = [0.5]
    for h in diff_h:
        hypers['H'] = h
        for data in range(2, 4):
            x1, x2 = read_data(f"Data/{data}.txt")
            denclue = Denclue_Algorithm(x1, x2, hypers)
            denclue.fit()

            denclue.render_dens_fig(f"output/{data}_density_h_{hypers['H']}.png")
            num_cluster, cluster_label = denclue.get_result()

            scatter(x1, x2, cluster_label, f"output/{data}_cluster_h_{hypers['H']}.png", number_cluster=num_cluster)
