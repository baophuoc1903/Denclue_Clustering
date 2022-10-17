import matplotlib.pyplot as plt
import numpy as np
from denclue import Denclue_Algorithm


def scatter(x, y, cluster_label, save_name):
    n = len(cluster_label)
    plt.scatter(x, y, c=[("C" + str(cluster_label[i])) if cluster_label[i] != -1 else "0.5" for i in range(n)])
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
    hypers = {"H": 0.5,  # Smoothing parameter
              "CLUSTER_MERGE_DISTANCE": 0.5,  # Cluster linkage distance
              "CONSIDER_CLOSE": 0.2,
              "DELTA": 0.2,  # Speed of convergence
              "XI": 0.01,  # Denoising parameter
              "MAX_ITERATIONS": 50}

    for data in range(1, 4):
        x1, x2 = read_data(f"Data/{data}.txt")
        denclue = Denclue_Algorithm(x1, x2, hypers)
        denclue.fit()

        denclue.render_dens_fig(f"output/{data}_density.png")
        num_cluster, cluster_label = denclue.get_result()

        scatter(x1, x2, cluster_label, f"output/{data}_cluster.png")
