import numpy as np
import math
from disjoinset import DisjoinSet
import matplotlib.pyplot as plt
import time


def gaussian_dist(x, y):
    return np.linalg.norm(x - y, ord=2)


def kernel_gaussian_density(x):
    return math.exp(-0.5 * x.T @ x) / (2 * math.pi)


def get_density(x, y, f):
    m, n = x.shape
    z = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            z[i, j] = f(x[i, j], y[i, j])
    return z


class Denclue_Algorithm(object):
    def __init__(self, x, y, hypers):
        self.x = x
        self.y = y
        self.n = len(x)
        self.hypers = hypers
        assert (self.n == len(y))
        self.list_point = [np.array([self.x[i], self.y[i]]) for i in range(self.n)]
        self.local_maximum = []
        self.cluster_label = []
        self.is_out = []
        self.cluster_id = []

    def render_dens_fig(self, path="./dens_fig.png"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("Density")
        num_point = 50
        X = np.linspace(0, 10, num_point)
        Y = np.linspace(0, 10, num_point)
        X, Y = np.meshgrid(X, Y)
        Z = get_density(X, Y, lambda x, y: self.kernel_density_estimator(np.array([x, y])))
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
        plt.savefig(path)
        plt.show()

    def kernel_density_estimator(self, x):
        s = 0
        for p in self.list_point:
            s += kernel_gaussian_density((x - p) / self.hypers["H"])
        return s / (self.n * (self.hypers["H"] ** 2))

    def kde_gradient(self, x):
        s = np.array([0., 0.])
        for p in self.list_point:
            s += kernel_gaussian_density((x - p) / self.hypers["H"]) * (p - x)
        return s / ((self.hypers["H"] ** 4) * self.n)

    def _step(self, x):
        d = self.kde_gradient(x)
        return x + d * self.hypers["DELTA"] / np.linalg.norm(d, ord=2)

    def get_local_maximum(self, start):
        old = start
        for i in range(self.hypers["MAX_ITERATIONS"]):
            new = self._step(old)
            if self.kernel_density_estimator(new) < self.kernel_density_estimator(old):
                break
            old = new
        return old

    def hill_climb(self):
        for i in range(self.n):
            mx = self.get_local_maximum(self.list_point[i])
            self.local_maximum.append(mx)

    def merge_pt_same_cluster(self):
        ds = DisjoinSet(self.n)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if gaussian_dist(self.local_maximum[i], self.local_maximum[j]) < self.hypers["CONSIDER_CLOSE"]:
                    ds.merge(i, j)
        ds.arrange()
        new_local_maximum = []
        for position in ds.correspond_root:
            new_local_maximum.append(self.local_maximum[position])
        self.local_maximum = new_local_maximum
        for i in range(self.n):
            self.cluster_label.append(ds.cluster_label[i])

    def noise_cluster_flags(self):
        for lm in self.local_maximum:
            dens = self.kernel_density_estimator(lm)
            self.is_out.append(dens < self.hypers["XI"])

    def merge_cluster(self):
        ds = DisjoinSet(len(self.local_maximum))
        is_higher = [self.kernel_density_estimator(p) >= self.hypers["XI"] for p in self.list_point]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.cluster_label[i] != self.cluster_label[j] \
                        and is_higher[i] \
                        and is_higher[j] \
                        and (not self.is_out[self.cluster_label[i]]) \
                        and (not self.is_out[self.cluster_label[j]]) \
                        and gaussian_dist(self.list_point[i], self.list_point[j]) < self.hypers[
                    "CLUSTER_MERGE_DISTANCE"]:
                    ds.merge(self.cluster_label[i], self.cluster_label[j])
        ds.arrange()
        for i in range(len(self.local_maximum)):
            self.cluster_id.append(ds.cluster_label[i])

    def get_result(self):
        res = []
        for i in range(self.n):
            if self.is_out[self.cluster_label[i]]:
                res.append(-1)
            else:
                res.append(self.cluster_id[self.cluster_label[i]])

        no = [-1 for i in range(len(self.local_maximum))]
        cnt = 0
        for i in range(len(res)):
            if res[i] != -1:
                if no[res[i]] == -1:
                    no[res[i]] = cnt
                    cnt += 1
                res[i] = no[res[i]]
        return cnt, res

    def fit(self):
        print("Start fitting data")
        start = time.time()
        self.hill_climb()
        self.merge_pt_same_cluster()
        self.noise_cluster_flags()
        self.merge_cluster()
        print(f"Done after: {time.time() - start} seconds")
