import numpy as np
import matplotlib.pyplot as plt


def kernel_function(mean, point, h=0.1):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-(point - mean) ** 2 / (2*h**2))


def kernel_density_estimator(means, h=0.1):
    means = sorted(means)
    point = np.linspace(means[0] - 3, means[-1] + 3, 100 * 5)
    final = np.array([0.0] * 500)
    for mean in means:
        final += kernel_function(mean, point, h)
    return final / len(means), point


def visualize_different_h():
    means = [1, 3.5, 6, 7.5, 8]
    curves = []
    points = []
    hs = np.arange(0.1, 1.1, 0.1)
    row = 2
    fig, ax = plt.subplots(nrows=row, ncols=(len(hs) + 1) // row, figsize=(15, 8))
    ax = ax.flatten()
    for idx, h in enumerate(hs):
        for mean in means:
            point = np.linspace(mean - 3, mean + 3, 100)
            points.append(point)
            curve = kernel_function(mean, point, h=h)

            curves.append(curve)

        ax[idx].scatter(means, [0] * len(means), color='red')

        final, point = kernel_density_estimator(means, h=h)
        ax[idx].plot(point, final)
        ax[idx].set_xlabel(f"h={h:.2f}")

        ax[idx].set_xticks([])
        ax[idx].set_yticks([])

    plt.suptitle("Kernel Density Estimator with different smoothing", weight='bold', color='black')
    plt.show()


def visualize_kde(h=0.5):
    means = [1, 3.5, 6, 7.5, 8]

    points = []
    curves = []

    for mean in means:
        point = np.linspace(mean - 3, mean + 3, 100)
        points.append(point)
        curve = kernel_function(mean, point, h=h)

        curves.append(curve)

    sct = plt.scatter(means, [0]*len(means), color='red', label="point")
    for idx, curve in enumerate(curves):
        if idx == 0:
            plt.plot(points[idx], curve, label='kernel', linestyle='dashed', color='black')
        else:
            plt.plot(points[idx], curve, linestyle='dashed', color='black')

    final, point = kernel_density_estimator(means, h=h)
    plt.plot(point, final*5, label='KDE', color='blue')
    plt.legend()
    plt.title(f"Smoothing: h={h}")
    plt.savefig(f"output/KDE_1D_{h}.png")
    plt.show()


if __name__ == '__main__':
    visualize_kde(h=0.75)
