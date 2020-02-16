import numpy as np
import matplotlib.pyplot as plt


def perturb(x):
    return (3.0 * np.random.random() - 1.0) * x + (
        2.0 * np.random.random() - 1.0
    ) * 0.05


def perturb2(x):
    x *= 3.0 * np.random.random() - 1.0
    x += (2.0 * np.random.random() - 1.0) * 0.05
    return x


def perturb_better(x):
    return np.random.uniform(-x - 0.05, 2 * x + 0.05)


if __name__ == "__main__":
    first = [perturb(2.0) for i in range(1000)]
    second = [perturb2(2.0) for i in range(1000)]

    print(np.array(first).mean(), np.array(first).std())
    print(np.array(second).mean(), np.array(second).std())
    plt.scatter(first, np.ones(len(first)))
    plt.scatter(second, 2 * np.ones(len(second)))
    plt.show()
