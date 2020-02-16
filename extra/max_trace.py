import numpy as np
import matplotlib.pyplot as plt


def update_trace_linear(trace, spike, alpha, decay):
    trace *= decay
    trace += alpha * spike
    return trace


def compute_final_trace(spikes, alpha, decay):
    final = (spikes * decay ** np.arange(spikes.size - 1, -1, -1)) * alpha
    return final.sum()


if __name__ == "__main__":
    time = 50
    alpha = 0.6
    decay = 0.1
    trace = 0.0
    trace_list = [trace]
    spikes = np.ones(time)

    for i in range(spikes.size):
        trace = update_trace_linear(trace, spikes[i], alpha, decay)
        trace_list.append(trace)

    print(trace_list[-1], compute_final_trace(spikes, alpha, decay))

    plt.plot(range(time + 1), trace_list)
    plt.show()
