
import numpy as np
from collections import deque
from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, hist_len=128, auto_plot_interval=100):
        self.history = deque(maxlen=hist_len)
        self.auto_plot_interval = auto_plot_interval
        self.counter = 0
        self.fig = plt.figure()
        self.ax_loss = self.fig.add_subplot(121)
        self.ax_acc = self.fig.add_subplot(122)

    def append(self, loss_g, loss_d, acc_g, acc_d):
        self.history.append([loss_g, loss_d, acc_g, acc_d])
        self.counter -= 1
        if self.counter <= 0:
            self.counter = self.auto_plot_interval
            self.plot()

    def plot(self):
        h_array = np.array(self.history)
        self.ax_loss.cla()
        self.ax_acc.cla()
        self.ax_loss.plot(h_array[:, 0], color="r", label="g")
        self.ax_loss.plot(h_array[:, 1], color="b", label="d")
        self.ax_acc.plot(h_array[:, 2], color="r", label="g")
        self.ax_acc.plot(h_array[:, 3], color="b", label="d")
        self.ax_loss.set_title("loss")
        self.ax_loss.set_ylim([-0.1, 2])
        self.ax_loss.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
        self.ax_acc.set_title("acc")
        self.ax_acc.set_ylim([-0.1, 1.1])
        self.ax_acc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
        plt.pause(0.01)


def plot_images(images, labels=None, rows=1, cols=None):

    if cols is None:
        cols = len(images) // rows

    assert len(images) >= rows*cols

    fig = plt.figure()
    for i in range(rows):
        for j in range(cols):
            im = images[i * cols + j]
            ax = fig.add_subplot(rows, cols, i * cols + j + 1)
            ax.imshow(im.squeeze())
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_xmargin(0)
            ax.set_ymargin(0)
            if labels is not None:
                ax.set_title(labels[i * cols + j])
    plt.tight_layout()
    plt.show()
