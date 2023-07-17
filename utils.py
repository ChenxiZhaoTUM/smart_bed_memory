import math, re, os
import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import colors


# compute learning rate with decay in second half
def computeLR(i, epochs, minLR, maxLR):
    if i < epochs * 0.5:
        return maxLR
    e = (i / float(epochs) - 0.5) * 2.
    # rescale second half to min/max range
    fmin = 0.
    fmax = 6.
    e = fmin + e * (fmax - fmin)
    f = math.pow(0.5, e)
    return minLR + (maxLR - minLR) * f


def makeDirs(directoryList):
    for directory in directoryList:
        if not os.path.exists(directory):
            os.makedirs(directory)


# def imageOut(filename, _input, _output, _target, max, min, saveTargets=False):
#     output = np.copy(_output)
#     fig, ax = plt.subplots()
#
#     ax.set_aspect('equal', 'box')
#     output_image = np.reshape(output, (32, 64))
#     im = ax.imshow(output_image, cmap='jet', vmin=min, vmax=max)
#     ax.axis('off')
#     cbar = plt.colorbar(im)
#
#     save_path = os.path.join(filename)
#     plt.savefig(save_path)
#
#     if saveTargets:
#         target = np.copy(_target)
#
#         target_image = np.reshape(target, (32, 64))
#         im2 = ax.imshow(target_image, cmap='jet', vmin=min, vmax=max)
#         # ax2.axis('off')
#
#         save_path2 = filename + "_target.png"
#         plt.savefig(save_path2)


def imageOut(filename, _input, _output, _target, max_val, min_val, saveTargets=False):
    output = np.copy(_output)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.set_aspect('equal', 'box')
    output_image = np.reshape(output, (32, 64))
    im1 = ax1.imshow(output_image, cmap='jet', vmin=min_val, vmax=max_val)
    ax1.axis('off')
    cbar1 = fig.colorbar(im1, ax=ax1)

    ax2.set_aspect('equal', 'box')
    input_strip = np.array(_input[-8:])
    input_strip_image = np.repeat(input_strip, repeats=2, axis=0)
    im2 = ax2.imshow(input_strip_image, cmap='jet')
    ax2.axis('off')
    cbar2 = fig.colorbar(im2, ax=ax2)

    save_path = os.path.join(filename)
    plt.savefig(save_path)
    plt.close(fig)

    if saveTargets:
        target = np.copy(_target)
        target_image = np.reshape(target, (32, 64))

        fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 5))

        ax3.set_aspect('equal', 'box')
        im3 = ax3.imshow(target_image, cmap='jet', vmin=min_val, vmax=max_val)
        ax3.axis('off')
        cbar3 = fig.colorbar(im3, ax=ax3)

        ax4.set_aspect('equal', 'box')
        im4 = ax4.imshow(input_strip_image, cmap='jet')
        ax4.axis('off')
        cbar4 = fig.colorbar(im4, ax=ax4)

        save_path2 = filename + "_target.png"
        plt.savefig(save_path2)

        plt.close(fig)


def log(file, line, doPrint=True):
    f = open(file, "a+")
    f.write(line + "\n")
    f.close()
    if doPrint:
        print(line)


# reset log file
def resetLog(file):
    f = open(file, "w")
    f.close()
