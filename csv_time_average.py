import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from scipy.ndimage import correlate
from datetime import datetime

folder_path = "dataset/for_train"
file_names = os.listdir(folder_path)
np.set_printoptions(threshold=np.inf)


# average the time by second
def parse_time_string(time_string):
    return datetime.strptime(time_string, "%H:%M:%S.%f")


def format_time_string(time):
    return time.strftime("%H:%M:%S")


def average_by_sec(time_arr, value_arr):
    time_objs = [parse_time_string(time_str) for time_str in time_arr]
    time_objs = [format_time_string(time_obj) for time_obj in time_objs]

    # create dictionary to save time and corresponding values
    sum_dict = {}
    count_dict = {}
    new_time_arr = []

    for i in range(len(time_objs)):
        new_time_str = time_objs[i]

        if new_time_str in sum_dict:
            sum_dict[new_time_str] += value_arr[i]
            count_dict[new_time_str] += 1
        else:
            sum_dict[new_time_str] = value_arr[i]
            count_dict[new_time_str] = 1
            new_time_arr.append(new_time_str)

    ####### test code #######
    # for key, value in sum_dict.items():
    #     print(type(value))  # class 'numpy.ndarray'
    #
    # for key, value in count_dict.items():
    #     print(type(value))  # class 'int'

    avg_value_arr = [sum_dict[new_time_str] / count_dict[new_time_str] for new_time_str in new_time_arr]
    return new_time_arr, avg_value_arr


def add_filter(value_arr):
    # filter operator
    h = np.ones((3, 3)) / 9.0
    f = np.ones((7, 7)) / 49.0
    filter = correlate(value_arr, h, mode='nearest')

    delta_value_arr = np.copy(value_arr)
    delta_value_arr[value_arr < 0.5] = 0
    delta_filter = np.copy(filter)
    delta_filter[delta_value_arr > 0] = delta_value_arr[delta_value_arr > 0]
    delta_filter[delta_value_arr == 0] *= 1.2

    def update(frame):
        image0 = np.reshape(value_arr[frame], (32, 64))
        image1 = np.reshape(filter[frame], (32, 64))
        image2 = np.reshape(delta_filter[frame], (32, 64))
        image3 = np.reshape(delta_filter[frame], (32, 64))
        im0.set_array(image0)
        im1.set_array(image1)
        im2.set_array(image2)
        im3.set_array(image3)

    fig, axs = plt.subplots(1, 5, figsize=(12, 4))

    for ax in axs:
        ax.set_aspect('equal', 'box')
        ax.axis('off')

    image0 = np.reshape(value_arr[0], (32, 64))
    image0 = np.flip(image0, axis=1)
    image0 = np.flip(image0, axis=0)

    image1 = np.reshape(filter[0], (32, 64))
    image1 = np.flip(image1, axis=1)
    image1 = np.flip(image1, axis=0)

    image2 = np.reshape(delta_filter[0], (32, 64))
    image2 = np.flip(image2, axis=1)
    image2 = np.flip(image2, axis=0)

    image3 = np.reshape(delta_filter[0], (32, 64))
    image3 = np.flip(image3, axis=1)
    image3 = np.flip(image3, axis=0)

    im0 = axs[0].imshow(image0, cmap='jet', norm=colors.Normalize(vmin=0, vmax=1))
    axs[0].set_title('original data')
    axs[0].invert_xaxis()
    axs[0].invert_yaxis()

    im1 = axs[1].imshow(image1, cmap='jet', norm=colors.Normalize(vmin=0, vmax=1))
    axs[1].set_title('original data filter')
    axs[1].invert_xaxis()
    axs[1].invert_yaxis()

    im2 = axs[2].imshow(image2, cmap='jet', norm=colors.Normalize(vmin=0, vmax=1))
    axs[2].set_title('recovery peak data')
    axs[2].invert_xaxis()
    axs[2].invert_yaxis()

    im3 = axs[3].imshow(image3, cmap='jet', interpolation='bilinear', norm=colors.Normalize(vmin=0, vmax=1))
    axs[3].set_title('interpolated data')
    axs[3].invert_xaxis()
    axs[3].invert_yaxis()

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im0, cax=cax)

    plt.subplots_adjust(wspace=0.05)

    ani = animation.FuncAnimation(fig, update, frames=len(value_arr), interval=500, blit=False)

    plt.show()


def dynamic_pic(value_arr):
    def update(frame):
        image = np.reshape(value_arr[frame], (32, 64))
        im.set_array(image)

    fig, ax = plt.subplots()

    ax.set_aspect('equal', 'box')
    image = np.reshape(value_arr[0], (32, 64))
    image = np.flip(image, axis=1)
    image = np.flip(image, axis=0)
    im = ax.imshow(image, cmap='jet', interpolation='bilinear', norm=colors.Normalize(vmin=0, vmax=1))
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.axis('off')

    ani = animation.FuncAnimation(fig, update, frames=len(value_arr), interval=500, blit=False)
    cbar = plt.colorbar(im)

    plt.show()


def static_pic(value_arr, index):
    fig, ax = plt.subplots()

    ax.set_aspect('equal', 'box')
    image = np.reshape(value_arr[index], (32, 64))
    image = np.flip(image, axis=1)
    image = np.flip(image, axis=0)
    im = ax.imshow(image, cmap='jet', interpolation='bicubic', norm=colors.Normalize(vmin=0, vmax=1))
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.axis('off')
    cbar = plt.colorbar(im)

    plt.show()


def pressure_norm(value_arr):
    value_arr = np.array(value_arr)

    min_value = np.min(value_arr)
    max_value = np.max(value_arr)

    value_arr = (value_arr - min_value) / (max_value - min_value)
    return value_arr


for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    if file_path.endswith(".CSV"):
        with open(file_path, 'r') as file:

            time_arr = []
            value_arr = []

            lines = file.readlines()[1:]

            for line in lines:
                line = line.strip()
                line = line.split(',')

                time_str = line[0].split(' ')[1]
                time_arr.append(time_str)

                value_str = line[1:]
                value_per_time = [int(value) for value in value_str]
                value_arr.append(value_per_time)

            ######## do normalization to [0, 1] #######
            value_arr = pressure_norm(value_arr)
            # print(time_arr)  # test code
            # print(value_arr)  # test code
            # print(len(time_arr))  # test code
            # print(value_arr[50])  # test code

            ######## plot dynamic or static pictures #######
            dynamic_pic(value_arr)
            # static_pic(value_arr, 50)
            # add_filter(value_arr)

            ######## do time average #######
            new_time_arr, avg_value_arr = average_by_sec(time_arr, value_arr)

            # print(len(new_time_arr))  # test code
            # print(len(avg_value_arr))  # test code
            print(new_time_arr)  # test code
            # print(avg_value_arr)  # test code
            # static_pic(avg_value_arr, 50)  # test code
