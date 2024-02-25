import os
import re
from datetime import datetime
import data_preprocessing_for_Nils as dp
from data_preprocessing_for_Nils import PressureDataset
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def parse_time_string(time_string):
    return datetime.strptime(time_string, "%H:%M:%S.%f")


def format_time_string(time):
    return time.strftime("%H:%M:%S")


def format_time_string_half_second(time):
    return time.strftime("%H:%M:%S.%f")


def round_to_nearest_half_second_down(time_obj):
    if time_obj.microsecond >= 500000:
        time_obj = time_obj.replace(microsecond=500000)
    else:
        time_obj = time_obj.replace(microsecond=0)

    return time_obj


def average_by_sec(required_time, time_arr, value_arr):
    time_objs = [parse_time_string(time_str) for time_str in time_arr]

    if (required_time == 1):
        time_objs = [format_time_string(time_obj) for time_obj in time_objs]
    else:
        time_objs = [round_to_nearest_half_second_down(time_obj) for time_obj in time_objs]
        time_objs = [format_time_string_half_second(time_obj) for time_obj in time_objs]

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

    avg_value_arr = [sum_dict[new_time_str] / count_dict[new_time_str] for new_time_str in new_time_arr]
    avg_value_arr = np.array(avg_value_arr)

    new_time_arr = np.array(new_time_arr)

    return new_time_arr, avg_value_arr


######## save data in arrays ########
def deal_with_txt_files(file_path):
    with open(file_path, 'r', errors='ignore') as file:
        lines = file.readlines()

        time_arr_txt = []
        value_arr_txt = []

        for line in lines:
            line = line.strip()

            # ignore the line not including []
            if "[" not in line and "]" not in line:
                continue

            time_start = line.find("[")
            time_end = line.find("]")

            value_str = line[time_end + 1:]
            value_str = re.sub(r'[^a-zA-Z0-9\s]', '', value_str)
            values = value_str.split()

            if len(values) != 20:
                continue

            time_str = line[time_start + 1:time_end]
            time_str = re.sub(r'[^a-zA-Z0-9:.]', '', time_str)
            time_arr_txt.append(time_str)

            value_per_time = [int(value, 16) for value in values]  # convert hexadecimal to decimal
            value_arr_txt.append(value_per_time)

        value_arr_txt = np.array(value_arr_txt)
        return time_arr_txt, value_arr_txt


def deal_with_csv_files(csv_file_path):
    with open(csv_file_path, 'r', errors='ignore') as file:
        time_arr_csv = []
        value_arr_csv = []

        lines = file.readlines()[1:]

        for line in lines:
            line = line.strip()
            line = line.split(',')

            time_str = line[0].split(' ')[1]
            time_arr_csv.append(time_str)

            value_str = line[1:]
            value_per_time = [int(value) for value in value_str]
            value_arr_csv.append(value_per_time)

        value_arr_csv = np.array(value_arr_csv)
        return time_arr_csv, value_arr_csv


def reshape_output_value(value_arr):
    new_reshape_value_arr = []
    for value in value_arr:
        value = np.reshape(value, (32, 64))
        new_reshape_value_arr.append(value)

    new_reshape_value_arr = np.array(new_reshape_value_arr)
    return new_reshape_value_arr


def save_data_from_files(isTest=False):
    if isTest:
        folder_path = "./dataset/for_test/"
        file_names = os.listdir("./dataset/for_test/")
    else:
        folder_path = "./dataset/for_train"
        file_names = os.listdir("./dataset/for_train")

    all_common_data = {}
    print()
    print("The operating files:")

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)

        if file_path.endswith(".TXT"):
            file_name_without_extension = os.path.splitext(file_name)[0]
            csv_file_name = file_name_without_extension + ".CSV"
            csv_file_path = os.path.join(folder_path, csv_file_name)

            if not os.path.isfile(csv_file_path):
                continue

            print(file_name_without_extension)  # test code

            time_arr_txt, value_arr_txt = deal_with_txt_files(file_path)
            time_arr_csv, value_arr_csv = deal_with_csv_files(csv_file_path)

            ######## do time average #######
            avg_time_arr_txt, avg_value_arr_txt = average_by_sec(0.5, time_arr_txt, value_arr_txt)
            avg_time_arr_csv, avg_value_arr_csv = average_by_sec(0.5, time_arr_csv, value_arr_csv)

            avg_value_arr_csv = reshape_output_value(avg_value_arr_csv)
            # print(avg_value_arr_csv.shape)  # test code

            ######## save all data in dictionary #######
            all_input_data = {}
            all_target_data = {}

            for i in range(len(avg_time_arr_txt)):
                time = avg_time_arr_txt[i]
                all_input_data[time] = avg_value_arr_txt[i]

            for i in range(len(avg_time_arr_csv)):
                time = avg_time_arr_csv[i]
                all_target_data[time] = avg_value_arr_csv[i]

            for time, input_data in all_input_data.items():
                if time in all_target_data:
                    target_data = all_target_data[time]
                    common_data_id = len(all_common_data)
                    all_common_data[common_data_id] = {
                        'time': time,
                        'input_data': input_data,
                        'target_data': target_data
                    }

    print()
    print("Number of data loaded:", len(all_common_data))

    return all_common_data


inputs = []
targets = []
times = []

all_common_data = save_data_from_files()

inputs_pressure = []

for common_data_id in range(len(all_common_data)):
    value = all_common_data[common_data_id]
    input_data = value['input_data']
    target_data = value['target_data']
    time = value['time']
    if common_data_id < len(all_common_data):
        inputs.append(input_data)
        targets.append(target_data)
        times.append(time)

        input_pressure = np.zeros((32, 64))
        for i in range(8):
            input_pressure[:, i * 8: (i + 1) * 8] = input_data[12 + i]
            # print(input_pressure)

        inputs_pressure.append(input_pressure)


# print(inputs_pressure)
def dynamic_pic(inputs_pressure, target_arr):
    def update(frame):
        target_frame = np.reshape(target_arr[frame], (32, 64))
        im1.set_array(target_frame)
        input_frame = np.reshape(inputs_pressure[frame], (32, 64))
        im2.set_array(input_frame)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.set_aspect('equal', 'box')
    output_image = np.reshape(target_arr[0], (32, 64))
    im1 = ax1.imshow(output_image, cmap='jet', interpolation='bilinear')
    ax1.axis('off')
    cbar1 = fig.colorbar(im1, ax=ax1)

    ax2.set_aspect('equal', 'box')
    input_image = np.reshape(inputs_pressure[0], (32, 64))
    im2 = ax2.imshow(input_image, cmap='jet', interpolation='bilinear')
    ax2.axis('off')
    cbar2 = fig.colorbar(im2, ax=ax2)

    ani = animation.FuncAnimation(fig, update, frames=len(target_arr), interval=500, blit=False)

    plt.tight_layout()
    plt.show()


dynamic_pic(inputs_pressure, targets)
# print(times)
