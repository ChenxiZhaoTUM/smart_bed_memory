import os
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

folder_path = "dataset"
file_names = os.listdir(folder_path)


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
            # print(type(value_arr[i]))  # test code
            count_dict[new_time_str] = 1
            new_time_arr.append(new_time_str)

    avg_value_arr = [sum_dict[new_time_str] / count_dict[new_time_str] for new_time_str in new_time_arr]
    return new_time_arr, avg_value_arr


for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    if file_path.endswith(".TXT"):
        with open(file_path, 'r') as file:
            lines = file.readlines()

            time_arr = []
            value_arr = []

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
                time_arr.append(time_str)

                value_per_time = [int(value, 16) for value in values]
                value_arr.append(value_per_time)

            value_arr = np.array(value_arr)

            ######## do time average #######
            # new_time_arr, avg_value_arr = average_by_sec(time_arr, value_arr)
            # print(new_time_arr)

            ######### plot 2D ########
            # plt.plot(time_arr, value_arr)
            # plt.xlabel('Time')
            # plt.ylabel('Value')
            # plt.title('Value over Time')
            # plt.show()

            # Convert time strings to datetime objects
            timestamps = [datetime.strptime(ts, '%H:%M:%S.%f') for ts in time_arr]
            # Convert datetime objects to numeric values
            numeric_timestamps = mdates.date2num(timestamps)

            # plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            group_index = list(range(1, 21))
            for i, index in enumerate(group_index):
                ax.plot3D(numeric_timestamps, [index] * len(numeric_timestamps), [value[i] for value in value_arr])

            ax.set_xlabel('Time')
            ax.set_ylabel('Index')
            ax.set_zlabel('Value')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

            plt.show()
