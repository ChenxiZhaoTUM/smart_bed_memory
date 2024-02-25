import numpy as np
from datetime import datetime, timedelta


def parse_time_string(time_string):
    return datetime.strptime(time_string, "%H:%M:%S.%f")


def format_time_string(time):
    return time.strftime("%H:%M:%S.%f")


def round_to_nearest_half_second_down(time_obj):

    if time_obj.microsecond >= 500000:
        time_obj = time_obj.replace(microsecond=500000)
    else:
        time_obj = time_obj.replace(microsecond=0)

    return time_obj


def average_by_half_second(time_arr, value_arr):
    time_objs = [parse_time_string(time_str) for time_str in time_arr]

    # 将时间对象舍入到最近的0.5秒间隔（向下舍入）
    time_objs = [round_to_nearest_half_second_down(time_obj) for time_obj in time_objs]
    time_objs = [format_time_string(time_obj) for time_obj in time_objs]

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

    return new_time_arr, avg_value_arr


# 示例用法：
time_arr = ["10:00:00.10", "10:00:00.30", "10:00:01.20", "10:00:02.70", "10:00:03.50"]
value_arr = [1, 2, 3, 4, 5]

new_time_arr, avg_value_arr = average_by_half_second(time_arr, value_arr)
for time_str, avg_value in zip(new_time_arr, avg_value_arr):
    print(f"{time_str}: {avg_value}")