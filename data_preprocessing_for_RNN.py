import os
import random
import re
from datetime import datetime
import numpy as np
from torch.utils.data import Dataset
import torch

np.set_printoptions(threshold=np.inf)
pressureNormalization = True
removePOffset = True


######## average values by seconds ########
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


def save_data_from_files(data, isTest=False, shuffle=2):
    if isTest:
        folder_path = data.dataDirTest
        file_names = os.listdir(data.dataDirTest)
    else:
        folder_path = data.dataDir
        file_names = os.listdir(data.dataDir)

    for i in range(shuffle):
        random.shuffle(file_names)

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
            avg_time_arr_txt, avg_value_arr_txt = average_by_sec(time_arr_txt, value_arr_txt)
            avg_time_arr_csv, avg_value_arr_csv = average_by_sec(time_arr_csv, value_arr_csv)

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

    data.common_data = all_common_data

    print()
    print("Number of data loaded:", len(data.common_data))

    return data


######## data normalization ########
def loader_normalizer(data):
    if removePOffset:
        input_data_values = [value['input_data'] for value in data.common_data.values() if
                             len(value['input_data']) > 0]

        offset = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 18, 18, 18, 18, 18, 18, 18]
        # offset = [18, 18, 18, 18, 18, 18, 18, 18]

        if len(input_data_values) > 0:
            for value in data.common_data.values():
                input_data = value['input_data']
                removePOffset_input_data = input_data - offset
                value['input_data'] = removePOffset_input_data

    if pressureNormalization:
        target_data_values = [value['target_data'] for value in data.common_data.values() if
                              len(value['target_data']) > 0]

        if len(target_data_values) > 0:
            data.target_min = np.amin(target_data_values)
            data.target_max = np.amax(target_data_values)

            for value in data.common_data.values():
                target_data = value['target_data']
                normalized_target_data = (target_data - data.target_min) / (data.target_max - data.target_min)
                value['target_data'] = normalized_target_data

            print("Max Pressure:" + str(data.target_max))
            print("Min Pressure:" + str(data.target_min))

    values = list(data.common_data.values())

    for id, value in enumerate(values):
        data.common_data[id] = value

    return data


class PressureDataset(Dataset):
    TRAIN = 0
    TEST = 2

    def __init__(self, mode=TRAIN, dataDir="./dataset/for_train", dataDirTest="./dataset/for_test/",
                 shuffle=0):
        global removePOffset, pressureNormalization
        """
        :param dataProp: for split&mix from multiple dirs, see LoaderNormalizer; None means off
        :param mode: TRAIN|TEST , toggle regular 80/20 split for training & validation data, or load test data
        :param dataDir: directory containing training data
        :param dataDirTest: second directory containing test data , needs training dir for normalization
        :param normMode: toggle normalization
        """
        if not (mode == self.TRAIN or mode == self.TEST):
            print("Error - TurbDataset invalid mode " + format(mode))
            exit(1)

        self.mode = mode
        self.dataDir = dataDir
        self.dataDirTest = dataDirTest  # only for mode==self.TEST

        self = save_data_from_files(self, isTest=(mode == self.TEST))

        # load & normalize data
        self = loader_normalizer(self)

        self.totalLength = len(self.common_data)
        if not self.mode == self.TEST:
            # split for train/validation sets (80/20) , max 400
            targetLength = self.totalLength

            self.inputs = []
            self.targets = []
            self.valiInputs = []
            self.valiTargets = []

            for common_data_id in range(self.totalLength):
                value = self.common_data[common_data_id]
                input_data = value['input_data']
                target_data = value['target_data']
                if common_data_id < targetLength:
                    self.inputs.append(input_data)
                    self.targets.append(target_data)
                else:
                    self.valiInputs.append(input_data)
                    self.valiTargets.append(target_data)

            self.valiLength = self.totalLength - targetLength
            self.totalLength = targetLength

        else:
            self.inputs = []
            self.targets = []
            for common_data_id in range(self.totalLength):
                value = self.common_data[common_data_id]
                input_data = value['input_data']
                target_data = value['target_data']
                self.inputs.append(input_data)
                self.targets.append(target_data)

    def __len__(self):
        return self.totalLength

    # def __getitem__(self, idx):
    #     return self.inputs[idx], self.targets[idx]

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        target_data = self.targets[idx]

        new_input_data = torch.zeros(13, 32, 64)

        for i in range(12):
            new_input_data[i, :, :] = input_data[i]

        torch.set_printoptions(profile="full")

        for i in range(8):
            # print(input_data[12 + i])
            new_input_data[12, :, i * 8: (i + 1) * 8] = input_data[12 + i]
            # new_input_data[12, :, i * 8: (i + 1) * 8] = input_data[19 - i]
            # print(new_input_data[12, :, :])

        # print("input_data shape:", input_data.shape)
        # print("new_input_data shape:", new_input_data.shape)
        # print("new_input_data content:", new_input_data)

        # print(new_input_data)

        # print(torch.from_numpy(target_data).size())  # torch.Size([32, 64])

        return new_input_data, torch.from_numpy(target_data)

    def denormalize(self, np_array):
        denormalized_data = np_array * (self.target_max - self.target_min) + self.target_min

        return denormalized_data

    def denormalizeInput(self, inputs_tensor):
        last_channel_index = inputs_tensor.size(1) - 1
        inputs_tensor[:, last_channel_index:, :, :] += 18
        return inputs_tensor

    def extract_data(self, time_step=10):
        X = []
        y = []

        for idx in range(self.totalLength - time_step):
            X.append(self.inputs[idx: idx + time_step])
            y.append(self.targets[idx: idx + time_step])
        X = np.array(X)
        y = np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 20)
        print(X.shape)
        # print(X)
        # print(y[0])
        return X, y


# simplified validation data set (main one is TurbDataset above)
class ValiDataset(PressureDataset):
    def __init__(self, dataset):
        self.inputs = dataset.valiInputs
        self.targets = dataset.valiTargets
        self.totalLength = dataset.valiLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


time_step = 10
data = PressureDataset()
data.extract_data(time_step=time_step)
