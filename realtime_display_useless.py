import data_preprocessing_for_Nils as dp
from data_preprocessing_for_Nils import PressureDataset
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

dataset = PressureDataset(mode=PressureDataset.TRAIN, dataDir="./dataset/for_train")

inputs = Variable(torch.FloatTensor(len(dataset), 13, 32, 64))
targets = Variable(torch.FloatTensor(len(dataset), 1, 32, 64))

inputs_denormalized = []
targets_denormalized = []

# print(inputs.size())  # torch.Size([642, 13, 32, 64])

for i, traindata in enumerate(dataset, 0):
    inputs_cpu, targets_cpu = traindata
    # print(type(targets_cpu))
    inputs_cpu = targets_cpu.unsqueeze(0)
    targets_cpu = targets_cpu.unsqueeze(0).unsqueeze(1)
    inputs.data.copy_(inputs_cpu.float())
    print(targets_cpu.size())
    targets.data.copy_(targets_cpu.float())

    # inputs_denormalized_element = dataset.denormalizeInput(inputs_cpu.cpu())
    # inputs_denormalized_element = inputs_denormalized_element.numpy()

    output_denormalized_element = dataset.denormalize(targets_cpu.cpu().numpy())

    print(output_denormalized_element)





def dynamic_pic(value_arr):
    def update(frame):
        image = np.reshape(value_arr[frame], (32, 64))
        im.set_array(image)

    fig, ax = plt.subplots()

    ax.set_aspect('equal', 'box')
    image = np.reshape(value_arr[0], (32, 64))
    image = np.flip(image, axis=1)
    image = np.flip(image, axis=0)
    # im = ax.imshow(image, cmap='jet', interpolation='bilinear', norm=colors.Normalize(vmin=0, vmax=1))
    im = ax.imshow(image, cmap='jet', interpolation='bilinear')
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.axis('off')

    ani = animation.FuncAnimation(fig, update, frames=len(value_arr), interval=500, blit=False)
    cbar = plt.colorbar(im)

    plt.show()

