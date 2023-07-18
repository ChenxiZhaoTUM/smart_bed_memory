import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

from GeneratorNet import weights_init, ConvTransposeNet, ConvNet
import data_preprocessing as dp
import utils
import matplotlib.pyplot as plt

######## Settings ########
# number of training iterations
iterations = 1000
# batch size
batch_size = 50
# learning rate, generator
lrG = 0.0006
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo = 4
# data set config
prop = None  # by default, use all from "./dataset/for_train"
# save txt files with per epoch loss?
saveL2 = True

##########################

prefix = ""
if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

dropout = 0
doLoad = ""  # optional, path to pre-trained model

print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Iterations: {}".format(iterations))

##########################

seed = random.randint(0, 2 ** 32 - 1)
print("Random seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# create pytorch data object with dataset
data = dp.PressureDataset()
trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
print("Training batches: {}".format(len(trainLoader)))
dataValidation = dp.ValiDataset(data)
valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=True)
print("Validation batches: {}".format(len(valiLoader)))

# setup training
epochs = int(iterations / len(trainLoader) + 0.5)
netG = ConvNet(channelExponent=expo, dropout=dropout)
print(netG)  # print full net
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized ConvNet with {} trainable params ".format(params))
print()

netG.apply(weights_init)
if len(doLoad) > 0:
    netG.load_state_dict(torch.load(doLoad))
    print("Loaded model " + doLoad)

criterionL2 = nn.MSELoss()
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)

inputs = Variable(torch.FloatTensor(batch_size, 20, 1, 1))
targets = Variable(torch.FloatTensor(batch_size, 1, 32, 64))

##########################
with open('output.txt', 'w') as file:
    pass

for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch + 1), epochs))

    netG.train()
    L2_accum = 0.0
    for i, traindata in enumerate(trainLoader, 0):
        inputs_cpu, targets_cpu = traindata
        inputs_cpu = inputs_cpu.unsqueeze(2).unsqueeze(3)
        targets_cpu = targets_cpu.unsqueeze(1)
        inputs.data.copy_(inputs_cpu.float())
        targets.data.copy_(targets_cpu.float())

        # compute LR decay
        if decayLr:
            currLr = utils.computeLR(epoch, epochs, lrG * 0.1, lrG)
            if currLr < lrG:
                for g in optimizerG.param_groups:
                    g['lr'] = currLr

        netG.zero_grad()
        gen_out = netG(inputs)

        lossL2 = criterionL2(gen_out, targets)
        lossL2.backward()

        optimizerG.step()

        lossL2viz = lossL2.item()
        L2_accum += lossL2viz

        if i == len(trainLoader) - 1:
            logline = "Epoch: {}, batch-idx: {}, L2: {}\n".format(epoch, i, lossL2viz)
            print(logline)

    # validation
    netG.eval()
    L2val_accum = 0.0
    for i, validata in enumerate(valiLoader, 0):
        inputs_cpu, targets_cpu = validata
        inputs_cpu = inputs_cpu.unsqueeze(2).unsqueeze(3)
        targets_cpu = targets_cpu.unsqueeze(1)
        inputs.data.copy_(inputs_cpu.float())
        targets.data.copy_(targets_cpu.float())

        outputs = netG(inputs)
        outputs_cpu = outputs.data.cpu().numpy()

        lossL2 = criterionL2(outputs, targets)
        L2val_accum += lossL2.item()

        inputs_denormalized = data.denormalizeInput(inputs_cpu.cpu())
        inputs_denormalized = inputs_denormalized.numpy()
        targets_denormalized = data.denormalize(targets_cpu.cpu().numpy())
        outputs_denormalized = data.denormalize(outputs_cpu)

        inputs_plot = torch.from_numpy(inputs_denormalized)
        targets_plot = torch.from_numpy(targets_denormalized)
        outputs_plot = torch.from_numpy(outputs_denormalized)

        # if epoch % 5 == 0:
        #     for j in range(batch_size):
        #         plt.figure(figsize=(10, 10))
        #
        #         plt.subplot(1, 3, 1)
        #         plt.title('Input Image')
        #         plt.imshow(inputs_plot[j, -8:, ], aspect='auto', cmap='viridis')
        #         plt.axis('off')
        #
        #         plt.subplot(1, 3, 2)
        #         plt.title('Target Image')
        #         plt.imshow(targets_plot[j].permute(1, 2, 0))
        #         plt.axis('off')
        #
        #         plt.subplot(1, 3, 3)
        #         plt.title('Output Image')
        #         plt.imshow(outputs_plot[j].permute(1, 2, 0))
        #         plt.axis('off')
        #
        #         plt.show()

        ######### test code ########
        if epoch == epochs - 1:
            inputs_list = inputs_denormalized.tolist()
            targets_list = targets_denormalized.tolist()
            outputs_list = outputs_denormalized.tolist()

            with open('output.txt', 'a') as file:
                file.write('Inputs:\n')
                for item in inputs_list:
                    file.write(str(item))
                    file.write('\n')

                file.write('Targets:\n')
                for item in targets_list:
                    file.write(str(item))
                    file.write('\n')

                file.write('Outputs:\n')
                for item in outputs_list:
                    file.write(str(item))
                    file.write('\n')

            for j in range(batch_size):
                utils.makeDirs(["results_train"])
                utils.imageOut("results_train/epoch{}_{}_{}".format(epoch, i, j), inputs_denormalized[j],
                               outputs_denormalized[j], targets_denormalized[j], data.target_max, data.target_min,
                               saveTargets=False)

    # data for graph plotting
    L2_accum /= len(trainLoader)
    L2val_accum /= len(valiLoader)
    if saveL2:
        if epoch == 0:
            utils.resetLog(prefix + "L2.txt")
            utils.resetLog(prefix + "L2val.txt")
        utils.log(prefix + "L2.txt", "{} ".format(L2_accum), False)
        utils.log(prefix + "L2val.txt", "{} ".format(L2val_accum), False)

torch.save(netG.state_dict(), prefix + "modelG")
