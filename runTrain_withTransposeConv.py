import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

from GeneratorNet import weights_init, DeepConvTransposeNet
import data_preprocessing as dp
import utils

######## Settings ########
# number of training iterations
iterations = 100000000000
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
saveL1 = True

##########################

prefix = ""
if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

# dropout = 0
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
netG = DeepConvTransposeNet()
print(netG)  # print full net
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized ConvNet with {} trainable params ".format(params))
print()

netG.apply(weights_init)
if len(doLoad) > 0:
    netG.load_state_dict(torch.load(doLoad))
    print("Loaded model " + doLoad)

criterionL1 = nn.L1Loss()
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)

inputs = Variable(torch.FloatTensor(batch_size, 20, 1, 1))
targets = Variable(torch.FloatTensor(batch_size, 1, 32, 64))

##########################
# with open('output.txt', 'w') as file:
#     pass

for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch + 1), epochs))

    netG.train()
    L1_accum = 0.0
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
        gen_out_cpu = gen_out.data.cpu().numpy()

        lossL1 = criterionL1(gen_out, targets)
        lossL1.backward()

        optimizerG.step()

        lossL1viz = lossL1.item()
        L1_accum += lossL1viz

        if i == len(trainLoader) - 1:
            logline = "Epoch: {}, batch-idx: {}, L1: {}\n".format(epoch, i, lossL1viz)
            print(logline)

        inputs_denormalized = data.denormalizeInput(inputs_cpu.cpu())
        inputs_denormalized = inputs_denormalized.numpy()
        targets_denormalized = data.denormalize(targets_cpu.cpu().numpy())
        outputs_denormalized = data.denormalize(gen_out_cpu)

        if lossL1viz < 0.02:
            for j in range(batch_size):
                utils.makeDirs(["train_results"])
                utils.imageOut("train_results/epoch{}_{}_{}".format(epoch, i, j), inputs_denormalized[j],
                               outputs_denormalized[j], targets_denormalized[j], data.target_max, data.target_min,
                               saveTargets=True)

    # validation
    netG.eval()
    L1val_accum = 0.0
    for i, validata in enumerate(valiLoader, 0):
        inputs_cpu, targets_cpu = validata
        inputs_cpu = inputs_cpu.unsqueeze(2).unsqueeze(3)
        targets_cpu = targets_cpu.unsqueeze(1)
        inputs.data.copy_(inputs_cpu.float())
        targets.data.copy_(targets_cpu.float())

        outputs = netG(inputs)
        outputs_cpu = outputs.data.cpu().numpy()

        lossL1 = criterionL1(outputs, targets)
        L1val_accum += lossL1.item()

        # print(i)  # 0 1
        # print(inputs_cpu)
        inputs_denormalized = data.denormalizeInput(inputs_cpu.cpu())
        inputs_denormalized = inputs_denormalized.numpy()
        targets_denormalized = data.denormalize(targets_cpu.cpu().numpy())
        outputs_denormalized = data.denormalize(outputs_cpu)

        ######### test code ########
        # inputs_list = inputs_denormalized.tolist()
        # targets_list = targets_denormalized.tolist()
        # outputs_list = outputs_denormalized.tolist()
        #
        # with open('output.txt', 'a') as file:
        #     file.write('Inputs:\n')
        #     for item in inputs_list:
        #         file.write(str(item))
        #         file.write('\n')
        #
        #     file.write('Targets:\n')
        #     for item in targets_list:
        #         file.write(str(item))
        #         file.write('\n')
        #
        #     file.write('Outputs:\n')
        #     for item in outputs_list:
        #         file.write(str(item))
        #         file.write('\n')
        #
        for j in range(batch_size):
            utils.makeDirs(["train_results"])
            utils.imageOut("train_results/epoch{}_{}_{}".format(epoch, i, j), inputs_denormalized[j],
                           outputs_denormalized[j], targets_denormalized[j], data.target_max, data.target_min,
                           saveTargets=True)

    # data for graph plotting
    L1_accum /= len(trainLoader)
    # L1val_accum /= len(valiLoader)
    if saveL1:
        if epoch == 0:
            utils.resetLog(prefix + "L1.txt")
            utils.resetLog(prefix + "L1val.txt")
        utils.log(prefix + "L1.txt", "{} ".format(L1_accum), False)
        # utils.log(prefix + "L1val.txt", "{} ".format(L1val_accum), False)

torch.save(netG.state_dict(), prefix + "modelG")
