import os, sys, random, math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from GeneratorNet import OnlyPressureConvNet
import data_preprocessing_for_Nils as dp
import utils

##########################

prefix = "expo3_Nils_for_pressure_1"
if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

suffix = ""
lf = "./" + prefix + "_testout{}.txt".format(suffix)
utils.makeDirs(["test_results_for_model01"])

expo = 3
dataset = dp.PressureDataset(mode=dp.PressureDataset.TEST, dataDirTest="/home/yyc/chenxi/smart_bed_memory/dataset/for_test")
testLoader = DataLoader(dataset, batch_size=1, shuffle=False)
print("Test batches: {}".format(len(testLoader)))

inputs = Variable(torch.FloatTensor(1, 13, 32, 64))
targets = Variable(torch.FloatTensor(1, 1, 32, 64))
inputs = inputs.cuda()
targets = targets.cuda()

targets_dn = Variable(torch.FloatTensor(1, 1, 32, 64))
outputs_dn = Variable(torch.FloatTensor(1, 1, 32, 64))
targets_dn = targets_dn.cuda()
outputs_dn = outputs_dn.cuda()

netG = OnlyPressureConvNet(channelExponent=expo)
print(netG)

doLoad = "/home/yyc/chenxi/expo3_Nils_for_pressure_1_modelG"
if len(doLoad) > 0:
    netG.load_state_dict(torch.load(doLoad))
    print("Loaded model " + doLoad)
netG.cuda()

criterionL1 = nn.L1Loss()
criterionL1.cuda()

L1val_accum = 0.0
L1val_dn_accum = 0.0
lossPer_p_accum = 0

netG.eval()

for i, testdata in enumerate(testLoader, 0):
    inputs_cpu, targets_cpu = testdata
    targets_cpu = targets_cpu.unsqueeze(1)
    inputs_cpu = inputs_cpu.float().cuda()
    targets_cpu = targets_cpu.float().cuda()
    inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
    targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

    outputs = netG(inputs)
    outputs_cpu = outputs.data.cpu().numpy()
    targets_cpu = targets.data.cpu().numpy()

    lossL1 = criterionL1(outputs, targets)
    L1val_accum += lossL1.item()

    lossPer_p = np.sum(np.abs(outputs_cpu - targets_cpu)) / np.sum(np.abs(targets_cpu))
    lossPer_p_accum += lossPer_p.item()

    utils.log(lf, "Test sample %d" % i)
    utils.log(lf, "    pressure:  abs. difference, ratio: %f , %f " % (
        np.sum(np.abs(outputs_cpu - targets_cpu)), lossPer_p.item()))

    inputs_denormalized = dataset.denormalizeInput(inputs_cpu.cpu())
    inputs_denormalized = inputs_denormalized.numpy()
    targets_denormalized = dataset.denormalize(targets_cpu)
    outputs_denormalized = dataset.denormalize(outputs_cpu)

    targets_denormalized_comp = torch.from_numpy(targets_denormalized)
    outputs_denormalized_comp = torch.from_numpy(outputs_denormalized)

    targets_denormalized_comp, outputs_denormalized_comp = targets_denormalized_comp.float().cuda(), outputs_denormalized_comp.float().cuda()

    outputs_dn.data.resize_as_(outputs_denormalized_comp).copy_(outputs_denormalized_comp)
    targets_dn.data.resize_as_(targets_denormalized_comp).copy_(targets_denormalized_comp)

    lossL1_dn = criterionL1(outputs_dn, targets_dn)
    L1val_dn_accum += lossL1_dn.item()

    os.chdir("./test_results_for_model01/")
    utils.imageOut("%04d" % (i), inputs_denormalized[0], outputs_denormalized, targets_denormalized, dataset.target_max, dataset.target_min,
                               saveTargets=True)
    os.chdir("../")

utils.log(lf, "\n")
L1val_accum /= len(testLoader)
lossPer_p_accum /= len(testLoader)
L1val_dn_accum /= len(testLoader)
utils.log(lf, "Loss percentage of p: %f %% " % (lossPer_p_accum * 100))
utils.log(lf, "L1 error: %f" % (L1val_accum))
utils.log(lf, "Denormalized error: %f" % (L1val_dn_accum))
utils.log(lf, "\n")
