import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Convolution') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# 2 layers convolutional neural network
class ConvTransposeNet(nn.Module):
    def __init__(self):
        super(ConvTransposeNet, self).__init__()

        self.conv1 = nn.ConvTranspose2d(20, 64, kernel_size=4, stride=2, padding=0, bias=True)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=(4, 6), stride=(2, 4), padding=1, bias=True)
        self.conv4 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1, bias=True)

    def forward(self, x):
        out1 = self.conv1(x)  # torch.Size([1, 64, 4, 4])
        out2 = self.conv2(out1)  # torch.Size([1, 32, 8, 8])
        out3 = self.conv3(out2)  # torch.Size([1, 16, 16, 32])
        out4 = self.conv4(out3)  # torch.Size([1, 1, 32, 64])
        return out4


class DeepConvTransposeNet(nn.Module):
    def __init__(self):
        super(DeepConvTransposeNet, self).__init__()

        self.conv1 = nn.ConvTranspose2d(20, 32, kernel_size=2, stride=2, padding=0, bias=True)
        self.conv2 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2, padding=0, bias=True)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv4 = nn.ConvTranspose2d(32, 16, kernel_size=(5, 4), stride=(1, 2), padding=(2, 1), bias=True)
        self.conv5 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv6 = nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1, bias=True)

    def forward(self, x):
        out1 = self.conv1(x)  # torch.Size([1, 32, 2, 2])
        out2 = self.conv2(out1)  # torch.Size([1, 64, 4, 4])
        out3 = self.conv3(out2)  # torch.Size([1, 32, 8, 8])
        out4 = self.conv4(out3)  # torch.Size([1, 16, 8, 16])
        out5 = self.conv5(out4)  # torch.Size([1, 8, 16, 32])
        out6 = self.conv6(out5)  # torch.Size([1, 1, 32, 64])
        return out6


class DeepConvTransposeNet2(nn.Module):
    def __init__(self):
        super(DeepConvTransposeNet2, self).__init__()

        self.conv1 = nn.ConvTranspose2d(20, 32, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv2 = nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=2, bias=True)
        self.conv4 = nn.ConvTranspose2d(32, 16, kernel_size=(4, 6), stride=2, padding=(5, 2), bias=True)
        self.conv5 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv6 = nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1, bias=True)

    def forward(self, x):
        out1 = self.conv1(x)  # torch.Size([1, 32, 2, 2])
        out2 = self.conv2(out1)  # torch.Size([1, 64, 4, 4])
        out3 = self.conv3(out2)  # torch.Size([1, 32, 8, 8])
        out4 = self.conv4(out3)  # torch.Size([1, 16, 8, 16])
        out5 = self.conv5(out4)  # torch.Size([1, 8, 16, 32])
        out6 = self.conv6(out5)  # torch.Size([1, 1, 32, 64])
        return out6


# def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, factor=2, size=(3, 3), stride=(1, 1), pad=(1, 1),
#               dropout=0.):
#     block = nn.Sequential()
#
#     if relu:
#         block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
#     else:
#         block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
#
#     if not transposed:
#         block.add_module('%s_conv' % name,
#                          nn.Conv2d(in_c, out_c, kernel_size=size, stride=stride, padding=pad, bias=True))
#     else:
#         block.add_module('%s_upsam' % name,
#                          nn.Upsample(scale_factor=factor, mode='bilinear'))
#         block.add_module('%s_tconv' % name,
#                          nn.Conv2d(in_c, out_c, kernel_size=size, stride=stride, padding=pad, bias=True))
#
#     if bn:
#         block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
#
#     if dropout > 0.:
#         block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))
#
#     return block
#
#
# class ConvNet(nn.Module):
#     def __init__(self, channelExponent=4, dropout=0.):
#         super(ConvNet, self).__init__()
#         channels = int(2 ** channelExponent + 0.5)
#
#         self.dlayer1 = blockUNet(20, channels * 4, 'dlayer1', transposed=True, bn=True, relu=True,
#                                  dropout=dropout, factor=6, pad=0)
#
#         self.dlayer2 = blockUNet(channels * 4, channels * 2, 'dlayer2', transposed=True, bn=True, relu=True,
#                                  dropout=dropout)
#         self.dlayer3 = blockUNet(channels * 2, channels, 'dlayer3', transposed=True, bn=True, relu=True,
#                                  dropout=dropout, factor=4, stride=(2, 1))
#         self.dlayer4 = blockUNet(channels, 1, 'dlayer4', transposed=True, bn=False, relu=True,
#                                  dropout=dropout)
#
#     def forward(self, x):
#         out1 = self.dlayer1(x)  # torch.Size([1, 64, 4, 4])
#         out2 = self.dlayer2(out1)  # torch.Size([1, 32, 8, 8])
#         out3 = self.dlayer3(out2)  # torch.Size([1, 16, 16, 32])
#         out4 = self.dlayer4(out3)  # torch.Size([1, 1, 32, 64])
#         return out4


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, factor=2, size=(4, 4), stride=(1, 1), pad=(1, 1),
              dropout=0.):
    block = nn.Sequential()

    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    if not transposed:
        block.add_module('%s_conv' % name,
                         nn.Conv2d(in_c, out_c, kernel_size=size, stride=stride, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name,
                         nn.Upsample(scale_factor=factor, mode='bilinear'))
        block.add_module('%s_tconv' % name,
                         nn.Conv2d(in_c, out_c, kernel_size=size, stride=stride, padding=pad, bias=True))

    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))

    if dropout > 0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))

    return block


class OnlyPressureConvNet(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(OnlyPressureConvNet, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(13, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels, channels * 2, 'layer2', transposed=False, bn=True, relu=False,
                                dropout=dropout, stride=2)

        self.layer2b = blockUNet(channels * 2, channels * 2, 'layer2b', transposed=False, bn=True, relu=False,
                                 dropout=dropout, stride=2)

        self.layer3 = blockUNet(channels * 2, channels * 4, 'layer3', transposed=False, bn=True, relu=False,
                                dropout=dropout, size=(5, 5), pad=(1, 0))

        self.layer4 = blockUNet(channels * 4, channels * 4, 'layer4', transposed=False, bn=True, relu=False,
                                dropout=dropout, size=(2, 3), pad=0)

        self.dlayer4 = blockUNet(channels * 4, channels * 4, 'dlayer4', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=(3, 3))

        self.dlayer3 = blockUNet(channels * 8, channels * 2, 'dlayer3', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=(3, 3))

        self.dlayer2b = blockUNet(channels * 4, channels * 2, 'dlayer2b', transposed=True, bn=True, relu=True,
                                  dropout=dropout, size=(3, 3))
        self.dlayer2 = blockUNet(channels * 4, channels, 'dlayer2', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=(3, 3))

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels * 2, 1, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)  # torch.Size([1, 64, 16, 32])
        out2 = self.layer2(out1)  # torch.Size([1, 128, 8, 16])
        out2b = self.layer2b(out2)  # torch.Size([1, 128, 4, 8])
        out3 = self.layer3(out2b)  # torch.Size([1, 256, 2, 4])
        out4 = self.layer4(out3)  # torch.Size([1, 256, 1, 2])

        dout4 = self.dlayer4(out4)  # torch.Size([1, 256, 2, 4])
        dout4_out3 = torch.cat([dout4, out3], 1)  # torch.Size([1, 512, 2, 4])

        dout3 = self.dlayer3(dout4_out3)  # torch.Size([1, 128, 4, 8])
        dout3_out2b = torch.cat([dout3, out2b], 1)  # torch.Size([1, 256, 4, 8])

        dout2b = self.dlayer2b(dout3_out2b)  # torch.Size([1, 128, 8, 16])
        dout2b_out2 = torch.cat([dout2b, out2], 1)  # torch.Size([1, 256, 8, 16])

        dout2 = self.dlayer2(dout2b_out2)  # torch.Size([1, 64, 16, 32])
        dout2_out1 = torch.cat([dout2, out1], 1)  # torch.Size([1, 128, 16, 32])

        dout1 = self.dlayer1(dout2_out1)  # torch.Size([1, 1, 32, 64])
        return dout1


class ConvNet(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(ConvNet, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(13, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels, channels * 2, 'layer2', transposed=False, bn=True, relu=False,
                                dropout=dropout, stride=2)

        self.layer2b = blockUNet(channels * 2, channels * 2, 'layer2b', transposed=False, bn=True, relu=False,
                                 dropout=dropout, stride=2)

        self.layer3 = blockUNet(channels * 2, channels * 4, 'layer3', transposed=False, bn=True, relu=False,
                                dropout=dropout, size=(5, 5), pad=(1, 0))

        self.layer4 = blockUNet(channels * 4, channels * 4, 'layer4', transposed=False, bn=True, relu=False,
                                dropout=dropout, size=(2, 3), pad=0)

        self.dlayer4 = blockUNet(channels * 4, channels * 4, 'dlayer4', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=(3, 3))

        self.dlayer3 = blockUNet(channels * 8, channels * 2, 'dlayer3', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=(3, 3))

        self.dlayer2b = blockUNet(channels * 4, channels * 2, 'dlayer2b', transposed=True, bn=True, relu=True,
                                  dropout=dropout, size=(3, 3))
        self.dlayer2 = blockUNet(channels * 4, channels, 'dlayer2', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=(3, 3))

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels * 2, 1, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)  # torch.Size([1, 64, 16, 32])
        out2 = self.layer2(out1)  # torch.Size([1, 128, 8, 16])
        out2b = self.layer2b(out2)  # torch.Size([1, 128, 4, 8])
        out3 = self.layer3(out2b)  # torch.Size([1, 256, 2, 4])
        out4 = self.layer4(out3)  # torch.Size([1, 256, 1, 2])

        dout4 = self.dlayer4(out4)  # torch.Size([1, 256, 2, 4])
        dout4_out3 = torch.cat([dout4, out3], 1)  # torch.Size([1, 512, 2, 4])

        dout3 = self.dlayer3(dout4_out3)  # torch.Size([1, 128, 4, 8])
        dout3_out2b = torch.cat([dout3, out2b], 1)  # torch.Size([1, 256, 4, 8])

        dout2b = self.dlayer2b(dout3_out2b)  # torch.Size([1, 128, 8, 16])
        dout2b_out2 = torch.cat([dout2b, out2], 1)  # torch.Size([1, 256, 8, 16])

        dout2 = self.dlayer2(dout2b_out2)  # torch.Size([1, 64, 16, 32])
        dout2_out1 = torch.cat([dout2, out1], 1)  # torch.Size([1, 128, 16, 32])

        dout1 = self.dlayer1(dout2_out1)  # torch.Size([1, 1, 32, 64])
        return dout1


# test01
# model = DeepConvTransposeNet2()
# input_tensor = torch.randn(1, 20, 1, 1)
# output_tensor = model(input_tensor)
# print(output_tensor.size())

# test02
# model = ConvNet()
# input_tensor = torch.randn(1, 13, 32, 64)
# output_tensor = model(input_tensor)
# print(output_tensor.size())
