import torch
import torch.nn as nn
import torch.nn.functional as F


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


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, factor=2, size=3, stride=(1, 1), pad=1, dropout=0.):
    block = nn.Sequential()

    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=stride, padding=pad, bias=True))
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


class ConvNet(nn.Module):
    def __init__(self, channelExponent=4, dropout=0.):
        super(ConvNet, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.dlayer1 = blockUNet(20, channels * 4, 'dlayer1', transposed=True, bn=True, relu=True,
                                 dropout=dropout, factor=6, pad=0)

        self.dlayer2 = blockUNet(channels * 4, channels * 2, 'dlayer2', transposed=True, bn=True, relu=True,
                                 dropout=dropout)
        self.dlayer3 = blockUNet(channels * 2, channels, 'dlayer3', transposed=True, bn=True, relu=True,
                                 dropout=dropout, factor=4, stride=(2, 1))
        self.dlayer4 = blockUNet(channels, 1, 'dlayer4', transposed=True, bn=False, relu=True,
                                 dropout=dropout)

    def forward(self, x):
        out1 = self.dlayer1(x)  # torch.Size([1, 64, 4, 4])
        out2 = self.dlayer2(out1)  # torch.Size([1, 32, 8, 8])
        out3 = self.dlayer3(out2)  # torch.Size([1, 16, 16, 32])
        out4 = self.dlayer4(out3)  # torch.Size([1, 1, 32, 64])
        return out4


# test
# model = ConvNet()
# input_tensor = torch.randn(1, 20, 1, 1)
# output_tensor = model(input_tensor)
# print(output_tensor.size())
