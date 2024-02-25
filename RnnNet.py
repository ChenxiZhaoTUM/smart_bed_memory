import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class RNNModel(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers, output_dim, time_steps):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_steps = time_steps

        self.rnn = nn.RNN2d(input_channels, hidden_size, num_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        batch_size, channels, height, width = x.size(0), x.size(2), x.size(3), x.size(4)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_().to(x.device)
        out, _ = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :, :, :])
        return out

input_channels = 13
input_size = (32, 64)
num_classes = 1
hidden_size = 128
num_layers = 2
time_steps = 10

model = RNNModel(input_channels, hidden_size, num_layers, num_classes, time_steps)
print(model)

input_tensor = torch.randn(632, 10, 13, 32, 64)
output_tensor = model(input_tensor)
print(output_tensor.size())


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


class RNNUNet(nn.Module):
    def __init__(self, channelExponent=6, dropout=0., time_step=10):
        super(RNNUNet, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.time_step = time_step

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

        self.layer_dim = 1
        self.hidden_dim = channels * 4
        self.rnn = nn.RNN(input_size=channels * 4, hidden_size=self.hidden_dim, num_layers=self.layer_dim,
                          batch_first=True,
                          nonlinearity='relu')
        self.fc = nn.Linear(self.hidden_dim, channels * 4)

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

        h0 = torch.zeros(self.layer_dim, out4.size(0), self.hidden_dim).requires_grad_().to(device)
        out4_rnn, hn = self.rnn(out4, h0.detach())
        out4_fc = self.fc(out4_rnn[:, -1, :])

        dout4 = self.dlayer4(out4_fc)  # torch.Size([1, 256, 2, 4])
        dout4_out3 = torch.cat([dout4, out3], 1)  # torch.Size([1, 512, 2, 4])

        dout3 = self.dlayer3(dout4_out3)  # torch.Size([1, 128, 4, 8])
        dout3_out2b = torch.cat([dout3, out2b], 1)  # torch.Size([1, 256, 4, 8])

        dout2b = self.dlayer2b(dout3_out2b)  # torch.Size([1, 128, 8, 16])
        dout2b_out2 = torch.cat([dout2b, out2], 1)  # torch.Size([1, 256, 8, 16])

        dout2 = self.dlayer2(dout2b_out2)  # torch.Size([1, 64, 16, 32])
        dout2_out1 = torch.cat([dout2, out1], 1)  # torch.Size([1, 128, 16, 32])

        dout1 = self.dlayer1(dout2_out1)  # torch.Size([1, 1, 32, 64])
        return dout1

# model = RNNUNet(time_step=10)
# input_tensor = torch.randn(60, 13, 32, 64)
# output_tensor = model(input_tensor)
# print(output_tensor.size())
