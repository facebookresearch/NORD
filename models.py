import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from torch import nn as nn
from torch.nn.utils import weight_norm


class Inception_dimred(nn.Module):
    def __init__(
        self,
        in_channel=64,
        conv1x1=16,
        reduce3x3=24,
        conv3x3=32,
        reduce5x5=16,
        conv5x5=8,
        pool_proj=8,
        pool=2,
    ):
        super(Inception_dimred, self).__init__()

        self.modules1 = nn.ModuleList()
        self.modules1.append(nn.Conv2d(in_channel, conv1x1, 1, (1, 1), 0))
        self.modules1.append(nn.Conv2d(in_channel, reduce3x3, 1, 1, 0))
        self.modules1.append(nn.Conv2d(reduce3x3, conv3x3, 3, (1, 1), 1))
        self.modules1.append(nn.Conv2d(in_channel, reduce5x5, 1, 1, 0))
        self.modules1.append(nn.Conv2d(reduce5x5, conv5x5, 5, (1, 1), 2))
        self.modules1.append(nn.MaxPool2d((3, 3), stride=(1, 1), padding=(1, 1)))
        self.modules1.append(nn.Conv2d(in_channel, pool_proj, 1, 1, 0))
        self.modules1.append(nn.MaxPool2d((1, pool)))

    def forward(self, x):

        a = F.relu(self.modules1[0](x))
        b = F.relu(self.modules1[2]((F.relu(self.modules1[1](x)))))
        c = F.relu(self.modules1[4]((F.relu(self.modules1[3](x)))))
        d = F.relu(self.modules1[5](x))
        d = F.relu(self.modules1[6](d))
        x1 = torch.cat((a, b, c, d), axis=1)
        x2 = F.relu(self.modules1[7](x1))
        return x2


class base_encoder(nn.Module):
    def __init__(self, dev=torch.device("cpu"), ch=3):
        super(base_encoder, self).__init__()
        self.dev = dev

        self.modelA = Inception_dimred(in_channel=ch, pool=2)
        self.modelB = Inception_dimred(in_channel=64, pool=2)
        self.modelC = Inception_dimred(in_channel=64, pool=2)
        self.modelD = Inception_dimred(in_channel=64, pool=2)
        self.modelE = Inception_dimred(in_channel=64, pool=2)
        self.modelF = Inception_dimred(in_channel=64, pool=2)
        self.modelG = Inception_dimred(in_channel=64, pool=2)

    def forward(self, x):
        #         x = (self.modelD(self.modelC(self.modelB(self.modelA(x)))))
        x = self.modelG(
            self.modelF(
                self.modelE(self.modelD(self.modelC(self.modelB(self.modelA(x)))))
            )
        )

        return x


class which_clean(nn.Module):
    def __init__(self):
        super(which_clean, self).__init__()
        n_layers = 2

        self.encoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.dp = nn.ModuleList()
        filter_size = 5
        dp_num = 0.50
        self.encoder.append(nn.Conv1d(256, 64, filter_size, padding=filter_size // 2))
        self.ebatch.append(nn.BatchNorm1d(64))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(64, 16, filter_size, padding=filter_size // 2))
        self.ebatch.append(nn.BatchNorm1d(16))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(16, 4, filter_size, padding=filter_size // 2))
        self.ebatch.append(nn.BatchNorm1d(4))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(4, 2, filter_size, padding=filter_size // 2))
        self.ebatch.append(nn.BatchNorm1d(2))
        self.dp.append(nn.Dropout(p=dp_num))

    def forward(self, x):
        # torch.Size([1, 128, 64])
        # print(x.shape)
        for i in range(4):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            if i != 3:
                x = F.leaky_relu(x, 0.1)
            x = self.dp[i](x)
        return x


class how_snr(nn.Module):
    def __init__(self, dim_emb=32, output=50):
        super(how_snr, self).__init__()
        n_layers = 2

        self.encoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.dp = nn.ModuleList()
        filter_size = 5
        dp_num = 0.50
        self.encoder.append(nn.Conv1d(256, 128, filter_size, padding=filter_size // 2))
        self.ebatch.append(nn.BatchNorm1d(128))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(128, 64, filter_size, padding=filter_size // 2))
        self.ebatch.append(nn.BatchNorm1d(64))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(64, 32, filter_size, padding=filter_size // 2))
        self.ebatch.append(nn.BatchNorm1d(32))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(32, 20, filter_size, padding=filter_size // 2))
        self.ebatch.append(nn.BatchNorm1d(20))
        self.dp.append(nn.Dropout(p=dp_num))

    def forward(self, x):
        # torch.Size([1, 32, 64])
        # print(x.shape)
        for i in range(4):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            if i != 3:
                x = F.leaky_relu(x, 0.1)
            x = self.dp[i](x)
        return x


class waveform_model_tracking(nn.Module):
    def __init__(self, dev=torch.device("cpu"), minit=1, nmr=1, ch=3):
        super(waveform_model_tracking, self).__init__()
        self.base_encoder = base_encoder(ch=ch)  # 1 x 32 x 64
        if nmr == 1:
            self.base_encoder_2 = TemporalConvNet(
                num_inputs=128, num_channels=[32, 64, 64, 128, 128], kernel_size=3
            )
        else:
            self.base_encoder_2 = TemporalConvNet(
                num_inputs=128, num_channels=[32, 64, 64, 128, 256], kernel_size=3
            )
        self.which_clean = which_clean()  # 1 x 128
        self.how_snr = how_snr()
        if minit == 1:
            self.base_encoder.apply(weights_init)
            self.which_clean.apply(weights_init)
            self.how_snr.apply(weights_init)
        self.ch = ch
        self.nmr = nmr

    def forward(self, x1, x2):

        batch_size = x1.shape[0]

        x1 = x1.reshape(x1.shape[0] * x1.shape[1], -1)
        x1 = torch.stft(
            x1,
            n_fft=512,
            hop_length=401,
            win_length=512,
            window=torch.hann_window(512).to(x1.device),
            return_complex=False,
            onesided=True,
        )  #
        real = x1[:, :, :, 0]
        im = x1[:, :, :, 1]
        power1 = (
            torch.sqrt(torch.pow(real, 2) + torch.pow(im, 2))
            .unsqueeze(-1)
            .permute(0, 3, 2, 1)
        )
        ang1 = (
            torch.atan2(x1[:, :, :, 1], x1[:, :, :, 0])
            .unsqueeze(-1)
            .permute(0, 3, 2, 1)
        )
        data1 = power1  # torch.cat([power1], dim=1)

        data1 = data1.reshape(batch_size, self.ch, data1.shape[2], data1.shape[3])

        x2 = x2.reshape(x2.shape[0] * x2.shape[1], -1)
        x2 = torch.stft(
            x2,
            n_fft=512,
            hop_length=401,
            win_length=512,
            window=torch.hann_window(512).to(x2.device),
            return_complex=False,
            onesided=True,
        )  #
        real = x2[:, :, :, 0]
        im = x2[:, :, :, 1]
        power2 = (
            torch.sqrt(torch.pow(real, 2) + torch.pow(im, 2))
            .unsqueeze(-1)
            .permute(0, 3, 2, 1)
        )
        ang2 = (
            torch.atan2(x2[:, :, :, 1], x2[:, :, :, 0])
            .unsqueeze(-1)
            .permute(0, 3, 2, 1)
        )
        data2 = power2  # torch.cat([power2, ang2], dim=1)
        data2 = data2.reshape(batch_size, self.ch, data2.shape[2], data2.shape[3])

        x1 = self.base_encoder.forward(data1)
        x1 = self.base_encoder_2(x1)

        x2 = self.base_encoder.forward(data2)
        x2 = self.base_encoder_2(x2)

        concat = torch.cat((x1, x2), 1)
        which_closer = self.which_clean.forward(concat)
        how_much_closer = self.how_snr.forward(concat)

        return which_closer.mean(2), how_much_closer


def softXEnt(input, target):
    logprobs = torch.nn.functional.log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=dilation,
                dilation=dilation,
            )
        )
        #         self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=dilation,
                dilation=dilation,
            )
        )
        #         self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x1):

        x1 = x1.reshape(x1.shape[0], -1, x1.shape[2])
        x = self.network(x1)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if (
        classname.find("Conv") != -1
        or classname.find("BatchNorm") != -1
        or classname.find("Linear") != -1
    ):
        torch.nn.init.normal_(m.weight)
        # default Linear init is kaiming_uniform_ / default Conv1d init is a scaled uniform /  default BN init is constant gamma=1 and bias=0
        try:
            torch.nn.init.constant_(m.bias, 0.01)
        except:
            pass


class siamese(nn.Module):
    def __init__(self):
        super(siamese, self).__init__()
        n_layers = 2

        self.encoder = nn.ModuleList()
        self.ebatch = nn.ModuleList()
        self.dp = nn.ModuleList()
        filter_size = 5
        dp_num = 0.50
        self.encoder.append(nn.Conv1d(128, 64, filter_size, padding=filter_size // 2))
        self.ebatch.append(nn.BatchNorm1d(64))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(64, 32, filter_size, padding=filter_size // 2))
        self.ebatch.append(nn.BatchNorm1d(32))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(32, 16, filter_size, padding=filter_size // 2))
        self.ebatch.append(nn.BatchNorm1d(16))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(16, 8, filter_size, padding=filter_size // 2))
        self.ebatch.append(nn.BatchNorm1d(8))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(8, 4, filter_size, padding=filter_size // 2))
        self.ebatch.append(nn.BatchNorm1d(4))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(4, 2, filter_size, padding=filter_size // 2))
        self.ebatch.append(nn.BatchNorm1d(2))
        self.dp.append(nn.Dropout(p=dp_num))
        self.encoder.append(nn.Conv1d(2, 1, filter_size, padding=filter_size // 2))
        self.ebatch.append(nn.BatchNorm1d(1))
        self.dp.append(nn.Dropout(p=dp_num))

    def forward(self, x):
        # torch.Size([1, 128, 64])
        # print(x.shape)
        for i in range(7):
            x = self.encoder[i](x)
            x = self.ebatch[i](x)
            if i != 6:
                x = F.leaky_relu(x, 0.1)
                x = self.dp[i](x)
        return x
