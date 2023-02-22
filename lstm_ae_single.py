from typing import DefaultDict
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from models import ConvLSTMCell
from torch.autograd import Function

class EncoderDecoderConvLSTMSingleStream(nn.Module):

    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTMSingleStream, self).__init__()

        """ ARCHITECTURE

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=3,
                                               bias=True)

        # self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
        #                                        hidden_dim=nf,
        #                                        kernel_size=3,
        #                                        bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf*15,  # nf + 1
                                               hidden_dim=nf*15,
                                               kernel_size=3,
                                               bias=True)

        # self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf*15,
        #                                        hidden_dim=nf*15,
        #                                        kernel_size=3,
        #                                        bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf*15,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))

        self.Linear1_src = nn.Linear(17500, 2400)
        self.Linear2_src = nn.Linear(2400, 600)
        self.Linear3_src = nn.Linear(600, 7)
        self.Linear1_tar = nn.Linear(17500, 2400)
        self.Linear2_tar = nn.Linear(2400, 600)
        self.Linear3_tar = nn.Linear(600, 7)
        self.relu = nn.ReLU(inplace=True)
        self.nf = nf

    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t3, c_t3):

        outputs = []
        output_enc_h = []
        output_enc_c = []
        output_dec_h = []
        output_dec_c = []

        # encoder
        b = x.shape[0]
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here


            output_enc_h += [h_t]

        encoder_vector = torch.stack(output_enc_h)
        encoder_vector = encoder_vector.reshape(b,  self.nf * seq_len, 50, 50)

        # decoder
        for t in range(future_step):

            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here


            output_dec_h += [h_t3]
            output_dec_c += [c_t3]

        output_dec_h = self.stack_permute(output_dec_h)

        output_last = self.decoder_CNN(output_dec_h)
        output_last = output_last.view((b, -1))

        return output_last

    def bottleneck_src(self, output_last):

        ln = self.Linear1_src(output_last)
        ln_relu = self.relu(ln)
        ln_last = self.Linear2_src(ln_relu)
        ln_last_relu = self.relu(ln_last)
        pred = self.Linear3_src(ln_last_relu)

        return pred, ln_last, ln

    def bottleneck_tar(self, output_last):

        ln = self.Linear1_tar(output_last)
        ln_relu = self.relu(ln)
        ln_last = self.Linear2_tar(ln_relu)
        ln_last_relu = self.relu(ln_last)
        pred = self.Linear3_tar(ln_last_relu)

        return pred, ln_last, ln

    def stack_permute(self, vec):
        vec = torch.stack(vec, 1)
        vec = vec.permute(0,2,1,3,4)
        return vec

    def forward(self, x, future_step, source_only, domain):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """
        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        # h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        # h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))


        output_last = self.autoencoder(x, seq_len, future_step, h_t, c_t, h_t3, c_t3)
        if domain == "src":

            outputs, feat1, feat2 = self.bottleneck_src(output_last)

        elif domain == "tar":
            outputs, feat1, feat2 = self.bottleneck_tar(output_last)

        return outputs, feat1, feat2
