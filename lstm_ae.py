
#%%
from typing import DefaultDict
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from models import ConvLSTMCell
from torch.autograd import Function

class EncoderDecoderConvLSTM(nn.Module):

    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTM, self).__init__()

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

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=3,
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf*15,  # nf + 1
                                               hidden_dim=nf*15,
                                               kernel_size=3,
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf*15,
                                               hidden_dim=nf*15,
                                               kernel_size=3,
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf*15,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))

        self.Linear1 = nn.Linear(17500, 2400)
        self.Linear2 = nn.Linear(2400, 600)
        self.Linear3 = nn.Linear(600, 7)
        self.relu = nn.ReLU(inplace=True)

    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []
        output_enc_h = []
        output_enc_c = []
        output_dec_h = []
        output_dec_c = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here

            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

            output_enc_h += [h_t2]
            output_enc_c += [c_t2]

        output_enc_h = self.stack_permute(output_enc_h)
        output_enc_c = self.stack_permute(output_enc_c)

        encoder_vector = output_enc_h.reshape(1, 32 * seq_len, 50, 50)
        b = encoder_vector.shape[0]

        # decoder
        for t in range(future_step):

            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here


            # h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
            #                                      cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here

            output_dec_h += [h_t3]
            output_dec_c += [c_t3]
            # encoder_vector = h_t4
            # output_enc_c += [c_t]

            # outputs += [h_t4]  # predictions



        output_dec_h = self.stack_permute(output_dec_h)
        output_dec_c = self.stack_permute(output_dec_c)

        output_last = self.decoder_CNN(output_dec_h)
        # outputs_last = outputs_last.permute(0, 2, 1, 3, 4)
        output_last = output_last.view((b, -1))

        ln_1 = self.Linear1(output_last)
        ln_1_relu = self.relu(ln_1)
        ln_2 = self.Linear2(ln_1_relu)
        ln_2_relu = self.relu(ln_2)
        pred = self.Linear3(ln_2_relu)

        return pred, output_last, ln_1, ln_2

    def stack_permute(self, vec):
        vec = torch.stack(vec, 1)
        vec = vec.permute(0,2,1,3,4)
        return vec

    def forward(self, x, future_step):

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
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs, feat1, feat2, feat3 = self.autoencoder(x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs, feat1, feat2, feat3

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
