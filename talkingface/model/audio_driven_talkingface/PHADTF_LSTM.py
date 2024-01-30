import math
from typing import Union, Sequence

import torch
from torch.nn import Parameter
from torch.nn.utils.rnn import PackedSequence

from .functional import AutogradConvRNN, _conv_cell_helper
from .utils import _single, _pair, _triple

from functools import partial

import collections
from itertools import repeat

import os
import glob
import time
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.modules.module import _addindent
import numpy as np
from collections import OrderedDict
import argparse

from dataset import LRW_1D_lstm_3dmm, LRW_1D_lstm_3dmm_pose
from dataset import News_1D_lstm_3dmm_pose

from models import ATC_net

from torch.nn import init
import pdb
import torch
import torch.nn.functional as F
#from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend

from .utils import _single, _pair, _triple


def RNNReLUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, linear_func=None):
    """ Copied from torch.nn._functions.rnn and modified """
    if linear_func is None:
        linear_func = F.linear
    hy = F.relu(linear_func(input, w_ih, b_ih) + linear_func(hidden, w_hh, b_hh))
    return hy


def RNNTanhCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, linear_func=None):
    """ Copied from torch.nn._functions.rnn and modified """
    if linear_func is None:
        linear_func = F.linear
    hy = F.tanh(linear_func(input, w_ih, b_ih) + linear_func(hidden, w_hh, b_hh))
    return hy


def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, linear_func=None):
    """ Copied from torch.nn._functions.rnn and modified """
    if linear_func is None:
        linear_func = F.linear
    if input.is_cuda and linear_func is F.linear:
        igates = linear_func(input, w_ih)
        hgates = linear_func(hidden[0], w_hh)
        #state = fusedBackend.LSTMFused.apply
        #return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)
        # Slice off the workspace arg (used only for backward)
        return _cuda_fused_lstm_cell(igates, hgates, hidden[1], b_ih, b_hh)[:2]

    hx, cx = hidden
    gates = linear_func(input, w_ih, b_ih) + linear_func(hx, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy


def PeepholeLSTMCell(input, hidden, w_ih, w_hh, w_pi, w_pf, w_po,
                     b_ih=None, b_hh=None, linear_func=None):
    if linear_func is None:
        linear_func = F.linear
    hx, cx = hidden
    gates = linear_func(input, w_ih, b_ih) + linear_func(hx, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate += linear_func(cx, w_pi)
    forgetgate += linear_func(cx, w_pf)
    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    outgate += linear_func(cy, w_po)
    outgate = F.sigmoid(outgate)

    hy = outgate * F.tanh(cy)

    return hy, cy


def GRUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, linear_func=None):
    """ Copied from torch.nn._functions.rnn and modified """
    if linear_func is None:
        linear_func = F.linear
    if input.is_cuda and linear_func is F.linear:
        gi = linear_func(input, w_ih)
        gh = linear_func(hidden, w_hh)
        #state = fusedBackend.GRUFused.apply
        #return state(gi, gh, hidden) if b_ih is None else state(gi, gh, hidden, b_ih, b_hh)
        return _cuda_fused_gru_cell(gi, gh, hidden, b_ih, b_hh)[0]
    gi = linear_func(input, w_ih, b_ih)
    gh = linear_func(hidden, w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = F.sigmoid(i_r + h_r)
    inputgate = F.sigmoid(i_i + h_i)
    newgate = F.tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy


def StackedRNN(inners, num_layers, lstm=False, dropout=0, train=True):
    """ Copied from torch.nn._functions.rnn and modified """

    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight, batch_sizes):
        assert(len(weight) == total_layers)
        next_hidden = []
        ch_dim = input.dim() - weight[0][0].dim() + 1

        if lstm:
            hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j

                hy, output = inner(input, hidden[l], weight[l], batch_sizes)
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, ch_dim)

            if dropout != 0 and i < num_layers - 1:
                input = F.dropout(input, p=dropout, training=train, inplace=False)

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(
                total_layers, *next_hidden[0].size())

        return next_hidden, input

    return forward


def Recurrent(inner, reverse=False):
    """ Copied from torch.nn._functions.rnn without any modification """
    def forward(input, hidden, weight, batch_sizes):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = inner(input[i], hidden, *weight)
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    return forward


def variable_recurrent_factory(inner, reverse=False):
    """ Copied from torch.nn._functions.rnn without any modification """
    if reverse:
        return VariableRecurrentReverse(inner)
    else:
        return VariableRecurrent(inner)


def VariableRecurrent(inner):
    """ Copied from torch.nn._functions.rnn without any modification """
    def forward(input, hidden, weight, batch_sizes):
        output = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
        for batch_size in batch_sizes:
            step_input = input[input_offset:input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(tuple(h[-dec:] for h in hidden))
                hidden = tuple(h[:-dec] for h in hidden)
            last_batch_size = batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                hidden = inner(step_input, hidden, *weight)

            output.append(hidden[0])
        hiddens.append(hidden)
        hiddens.reverse()

        hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        assert hidden[0].size(0) == batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = torch.cat(output, 0)

        return hidden, output

    return forward


def VariableRecurrentReverse(inner):
    """ Copied from torch.nn._functions.rnn without any modification """
    def forward(input, hidden, weight, batch_sizes):
        output = []
        input_offset = input.size(0)
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
            initial_hidden = (initial_hidden,)
        hidden = tuple(h[:batch_sizes[-1]] for h in hidden)
        for i in reversed(range(len(batch_sizes))):
            batch_size = batch_sizes[i]
            inc = batch_size - last_batch_size
            if inc > 0:
                hidden = tuple(torch.cat((h, ih[last_batch_size:batch_size]), 0)
                               for h, ih in zip(hidden, initial_hidden))
            last_batch_size = batch_size
            step_input = input[input_offset - batch_size:input_offset]
            input_offset -= batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                hidden = inner(step_input, hidden, *weight)
            output.append(hidden[0])

        output.reverse()
        output = torch.cat(output, 0)
        if flat_hidden:
            hidden = hidden[0]
        return hidden, output

    return forward


def ConvNdWithSamePadding(convndim=2, stride=1, dilation=1, groups=1):
    def forward(input, w, b=None):
        if convndim == 1:
            ntuple = _single
        elif convndim == 2:
            ntuple = _pair
        elif convndim == 3:
            ntuple = _triple
        else:
            raise ValueError('convndim must be 1, 2, or 3, but got {}'.format(convndim))

        if input.dim() != convndim + 2:
            raise RuntimeError('Input dim must be {}, bot got {}'.format(convndim + 2, input.dim()))
        if w.dim() != convndim + 2:
            raise RuntimeError('w must be {}, bot got {}'.format(convndim + 2, w.dim()))

        insize = input.shape[2:]
        kernel_size = w.shape[2:]
        _stride = ntuple(stride)
        _dilation = ntuple(dilation)

        ps = [(i + 1 - h + s * (h - 1) + d * (k - 1)) // 2
              for h, k, s, d in list(zip(insize, kernel_size, _stride, _dilation))[::-1] for i in range(2)]
        # Padding to make the output shape to have the same shape as the input
        input = F.pad(input, ps, 'constant', 0)
        return getattr(F, 'conv{}d'.format(convndim))(
            input, w, b, stride=_stride, padding=ntuple(0), dilation=_dilation, groups=groups)
    return forward


def _conv_cell_helper(mode, convndim=2, stride=1, dilation=1, groups=1):
    linear_func = ConvNdWithSamePadding(convndim=convndim, stride=stride, dilation=dilation, groups=groups)

    if mode == 'RNN_RELU':
        cell = partial(RNNReLUCell, linear_func=linear_func)
    elif mode == 'RNN_TANH':
        cell = partial(RNNTanhCell, linear_func=linear_func)
    elif mode == 'LSTM':
        cell = partial(LSTMCell, linear_func=linear_func)
    elif mode == 'GRU':
        cell = partial(GRUCell, linear_func=linear_func)
    elif mode == 'PeepholeLSTM':
        cell = partial(PeepholeLSTMCell, linear_func=linear_func)
    else:
        raise Exception('Unknown mode: {}'.format(mode))
    return cell


def AutogradConvRNN(
        mode, num_layers=1, batch_first=False,
        dropout=0, train=True, bidirectional=False, variable_length=False,
        convndim=2, stride=1, dilation=1, groups=1):
    """ Copied from torch.nn._functions.rnn and modified """
    cell = _conv_cell_helper(mode, convndim=convndim, stride=stride, dilation=dilation, groups=groups)

    rec_factory = variable_recurrent_factory if variable_length else Recurrent

    if bidirectional:
        layer = (rec_factory(cell), rec_factory(cell, reverse=True))
    else:
        layer = (rec_factory(cell),)

    func = StackedRNN(layer, num_layers, (mode in ('LSTM', 'PeepholeLSTM')), dropout=dropout, train=train)

    def forward(input, weight, hidden, batch_sizes):
        if batch_first and batch_sizes is None:
            input = input.transpose(0, 1)

        nexth, output = func(input, hidden, weight, batch_sizes)

        if batch_first and batch_sizes is None:
            output = output.transpose(0, 1)

        return output, nexth

    return forward

class ConvNdRNNBase(torch.nn.Module):
    def __init__(self,
                 mode,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 convndim=2,
                 stride=1,
                 dilation=1,
                 groups=1):
        super(ConvNdRNNBase, self).__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.convndim = convndim

        if convndim == 1:
            ntuple = _single
        elif convndim == 2:
            ntuple = _pair
        elif convndim == 3:
            ntuple = _triple
        else:
            raise ValueError('convndim must be 1, 2, or 3, but got {}'.format(convndim))

        self.kernel_size = ntuple(kernel_size)
        self.stride = ntuple(stride)
        self.dilation = ntuple(dilation)

        self.groups = groups

        num_directions = 2 if bidirectional else 1

        if mode in ('LSTM', 'PeepholeLSTM'):
            gate_size = 4 * out_channels
        elif mode == 'GRU':
            gate_size = 3 * out_channels
        else:
            gate_size = out_channels

        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = in_channels if layer == 0 else out_channels * num_directions
                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size // groups, *self.kernel_size))
                w_hh = Parameter(torch.Tensor(gate_size, out_channels // groups, *self.kernel_size))

                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))

                if mode == 'PeepholeLSTM':
                    w_pi = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))
                    w_pf = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))
                    w_po = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))
                    layer_params = (w_ih, w_hh, w_pi, w_pf, w_po, b_ih, b_hh)
                    param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}',
                                   'weight_pi_l{}{}', 'weight_pf_l{}{}', 'weight_po_l{}{}']
                else:
                    layer_params = (w_ih, w_hh, b_ih, b_hh)
                    param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']

                suffix = '_reverse' if direction == 1 else ''
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_channels)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = (2 if is_input_packed else 3) + self.convndim
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        ch_dim = 1 if is_input_packed else 2
        if self.in_channels != input.size(ch_dim):
            raise RuntimeError(
                'input.size({}) must be equal to in_channels . Expected {}, got {}'.format(
                    ch_dim, self.in_channels, input.size(ch_dim)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.out_channels) + input.shape[ch_dim + 1:]

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode in ('LSTM', 'PeepholeLSTM'):
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
            insize = input.shape[2:]
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            insize = input.shape[3:]

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(self.num_layers * num_directions, max_batch_size, self.out_channels,
                                 *insize, requires_grad=False)
            if self.mode in ('LSTM', 'PeepholeLSTM'):
                hx = (hx, hx)

        self.check_forward_args(input, hx, batch_sizes)
        func = AutogradConvRNN(
            self.mode,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            variable_length=batch_sizes is not None,
            convndim=self.convndim,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups
            )
        output, hidden = func(input, self.all_weights, hx, batch_sizes)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(ConvNdRNNBase, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                if self.mode == 'PeepholeLSTM':
                    weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}',
                               'weight_pi_l{}{}', 'weight_pf_l{}{}', 'weight_po_l{}{}',
                               'bias_ih_l{}{}', 'bias_hh_l{}{}']
                else:
                    weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}',
                               'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:len(weights) // 2]]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]



class Conv2dRNN(ConvNdRNNBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 nonlinearity='tanh',
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 stride=1,
                 dilation=1,
                 groups=1):
        if nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super().__init__(
            mode=mode,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            convndim=2,
            stride=stride,
            dilation=dilation,
            groups=groups)


class Conv2dLSTM(ConvNdRNNBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 stride=1,
                 dilation=1,
                 groups=1):
        super().__init__(
            mode='LSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            convndim=2,
            stride=stride,
            dilation=dilation,
            groups=groups)


class Conv2dPeepholeLSTM(ConvNdRNNBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 stride=1,
                 dilation=1,
                 groups=1):
        super().__init__(
            mode='PeepholeLSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            convndim=2,
            stride=stride,
            dilation=dilation,
            groups=groups)


class Conv2dGRU(ConvNdRNNBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 stride=1,
                 dilation=1,
                 groups=1):
        super(Conv2dGRU, self).__init__(
            mode='GRU',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            convndim=2,
            stride=stride,
            dilation=dilation,
            groups=groups)


class Conv3dRNN(ConvNdRNNBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 nonlinearity='tanh',
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 stride=1,
                 dilation=1,
                 groups=1):
        if nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super().__init__(
            mode=mode,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            convndim=3,
            stride=stride,
            dilation=dilation,
            groups=groups)


class Conv3dLSTM(ConvNdRNNBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 stride=1,
                 dilation=1,
                 groups=1):
        super().__init__(
            mode='LSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            convndim=3,
            stride=stride,
            dilation=dilation,
            groups=groups)


class Conv3dPeepholeLSTM(ConvNdRNNBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 stride=1,
                 dilation=1,
                 groups=1):
        super().__init__(
            mode='PeepholeLSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            convndim=3,
            stride=stride,
            dilation=dilation,
            groups=groups)


class Conv3dGRU(ConvNdRNNBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 stride=1,
                 dilation=1,
                 groups=1):
        super().__init__(
            mode='GRU',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            convndim=3,
            stride=stride,
            dilation=dilation,
            groups=groups)


class ConvRNNCellBase(torch.nn.Module):
    def __init__(self,
                 mode,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 convndim=2,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.convndim = convndim

        if convndim == 1:
            ntuple = _single
        elif convndim == 2:
            ntuple = _pair
        elif convndim == 3:
            ntuple = _triple
        else:
            raise ValueError('convndim must be 1, 2, or 3, but got {}'.format(convndim))

        self.kernel_size = ntuple(kernel_size)
        self.stride = ntuple(stride)
        self.dilation = ntuple(dilation)

        self.groups = groups

        if mode in ('LSTM', 'PeepholeLSTM'):
            gate_size = 4 * out_channels
        elif mode == 'GRU':
            gate_size = 3 * out_channels
        else:
            gate_size = out_channels

        self.weight_ih = Parameter(torch.Tensor(gate_size, in_channels // groups, *self.kernel_size))
        self.weight_hh = Parameter(torch.Tensor(gate_size, out_channels // groups, *self.kernel_size))

        if bias:
            self.bias_ih = Parameter(torch.Tensor(gate_size))
            self.bias_hh = Parameter(torch.Tensor(gate_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        if mode == 'PeepholeLSTM':
            self.weight_pi = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))
            self.weight_pf = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))
            self.weight_po = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))

        self.reset_parameters()

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.in_channels:
            raise RuntimeError(
                "input has inconsistent channels: got {}, expected {}".format(
                    input.size(1), self.in_channels))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.out_channels:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.out_channels))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_channels)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        self.check_forward_input(input)

        if hx is None:
            batch_size = input.size(0)
            insize = input.shape[2:]
            hx = input.new_zeros(batch_size, self.out_channels, *insize, requires_grad=False)
            if self.mode in ('LSTM', 'PeepholeLSTM'):
                hx = (hx, hx)
        if self.mode in ('LSTM', 'PeepholeLSTM'):
            self.check_forward_hidden(input, hx[0])
            self.check_forward_hidden(input, hx[1])
        else:
            self.check_forward_hidden(input, hx)

        cell = _conv_cell_helper(
            self.mode,
            convndim=self.convndim,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups)
        if self.mode == 'PeepholeLSTM':
            return cell(
                input, hx,
                self.weight_ih, self.weight_hh, self.weight_pi, self.weight_pf, self.weight_po,
                self.bias_ih, self.bias_hh
            )
        else:
            return cell(
                input, hx,
                self.weight_ih, self.weight_hh,
                self.bias_ih, self.bias_hh,
            )
""" Copied from torch.nn.modules.utils """


    def _ntuple(n):
         def parse(x):
             if isinstance(x, collections.Iterable):
                return x
             return tuple(repeat(x, n))
        return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

class Conv1dRNNCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 nonlinearity='tanh',
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        if nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super().__init__(
            mode=mode,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=1,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv1dLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='LSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=1,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv1dPeepholeLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='PeepholeLSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=1,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv1dGRUCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='GRU',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=1,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv2dRNNCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 nonlinearity='tanh',
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        if nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super().__init__(
            mode=mode,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=2,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv2dLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='LSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=2,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv2dPeepholeLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='PeepholeLSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=2,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv2dGRUCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='GRU',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=2,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv3dRNNCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 nonlinearity='tanh',
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        if nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super().__init__(
            mode=mode,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=3,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv3dLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='LSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=3,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv3dPeepholeLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='PeepholeLSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=3,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv3dGRUCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='GRU',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=3,
            stride=stride,
            dilation=dilation,
            groups=groups
        )

def multi2single(model_path, id):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint
    if id ==1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def initialize_weights( net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

class Trainer():
    def __init__(self, config):
        if config.lstm == True:
            if config.pose == 0:
                self.generator = ATC_net(config.para_dim)
            else:
                self.generator = ATC_net(config.para_dim+6)
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in self.generator.parameters():
            num_params += param.numel()
        print('[Network] Total number of parameters : %.3f M' % ( num_params / 1e6))
        print('-----------------------------------------------')
        #pdb.set_trace()
        self.l1_loss_fn =  nn.L1Loss()
        self.mse_loss_fn = nn.MSELoss()
        self.config = config

        if config.cuda:
            device_ids = [int(i) for i in config.device_ids.split(',')]
            if len(device_ids) > 1:
                self.generator     = nn.DataParallel(self.generator, device_ids=device_ids).cuda()
            else:
                self.generator     = self.generator.cuda()
            self.mse_loss_fn   = self.mse_loss_fn.cuda()
            self.l1_loss_fn = self.l1_loss_fn.cuda()
        initialize_weights(self.generator)
        if config.continue_train:
            state_dict = multi2single(config.model_name, 0)
            self.generator.load_state_dict(state_dict)
            print('load pretrained [{}]'.format(config.model_name))
        self.start_epoch = 0
        if config.load_model:
            self.start_epoch = config.start_epoch
            self.load(config.pretrained_dir, config.pretrained_epoch)
        self.opt_g = torch.optim.Adam( self.generator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))
        if config.lstm:
            if config.pose == 0:
                self.dataset = LRW_1D_lstm_3dmm(config.dataset_dir, train=config.is_train, indexes=config.indexes)
            else:
                if config.dataset == 'lrw':
                    self.dataset = LRW_1D_lstm_3dmm_pose(config.dataset_dir, train=config.is_train, indexes=config.indexes, relativeframe=config.relativeframe)
                    self.dataset2 = LRW_1D_lstm_3dmm_pose(config.dataset_dir, train='test', indexes=config.indexes, relativeframe=config.relativeframe)
                elif config.dataset == 'news':
                    self.dataset = News_1D_lstm_3dmm_pose(config.dataset_dir, train=config.is_train, indexes=config.indexes, relativeframe=config.relativeframe,
                                newsname=config.newsname, start=config.start, trainN=config.trainN, testN=config.testN)
        
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
        if config.dataset == 'lrw':
            self.data_loader_val = DataLoader(self.dataset2,
                                      batch_size=config.batch_size,
                                      num_workers= config.num_thread,
                                      shuffle=False, drop_last=True)
        

    def fit(self):
        config = self.config
        L = config.para_dim

        num_steps_per_epoch = len(self.data_loader)
        print("num_steps_per_epoch", num_steps_per_epoch)
        cc = 0
        t00 = time.time()
        t0 = time.time()
        

        for epoch in range(self.start_epoch, config.max_epochs):
            for step, (coeff, audio, coeff2) in enumerate(self.data_loader):
                t1 = time.time()

                if config.cuda:
                    coeff = Variable(coeff.float()).cuda()
                    audio = Variable(audio.float()).cuda()
                else:
                    coeff = Variable(coeff.float())
                    audio = Variable(audio.float())

                #print(audio.shape, coeff.shape) # torch.Size([16, 16, 28, 12]) torch.Size([16, 16, 70])
                fake_coeff= self.generator(audio)

                loss =  self.mse_loss_fn(fake_coeff , coeff)
                
                if config.less_constrain:
                    loss =  self.mse_loss_fn(fake_coeff[:,:,:L], coeff[:,:,:L]) + config.lambda_pose * self.mse_loss_fn(fake_coeff[:,:,L:], coeff[:,:,L:])
                
                # put smooth on pose
                # tidu ermo pingfang
                if config.smooth_loss:
                    loss1 = loss.clone()
                    frame_dif = fake_coeff[:,1:,L:] - fake_coeff[:,:-1,L:] # [16, 15, 6]
                    #norm2 = torch.norm(frame_dif, dim = 1) # default 2-norm, [16, 6]
                    #norm2_ss1 = torch.sum(torch.mul(norm2, norm2), dim=1) # [16, 1]
                    norm2_ss = torch.sum(torch.mul(frame_dif,frame_dif), dim=[1,2])
                    loss2 = torch.mean(norm2_ss)
                    #pdb.set_trace()
                    loss = loss1 + loss2 * config.lambda_smooth
                
                # put smooth on expression
                if config.smooth_loss2:
                    loss3 = loss.clone()
                    frame_dif2 = fake_coeff[:,1:,:L] - fake_coeff[:,:-1,:L]
                    norm2_ss2 = torch.sum(torch.mul(frame_dif2,frame_dif2), dim=[1,2])
                    loss4 = torch.mean(norm2_ss2)
                    loss = loss3 + loss4 * config.lambda_smooth2


                loss.backward() 
                self.opt_g.step()
                self._reset_gradients()


                if (step+1) % 10 == 0 or (step+1) == num_steps_per_epoch:
                    steps_remain = num_steps_per_epoch-step+1 + \
                        (config.max_epochs-epoch+1)*num_steps_per_epoch

                    if not config.smooth_loss and not config.smooth_loss2:
                        print("[{}/{}][{}/{}]   loss1: {:.8f},data time: {:.4f},  model time: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss,  t1-t0,  time.time() - t1))
                    elif config.smooth_loss and not config.smooth_loss2:
                        print("[{}/{}][{}/{}]   loss1: {:.8f},lossgt: {:.8f},losstv: {:.8f},data time: {:.4f},  model time: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss,  loss1, loss2*config.lambda_smooth, t1-t0,  time.time() - t1))
                    elif not config.smooth_loss and config.smooth_loss2:
                        print("[{}/{}][{}/{}]   loss1: {:.8f},lossgt: {:.8f},losstv2: {:.8f},data time: {:.4f},  model time: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss,  loss3, loss4*config.lambda_smooth2, t1-t0,  time.time() - t1))
                    else:
                        print("[{}/{}][{}/{}]   loss1: {:.8f},lossgt: {:.8f},losstv: {:.8f},losstv2: {:.8f},data time: {:.4f},  model time: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss,  loss1, loss2*config.lambda_smooth, loss4*config.lambda_smooth2, t1-t0,  time.time() - t1))
                
                if (num_steps_per_epoch > 100 and (step) % (int(num_steps_per_epoch  / 10 )) == 0 and step != 0) or (num_steps_per_epoch <= 100 and (step+1) == num_steps_per_epoch):
                    if config.lstm:
                        for indx in range(3):
                            for jj in range(16):
                                name = "{}/real_{}_{}_{}.npy".format(config.sample_dir,cc, indx,jj)
                                coeff2 = coeff.data.cpu().numpy()
                                np.save(name, coeff2[indx,jj])
                                if config.relativeframe:
                                    name = "{}/real2_{}_{}_{}.npy".format(config.sample_dir,cc, indx,jj)
                                    np.save(name, coeff2[indx,jj])
                                name = "{}/fake_{}_{}_{}.npy".format(config.sample_dir,cc, indx,jj)
                                fake_coeff2 = fake_coeff.data.cpu().numpy()
                                np.save(name, fake_coeff2[indx,jj]) 
                        # check val set loss
                        vloss = 0
                        if config.dataset == 'lrw':
                            for step,  (coeff, audio, coeff2) in enumerate(self.data_loader_val):
                                with torch.no_grad():
                                    if step == 100:
                                        break
                                    if config.cuda:
                                        coeff = Variable(coeff.float()).cuda()
                                        audio = Variable(audio.float()).cuda()
                                    fake_coeff= self.generator(audio)
                                    valloss = self.mse_loss_fn(fake_coeff,coeff)
                                    if config.less_constrain:
                                        valloss =  self.mse_loss_fn(fake_coeff[:,:,:L], coeff[:,:,:L]) + config.lambda_pose * self.mse_loss_fn(fake_coeff[:,:,L:], coeff[:,:,L:])
                                    vloss += valloss.cpu().numpy()
                            print("[{}/{}][{}/{}]   val loss:{}".format(epoch+1, config.max_epochs,
                                    step+1, num_steps_per_epoch, vloss/100.))
                        # save model
                        print("[{}/{}][{}/{}]   save model".format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch))
                        torch.save(self.generator.state_dict(),
                                   "{}/atcnet_lstm_{}.pth"
                                   .format(config.model_dir,cc))
                    cc += 1
                 
                t0 = time.time()
        print("total time: {} second".format(time.time()-t00))
 
    def _reset_gradients(self):
        self.generator.zero_grad()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.0002)
    parser.add_argument("--beta1",
                        type=float,
                        default=0.5)
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999)
    parser.add_argument("--lambda1",
                        type=int,
                        default=100)
    parser.add_argument("--batch_size",
                        type=int,
                        default=16)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=10)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="../dataset/")
    parser.add_argument("--model_dir",
                        type=str,
                        default="../model/atcnet/")
    parser.add_argument("--sample_dir",
                        type=str,
                        default="../sample/atcnet/")
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='lrw')
    parser.add_argument('--lstm', type=bool, default= True)
    parser.add_argument('--num_thread', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--pretrained_dir', type=str)
    parser.add_argument('--pretrained_epoch', type=int)
    parser.add_argument('--start_epoch', type=int, default=0, help='start from 0')
    parser.add_argument('--rnn', type=bool, default=True)
    parser.add_argument('--para_dim', type=int, default=64)
    parser.add_argument('--index', type=str, default='80,144', help='index ranges')
    parser.add_argument('--pose', type=int, default=0, help='whether predict pose')
    parser.add_argument('--relativeframe', type=int, default=0, help='whether use relative frame value for pose')
    # for personalized data
    parser.add_argument('--newsname', type=str, default='Learn_English')
    parser.add_argument('--start', type=int, default=357)
    parser.add_argument('--trainN', type=int, default=300)
    parser.add_argument('--testN', type=int, default=100)
    # for continnue train
    parser.add_argument('--continue_train', type=bool, default=False)
    parser.add_argument("--model_name", type=str, default='../model/atcnet_pose0/atcnet_lstm_24.pth')
    parser.add_argument('--preserve_mouth', type=bool, default=False)
    # for remove jittering
    parser.add_argument('--smooth_loss', type=bool, default=False) # smooth in time, similar to total variation
    parser.add_argument('--smooth_loss2', type=bool, default=False) # smooth in time, for expression
    parser.add_argument('--lambda_smooth', type=float, default=0.01)
    parser.add_argument('--lambda_smooth2', type=float, default=0.0001)
    # for less constrain for pose
    parser.add_argument('--less_constrain', type=bool, default=False)
    parser.add_argument('--lambda_pose', type=float, default=0.2)

    return parser.parse_args()


def main(config):
    t = trainer.Trainer(config)
    t.fit()

if __name__ == "__main__":

    config = parse_args()
    str_ids = config.index.split(',')
    config.indexes = []
    for i in range(int(len(str_ids)/2)):
        start = int(str_ids[2*i])
        end = int(str_ids[2*i+1])
        if end > start:
            config.indexes += range(start, end)
    #print('indexes', config.indexes)
    print('device', config.device_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device_ids
    config.is_train = 'train'
    import atcnet as trainer
    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    if not os.path.exists(config.sample_dir):
        os.mkdir(config.sample_dir)
    config.cuda1 = torch.device('cuda:{}'.format(config.device_ids))
    main(config)


