import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from collections import OrderedDict
from talkingface.utils import text_to_sequence
from talkingface.utils.waveglow.glow import WaveGlow
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

from talkingface.data.dataset.fastspeech_dataset import FastSpeechDataset
from talkingface.model.abstract_talkingface import AbstractTalkingFace
from talkingface.utils import waveglow
from talkingface.utils import audio

from talkingface.utils.fastspeech_transformer.Models import Encoder, Decoder
from talkingface.utils.fastspeech_transformer.Layers import Linear,PostNet


class FastSpeech(AbstractTalkingFace):
    """ FastSpeech """

    def __init__(self,config):
        super(FastSpeech, self).__init__()

        self.config=config
        self.encoder = Encoder(config=config)
        self.length_regulator = LengthRegulator(config)
        self.decoder = Decoder(config=config)

        self.mel_linear = Linear(self.config['decoder_dim'], self.config['num_mels'])
        self.postnet = CBHG(self.config['num_mels'], K=8,
                            projections=[256, self.config['num_mels']])
        self.last_linear = Linear(self.config['num_mels'] * 2, self.config['num_mels'])

        #loss部分
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, alpha=1.0):
        encoder_output, _ = self.encoder(src_seq, src_pos)

        if self.training:
            length_regulator_output, duration_predictor_output = self.length_regulator(encoder_output,
                                                                                       target=length_target,
                                                                                       alpha=alpha,
                                                                                       mel_max_length=mel_max_length)
            decoder_output = self.decoder(length_regulator_output, mel_pos)

            mel_output = self.mel_linear(decoder_output)
            mel_output = self.mask_tensor(mel_output, mel_pos, mel_max_length)
            residual = self.postnet(mel_output)
            residual = self.last_linear(residual)
            mel_postnet_output = mel_output + residual
            mel_postnet_output = self.mask_tensor(mel_postnet_output,
                                                  mel_pos,
                                                  mel_max_length)

            return mel_output, mel_postnet_output, duration_predictor_output
        else:
            length_regulator_output, decoder_pos = self.length_regulator(encoder_output,
                                                                         alpha=alpha)

            decoder_output = self.decoder(length_regulator_output, decoder_pos)

            mel_output = self.mel_linear(decoder_output)
            residual = self.postnet(mel_output)
            residual = self.last_linear(residual)
            mel_postnet_output = mel_output + residual

            return mel_output, mel_postnet_output

    def calculate_loss(self, interaction):

        character = interaction["text"].long().to(self.config['device'])
        mel_target = interaction["mel_target"].float().to(self.config['device'])
        duration = interaction["duration"].int().to(self.config['device'])
        mel_pos = interaction["mel_pos"].long().to(self.config['device'])
        src_pos = interaction["src_pos"].long().to(self.config['device'])
        max_mel_len = interaction["mel_max_len"]

        mel_output, mel_postnet_output, duration_predictor_output = self.forward(character,
                                                                          src_pos,
                                                                          mel_pos=mel_pos,
                                                                          mel_max_length=max_mel_len,
                                                                          length_target=duration)
        mel_target.requires_grad = False
        mel_loss = self.mse_loss(mel_output, mel_target)
        mel_postnet_loss = self.mse_loss(mel_postnet_output, mel_target)

        duration.requires_grad = False
        duration_predictor_loss = self.l1_loss(duration_predictor_output,
                                               duration.float())
        total_loss=mel_loss+mel_postnet_loss+duration_predictor_loss
        loss_dict={
            'loss':total_loss,
            'mel_loss':mel_loss,
            'mel_postnet_loss':mel_postnet_loss,
            'duration_predictor_loss':duration_predictor_loss
        }
        return loss_dict

    def predict(self, src_seq,src_pos):
        return self.forward(src_seq,src_pos)

    def generate_batch(self):
        print('开始评估，生成语音文件地址为：', self.config['test_filelist'])
        WaveGlow = self.get_WaveGlow()

        file_dict = {'generated_video': [], 'real_video': []}
        generated_audio_list=[]
        real_audio_list=[]
        with open(self.config['test_filelist'], "r", encoding="utf-8") as f:
            txt_list = []
            for line in f.readlines():
                parts = line.strip().split('|')
                txt_list.append([parts[0],int(parts[1]),text_to_sequence(parts[2],self.config['text_cleaners'])])

        for name, i, phn in tqdm(txt_list):
            mel, mel_cuda = self.synthesis(phn)
            if not os.path.exists('results'):
                os.mkdir('results')
            if not os.path.exists(os.path.join('results','temp')):
                os.mkdir(os.path.join('results','temp'))
            if not os.path.exists(self.config['temp_dir']):
                os.mkdir(self.config['temp_dir'])

            waveglow.inference.inference(
                mel_cuda, WaveGlow,
                self.config['temp_dir'] +'/'+ str(name) + "_" + str(i) + "_waveglow.wav")
            generated_audio_path=os.path.join(self.config['temp_dir'],str(name)+"_" + str(i) + "_waveglow.wav")
            file_dict['generated_video'].append(generated_audio_path)
            real_audio_path=os.path.join(self.config['data_root'],'wavs',str(name)+'.wav')
            file_dict['real_video'].append(real_audio_path)
        return file_dict

    def get_WaveGlow(self):
        original_path = sys.path.copy()

        waveglow_path = self.config['waveglow_checkpoint_path']
        new_path = self.config['wavglwo_net_path']
        sys.path.append(new_path)

        warnings.filterwarnings("ignore")
        wave_glow = torch.load(waveglow_path)['model']
        sys.path = original_path
        wave_glow = wave_glow.remove_weightnorm(wave_glow)
        wave_glow.cuda().eval()
        for m in wave_glow.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')

        return wave_glow

    def synthesis(self, text, alpha=1.0):
        text = np.array(text)
        text = np.stack([text])
        src_pos = np.array([i + 1 for i in range(text.shape[1])])
        src_pos = np.stack([src_pos])
        sequence = torch.from_numpy(text).cuda().long()
        src_pos = torch.from_numpy(src_pos).cuda().long()

        with torch.no_grad():
            _, mel = self.predict(sequence, src_pos)
        return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat

#下面是长度调节器
class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self,config):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = DurationPredictor(config)
        self.config=config

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(self.config['device'])

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)

        if target is not None:
            output = self.LR(x, target, mel_max_length=mel_max_length)
            return output, duration_predictor_output
        else:
            duration_predictor_output = (
                (duration_predictor_output + 0.5) * alpha).int()
            output = self.LR(x, duration_predictor_output)
            mel_pos = torch.stack(
                [torch.Tensor([i+1 for i in range(output.size(1))])]).long().to(self.config['device'])

            return output, mel_pos


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self,config):
        super(DurationPredictor, self).__init__()
        self.config=config
        self.input_size = config['encoder_dim']
        self.filter_size = config['duration_predictor_filter_size']
        self.kernel = config['duration_predictor_kernel_size']
        self.conv_output_size = config['duration_predictor_filter_size']
        self.dropout = config['dropout']

        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(self.input_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("relu_1", nn.ReLU()),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("conv1d_2", Conv(self.filter_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("relu_2", nn.ReLU()),
            ("dropout_2", nn.Dropout(self.dropout))
        ]))

        self.linear_layer = Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class BatchNormConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding,
                 activation=None, w_init_gain='linear'):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_dim, out_dim,
                                kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.activation = activation

        torch.nn.init.xavier_uniform_(
            self.conv1d.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)


class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class CBHG(nn.Module):
    """CBHG module: a recurrent neural network composed of:
        - 1-d convolution banks
        - Highway networks + residual connections
        - Bidirectional gated recurrent units
    """

    def __init__(self, in_dim, K=16, projections=[128, 128]):
        super(CBHG, self).__init__()
        self.in_dim = in_dim
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList(
            [BatchNormConv1d(in_dim, in_dim, kernel_size=k, stride=1,
                             padding=k // 2, activation=self.relu)
             for k in range(1, K + 1)])
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        in_sizes = [K * in_dim] + projections[:-1]
        activations = [self.relu] * (len(projections) - 1) + [None]
        self.conv1d_projections = nn.ModuleList(
            [BatchNormConv1d(in_size, out_size, kernel_size=3, stride=1,
                             padding=1, activation=ac)
             for (in_size, out_size, ac) in zip(
                 in_sizes, projections, activations)])

        self.pre_highway = nn.Linear(projections[-1], in_dim, bias=False)
        self.highways = nn.ModuleList(
            [Highway(in_dim, in_dim) for _ in range(4)])

        self.gru = nn.GRU(
            in_dim, in_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        # (B, T_in, in_dim)
        x = inputs

        # Needed to perform conv1d on time-axis
        # (B, in_dim, T_in)
        if x.size(-1) == self.in_dim:
            x = x.transpose(1, 2)

        T = x.size(-1)

        # (B, in_dim*K, T_in)
        # Concat conv1d bank outputs
        x = torch.cat([conv1d(x)[:, :, :T]
                       for conv1d in self.conv1d_banks], dim=1)
        assert x.size(1) == self.in_dim * len(self.conv1d_banks)
        x = self.max_pool1d(x)[:, :, :T]

        for conv1d in self.conv1d_projections:
            x = conv1d(x)

        # (B, T_in, in_dim)
        # Back to the original shape
        x = x.transpose(1, 2)

        if x.size(-1) != self.in_dim:
            x = self.pre_highway(x)

        # Residual connection
        x += inputs
        for highway in self.highways:
            x = highway(x)

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True)

        # (B, T_in, in_dim*2)
        self.gru.flatten_parameters()
        outputs, _ = self.gru(x)

        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)

        return outputs

class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

if __name__ == "__main__":
    # Test
    config = {
        'num_mels': 80,
        'text_cleaners': ['english_cleaners'],

        'vocab_size': 300,
        'max_seq_len': 3000,

        'encoder_dim': 256,
        'encoder_n_layer': 4,
        'encoder_head': 2,
        'encoder_conv1d_filter_size': 1024,

        'decoder_dim': 256,
        'decoder_n_layer': 4,
        'decoder_head': 2,
        'decoder_conv1d_filter_size': 1024,

        'fft_conv1d_kernel': (9, 1),
        'fft_conv1d_padding': (4, 0),

        'duration_predictor_filter_size': 256,
        'duration_predictor_kernel_size': 3,
        'dropout': 0.1,
    }
    #确认模型可用
    model = FastSpeech(config=config)
    print(sum(param.numel() for param in model.parameters()))





