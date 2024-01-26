import math
import copy
import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F

from talkingface.utils.glowTTS_utils import commons
from talkingface.utils.glowTTS_utils import monotonic_align
from talkingface.utils.glowTTS_utils import attentions
from talkingface.utils.glowTTS_utils import modules
from talkingface.model.abstract_talkingface import AbstractTalkingFace


class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = attentions.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = attentions.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

  def forward(self, x, x_mask):
    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask


class TextEncoder(nn.Module):
  def __init__(self, 
      n_vocab, 
      out_channels, 
      hidden_channels, 
      filter_channels, 
      filter_channels_dp, 
      n_heads, 
      n_layers, 
      kernel_size, 
      p_dropout, 
      window_size=None,
      block_length=None,
      mean_only=False,
      prenet=False,
      gin_channels=0):

    super().__init__()

    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.filter_channels_dp = filter_channels_dp
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.window_size = window_size
    self.block_length = block_length
    self.mean_only = mean_only
    self.prenet = prenet
    self.gin_channels = gin_channels

    self.emb = nn.Embedding(n_vocab, hidden_channels)
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    if prenet:
      self.pre = modules.ConvReluNorm(hidden_channels, hidden_channels, hidden_channels, kernel_size=5, n_layers=3, p_dropout=0.5)
    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout,
      window_size=window_size,
      block_length=block_length,
    )

    self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)
    if not mean_only:
      self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)
    self.proj_w = DurationPredictor(hidden_channels + gin_channels, filter_channels_dp, kernel_size, p_dropout)
  
  def forward(self, x, x_lengths, g=None):
    x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    if self.prenet:
      x = self.pre(x, x_mask)
    x = self.encoder(x, x_mask)

    if g is not None:
      g_exp = g.expand(-1, -1, x.size(-1))
      x_dp = torch.cat([torch.detach(x), g_exp], 1)
    else:
      x_dp = torch.detach(x)

    x_m = self.proj_m(x) * x_mask
    if not self.mean_only:
      x_logs = self.proj_s(x) * x_mask
    else:
      x_logs = torch.zeros_like(x_m)

    logw = self.proj_w(x_dp, x_mask)
    return x_m, x_logs, logw, x_mask


class FlowSpecDecoder(nn.Module):
  def __init__(self, 
      in_channels, 
      hidden_channels, 
      kernel_size, 
      dilation_rate, 
      n_blocks, 
      n_layers, 
      p_dropout=0., 
      n_split=4,
      n_sqz=2,
      sigmoid_scale=False,
      gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_blocks = n_blocks
    self.n_layers = n_layers
    self.p_dropout = p_dropout
    self.n_split = n_split
    self.n_sqz = n_sqz
    self.sigmoid_scale = sigmoid_scale
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for b in range(n_blocks):
      self.flows.append(modules.ActNorm(channels=in_channels * n_sqz))
      self.flows.append(modules.InvConvNear(channels=in_channels * n_sqz, n_split=n_split))
      self.flows.append(
        attentions.CouplingBlock(
          in_channels * n_sqz,
          hidden_channels,
          kernel_size=kernel_size, 
          dilation_rate=dilation_rate,
          n_layers=n_layers,
          gin_channels=gin_channels,
          p_dropout=p_dropout,
          sigmoid_scale=sigmoid_scale))

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      flows = self.flows
      logdet_tot = 0
    else:
      flows = reversed(self.flows)
      logdet_tot = None

    if self.n_sqz > 1:
      x, x_mask = commons.squeeze(x, x_mask, self.n_sqz)
    for f in flows:
      if not reverse:
        x, logdet = f(x, x_mask, g=g, reverse=reverse)
        logdet_tot += logdet
      else:
        x, logdet = f(x, x_mask, g=g, reverse=reverse)
    if self.n_sqz > 1:
      x, x_mask = commons.unsqueeze(x, x_mask, self.n_sqz)
    return x, logdet_tot

  def store_inverse(self):
    for f in self.flows:
      f.store_inverse()


class FlowGenerator(AbstractTalkingFace):
  def __init__(self, 
      n_vocab, 
      hidden_channels, 
      filter_channels, 
      filter_channels_dp, 
      out_channels,
      kernel_size=3, 
      n_heads=2, 
      n_layers_enc=6,
      p_dropout=0., 
      n_blocks_dec=12, 
      kernel_size_dec=5, 
      dilation_rate=5, 
      n_block_layers=4,
      p_dropout_dec=0., 
      n_speakers=0, 
      gin_channels=0, 
      n_split=4,
      n_sqz=1,
      sigmoid_scale=False,
      window_size=None,
      block_length=None,
      mean_only=False,
      hidden_channels_enc=None,
      hidden_channels_dec=None,
      prenet=False,
      **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.filter_channels_dp = filter_channels_dp
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.n_heads = n_heads
    self.n_layers_enc = n_layers_enc
    self.p_dropout = p_dropout
    self.n_blocks_dec = n_blocks_dec
    self.kernel_size_dec = kernel_size_dec
    self.dilation_rate = dilation_rate
    self.n_block_layers = n_block_layers
    self.p_dropout_dec = p_dropout_dec
    self.n_speakers = n_speakers
    self.gin_channels = gin_channels
    self.n_split = n_split
    self.n_sqz = n_sqz
    self.sigmoid_scale = sigmoid_scale
    self.window_size = window_size
    self.block_length = block_length
    self.mean_only = mean_only
    self.hidden_channels_enc = hidden_channels_enc
    self.hidden_channels_dec = hidden_channels_dec
    self.prenet = prenet

    self.encoder = TextEncoder(
        n_vocab, 
        out_channels, 
        hidden_channels_enc or hidden_channels, 
        filter_channels, 
        filter_channels_dp, 
        n_heads, 
        n_layers_enc, 
        kernel_size, 
        p_dropout, 
        window_size=window_size,
        block_length=block_length,
        mean_only=mean_only,
        prenet=prenet,
        gin_channels=gin_channels)

    self.decoder = FlowSpecDecoder(
        out_channels, 
        hidden_channels_dec or hidden_channels, 
        kernel_size_dec, 
        dilation_rate, 
        n_blocks_dec, 
        n_block_layers, 
        p_dropout=p_dropout_dec, 
        n_split=n_split,
        n_sqz=n_sqz,
        sigmoid_scale=sigmoid_scale,
        gin_channels=gin_channels)

    if n_speakers > 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels)
      nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

  def forward(self, x, x_lengths, y=None, y_lengths=None, g=None, gen=False, noise_scale=1., length_scale=1.):
    if g is not None:
      g = F.normalize(self.emb_g(g)).unsqueeze(-1) # [b, h]
    x_m, x_logs, logw, x_mask = self.encoder(x, x_lengths, g=g)

    if gen:
      w = torch.exp(logw) * x_mask * length_scale
      w_ceil = torch.ceil(w)
      y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
      y_max_length = None
    else:
      y_max_length = y.size(2)
    y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)
    z_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

    if gen:
      attn = commons.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
      z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
      z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
      logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

      z = (z_m + torch.exp(z_logs) * torch.randn_like(z_m) * noise_scale) * z_mask
      y, logdet = self.decoder(z, z_mask, g=g, reverse=True)
      return (y, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_)
    else:
      z, logdet = self.decoder(y, z_mask, g=g, reverse=False)
      with torch.no_grad():
        x_s_sq_r = torch.exp(-2 * x_logs)
        logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1) # [b, t, 1]
        logp2 = torch.matmul(x_s_sq_r.transpose(1,2), -0.5 * (z ** 2)) # [b, t, d] x [b, d, t'] = [b, t, t']
        logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1,2), z) # [b, t, d] x [b, d, t'] = [b, t, t']
        logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1) # [b, t, 1]
        logp = logp1 + logp2 + logp3 + logp4 # [b, t, t']

        attn = monotonic_align.maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
      z_m = torch.matmul(attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
      z_logs = torch.matmul(attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
      logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask
      return (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_)

  def preprocess(self, y, y_lengths, y_max_length):
    if y_max_length is not None:
      y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
      y = y[:,:,:y_max_length]
    y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
    return y, y_lengths, y_max_length

  def store_inverse(self):
    self.decoder.store_inverse()


  def calculate_loss(self, interaction):
    """Calculate the training loss for a batch data.

    Args:
        interaction (Interaction): Interaction class of the batch.

    Returns:
        dict: {"loss": loss, "xxx": xxx}
        返回是一个字典,loss 这个键必须有,它代表了加权之后的总loss。
        因为有时总loss可能由多个部分组成。xxx代表其它各部分loss
    """
    from talkingface.utils.glowTTS_utils import commons
    _interaction,generator = interaction
    x, x_lengths = _interaction["text_padded"],_interaction["input_lengths"]
    y, y_lengths = _interaction["mel_padded"],_interaction["output_lengths"]
    x, x_lengths = x.cuda(non_blocking=True), x_lengths.cuda(non_blocking=True)
    y, y_lengths = y.cuda(non_blocking=True), y_lengths.cuda(non_blocking=True)

    
    (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = generator(x, x_lengths, y, y_lengths, gen=False)
    l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
    l_length = commons.duration_loss(logw, logw_, x_lengths)

    loss_gs = [l_mle, l_length]
    loss_g = sum(loss_gs)
    return {"loss":loss_g,"loss_gs":loss_gs}


def predict(self, interaction):
    """Predict the scores between users and items.

    Args:
        interaction (Interaction): Interaction class of the batch.

    Returns:
        video/image numpy/tensor
    """

    return self.forward(**interaction)

def generate_batch():
    pass


# class SynthesizerTrn(AbstractTalkingFace):
#   def __init__(self, config):
#     super().__init__(config)
#     self.model = FlowGenerator(
#         n_vocab=config.n_vocab, 
#         hidden_channels=config.hidden_channels, 
#         filter_channels=config.filter_channels, 
#         filter_channels_dp=config.filter_channels_dp, 
#         out_channels=config.out_channels,
#         kernel_size=config.kernel_size, 
#         n_heads=config.n_heads, 
#         n_layers_enc=config.n_layers_enc,
#         p_dropout=config.p_dropout, 
#         n_blocks_dec=config.n_blocks_dec, 
#         kernel_size_dec=config.kernel_size_dec, 
#         dilation_rate=config.dilation_rate, 
#         n_block_layers=config.n_block_layers,
#         p_dropout_dec=config.p_dropout_dec, 
#         n_speakers=config.n_speakers, 
#         gin_channels=config.gin_channels, 
#         n_split=config.n_split,
#         n_sqz=config.n_sqz,
#         sigmoid_scale=config.sigmoid_scale,
#         window_size=config.window_size,
#         block_length=config.block_length,
#         mean_only=config.mean_only,
#         hidden_channels_enc=config.hidden_channels_enc,
#         hidden_channels_dec=config.hidden_channels_dec,
#         prenet=config.prenet)

#   def forward(self, x, x_lengths, y=None, y_lengths=None, g=None, gen=False, noise_scale=1., length_scale=1.):
#     return self.model(x, x_lengths, y=y, y_lengths=y_lengths, g=g, gen=gen, noise_scale=noise_scale, length_scale=length_scale)

#   def infer(self, x, x_lengths, y=None, y_lengths=None, g=None, noise_scale=1., length_scale=1.):
#     return self.model(x, x_lengths, y=y, y_lengths=y_lengths, g=g, gen=True, noise_scale=noise_scale, length_scale=length_scale)

#   def store_inverse(self):
#     self.model.store_inverse()

#   def load(self, path):
#     checkpoint = torch.load(path, map_location='cpu')
#     self.model.load_state_dict(checkpoint['model'])
#     self.model.store_inverse()

#   def save(self, path):
#     torch.save({'model': self.model.state_dict()}, path)

#   def load_pretrained(self, path):
#     checkpoint = torch.load(path, map_location='cpu')
#     self.model.load_state_dict(checkpoint['model'])

#   def save_pretrained(self, path):
#     torch.save({'model': self.model.state_dict()}, path)

#   def load_gin(self, path):
#     checkpoint = torch.load(path, map_location='cpu')
#     self.model.load_state_dict(checkpoint['model'])
    
#   def save_gin(self, path):
#     torch.save({'model': self.model.state_dict()}, path)


class LayerNorm(nn.Module):
  def __init__(self, channels, eps=1e-4):
      super().__init__()
      self.channels = channels
      self.eps = eps

      self.gamma = nn.Parameter(torch.ones(channels))
      self.beta = nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    n_dims = len(x.shape)
    mean = torch.mean(x, 1, keepdim=True)
    variance = torch.mean((x -mean)**2, 1, keepdim=True)

    x = (x - mean) * torch.rsqrt(variance + self.eps)

    shape = [1, -1] + [1] * (n_dims - 2)
    x = x * self.gamma.view(*shape) + self.beta.view(*shape)
    return x

 
class ConvReluNorm(nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
    super().__init__()
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.p_dropout = p_dropout
    assert n_layers > 1, "Number of layers should be larger than 0."

    self.conv_layers = nn.ModuleList()
    self.norm_layers = nn.ModuleList()
    self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2))
    self.norm_layers.append(LayerNorm(hidden_channels))
    self.relu_drop = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(p_dropout))
    for _ in range(n_layers-1):
      self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))
      self.norm_layers.append(LayerNorm(hidden_channels))
    self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
    self.proj.weight.data.zero_()
    self.proj.bias.data.zero_()

  def forward(self, x, x_mask):
    x_org = x
    for i in range(self.n_layers):
      x = self.conv_layers[i](x * x_mask)
      x = self.norm_layers[i](x)
      x = self.relu_drop(x)
    x = x_org + self.proj(x)
    return x * x_mask


class WN(torch.nn.Module):
  def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
      super(WN, self).__init__()
      assert(kernel_size % 2 == 1)
      assert(hidden_channels % 2 == 0)
      self.in_channels = in_channels
      self.hidden_channels =hidden_channels
      self.kernel_size = kernel_size,
      self.dilation_rate = dilation_rate
      self.n_layers = n_layers
      self.gin_channels = gin_channels
      self.p_dropout = p_dropout

      self.in_layers = torch.nn.ModuleList()
      self.res_skip_layers = torch.nn.ModuleList()
      self.drop = nn.Dropout(p_dropout)

      if gin_channels != 0:
        cond_layer = torch.nn.Conv1d(gin_channels, 2*hidden_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

      for i in range(n_layers):
        dilation = dilation_rate ** i
        padding = int((kernel_size * dilation - dilation) / 2)
        in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                   dilation=dilation, padding=padding)
        in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
        self.in_layers.append(in_layer)

        # last one is not necessary
        if i < n_layers - 1:
            res_skip_channels = 2 * hidden_channels
        else:
            res_skip_channels = hidden_channels

        res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
        res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
        self.res_skip_layers.append(res_skip_layer)

  def forward(self, x, x_mask=None, g=None, **kwargs):
      output = torch.zeros_like(x)
      n_channels_tensor = torch.IntTensor([self.hidden_channels])

      if g is not None:
        g = self.cond_layer(g)

      for i in range(self.n_layers):
          x_in = self.in_layers[i](x)
          x_in = self.drop(x_in)
          if g is not None:
            cond_offset = i * 2 * self.hidden_channels
            g_l = g[:,cond_offset:cond_offset+2*self.hidden_channels,:]
          else:
            g_l = torch.zeros_like(x_in)

          acts = commons.fused_add_tanh_sigmoid_multiply(
              x_in,
              g_l,
              n_channels_tensor)

          res_skip_acts = self.res_skip_layers[i](acts)
          if i < self.n_layers - 1:
            x = (x + res_skip_acts[:,:self.hidden_channels,:]) * x_mask
            output = output + res_skip_acts[:,self.hidden_channels:,:]
          else:
            output = output + res_skip_acts
      return output * x_mask

  def remove_weight_norm(self):
    if self.gin_channels != 0:
      torch.nn.utils.remove_weight_norm(self.cond_layer)
    for l in self.in_layers:
      torch.nn.utils.remove_weight_norm(l)
    for l in self.res_skip_layers:
     torch.nn.utils.remove_weight_norm(l)


class ActNorm(nn.Module):
  def __init__(self, channels, ddi=False, **kwargs):
    super().__init__()
    self.channels = channels
    self.initialized = not ddi

    self.logs = nn.Parameter(torch.zeros(1, channels, 1))
    self.bias = nn.Parameter(torch.zeros(1, channels, 1))

  def forward(self, x, x_mask=None, reverse=False, **kwargs):
    if x_mask is None:
      x_mask = torch.ones(x.size(0), 1, x.size(2)).to(device=x.device, dtype=x.dtype)
    x_len = torch.sum(x_mask, [1, 2])
    if not self.initialized:
      self.initialize(x, x_mask)
      self.initialized = True

    if reverse:
      z = (x - self.bias) * torch.exp(-self.logs) * x_mask
      logdet = None
    else:
      z = (self.bias + torch.exp(self.logs) * x) * x_mask
      logdet = torch.sum(self.logs) * x_len # [b]

    return z, logdet

  def store_inverse(self):
    pass

  def set_ddi(self, ddi):
    self.initialized = not ddi

  def initialize(self, x, x_mask):
    with torch.no_grad():
      denom = torch.sum(x_mask, [0, 2])
      m = torch.sum(x * x_mask, [0, 2]) / denom
      m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
      v = m_sq - (m ** 2)
      logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

      bias_init = (-m * torch.exp(-logs)).view(*self.bias.shape).to(dtype=self.bias.dtype)
      logs_init = (-logs).view(*self.logs.shape).to(dtype=self.logs.dtype)

      self.bias.data.copy_(bias_init)
      self.logs.data.copy_(logs_init)


class InvConvNear(nn.Module):
  def __init__(self, channels, n_split=4, no_jacobian=False, **kwargs):
    super().__init__()
    assert(n_split % 2 == 0)
    self.channels = channels
    self.n_split = n_split
    self.no_jacobian = no_jacobian
    
    w_init = torch.qr(torch.FloatTensor(self.n_split, self.n_split).normal_())[0]
    if torch.det(w_init) < 0:
      w_init[:,0] = -1 * w_init[:,0]
    self.weight = nn.Parameter(w_init)

  def forward(self, x, x_mask=None, reverse=False, **kwargs):
    b, c, t = x.size()
    assert(c % self.n_split == 0)
    if x_mask is None:
      x_mask = 1
      x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
    else:
      x_len = torch.sum(x_mask, [1, 2])

    x = x.view(b, 2, c // self.n_split, self.n_split // 2, t)
    x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c // self.n_split, t)

    if reverse:
      if hasattr(self, "weight_inv"):
        weight = self.weight_inv
      else:
        weight = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
      logdet = None
    else:
      weight = self.weight
      if self.no_jacobian:
        logdet = 0
      else:
        logdet = torch.logdet(self.weight) * (c / self.n_split) * x_len # [b]

    weight = weight.view(self.n_split, self.n_split, 1, 1)
    z = F.conv2d(x, weight)

    z = z.view(b, 2, self.n_split // 2, c // self.n_split, t)
    z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
    return z, logdet

  def store_inverse(self):
    self.weight_inv = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)

        
class Encoder(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=None, block_length=None, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.window_size = window_size
    self.block_length = block_length

    self.drop = nn.Dropout(p_dropout)
    self.attn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_2 = nn.ModuleList()
    for i in range(self.n_layers):
      self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, window_size=window_size, p_dropout=p_dropout, block_length=block_length))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
      self.norm_layers_2.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask):
    attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    for i in range(self.n_layers):
      x = x * x_mask
      y = self.attn_layers[i](x, x, attn_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y)

      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_2[i](x + y)
    x = x * x_mask
    return x


class CouplingBlock(nn.Module):
  def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0, sigmoid_scale=False):
    super().__init__()
    self.in_channels = in_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.p_dropout = p_dropout
    self.sigmoid_scale = sigmoid_scale

    start = torch.nn.Conv1d(in_channels//2, hidden_channels, 1)
    start = torch.nn.utils.weight_norm(start)
    self.start = start
    # Initializing last layer to 0 makes the affine coupling layers
    # do nothing at first.  It helps to stabilze training.
    end = torch.nn.Conv1d(hidden_channels, in_channels, 1)
    end.weight.data.zero_()
    end.bias.data.zero_()
    self.end = end

    self.wn = modules.WN(in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, p_dropout)


  def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):
    b, c, t = x.size()
    if x_mask is None:
      x_mask = 1
    x_0, x_1 = x[:,:self.in_channels//2], x[:,self.in_channels//2:]

    x = self.start(x_0) * x_mask
    x = self.wn(x, x_mask, g)
    out = self.end(x)

    z_0 = x_0
    m = out[:, :self.in_channels//2, :]
    logs = out[:, self.in_channels//2:, :]
    if self.sigmoid_scale:
      logs = torch.log(1e-6 + torch.sigmoid(logs + 2))

    if reverse:
      z_1 = (x_1 - m) * torch.exp(-logs) * x_mask
      logdet = None
    else:
      z_1 = (m + torch.exp(logs) * x_1) * x_mask
      logdet = torch.sum(logs * x_mask, [1, 2])

    z = torch.cat([z_0, z_1], 1)
    return z, logdet

  def store_inverse(self):
    self.wn.remove_weight_norm()


class MultiHeadAttention(nn.Module):
  def __init__(self, channels, out_channels, n_heads, window_size=None, heads_share=True, p_dropout=0., block_length=None, proximal_bias=False, proximal_init=False):
    super().__init__()
    assert channels % n_heads == 0

    self.channels = channels
    self.out_channels = out_channels
    self.n_heads = n_heads
    self.window_size = window_size
    self.heads_share = heads_share
    self.block_length = block_length
    self.proximal_bias = proximal_bias
    self.p_dropout = p_dropout
    self.attn = None

    self.k_channels = channels // n_heads
    self.conv_q = nn.Conv1d(channels, channels, 1)
    self.conv_k = nn.Conv1d(channels, channels, 1)
    self.conv_v = nn.Conv1d(channels, channels, 1)
    if window_size is not None:
      n_heads_rel = 1 if heads_share else n_heads
      rel_stddev = self.k_channels**-0.5
      self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
      self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
    self.conv_o = nn.Conv1d(channels, out_channels, 1)
    self.drop = nn.Dropout(p_dropout)

    nn.init.xavier_uniform_(self.conv_q.weight)
    nn.init.xavier_uniform_(self.conv_k.weight)
    if proximal_init:
      self.conv_k.weight.data.copy_(self.conv_q.weight.data)
      self.conv_k.bias.data.copy_(self.conv_q.bias.data)
    nn.init.xavier_uniform_(self.conv_v.weight)
      
  def forward(self, x, c, attn_mask=None):
    q = self.conv_q(x)
    k = self.conv_k(c)
    v = self.conv_v(c)
    
    x, self.attn = self.attention(q, k, v, mask=attn_mask)

    x = self.conv_o(x)
    return x
    
  def attention(self, query, key, value, mask=None):
    # reshape [b, d, t] -> [b, n_h, t, d_k]
    b, d, t_s, t_t = (*key.size(), query.size(2))
    query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
    key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)
    if self.window_size is not None:
      assert t_s == t_t, "Relative attention is only available for self-attention."
      key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
      rel_logits = self._matmul_with_relative_keys(query, key_relative_embeddings)
      rel_logits = self._relative_position_to_absolute_position(rel_logits)
      scores_local = rel_logits / math.sqrt(self.k_channels)
      scores = scores + scores_local
    if self.proximal_bias:
      assert t_s == t_t, "Proximal bias is only available for self-attention."
      scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e4)
      if self.block_length is not None:
        block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
        scores = scores * block_mask + -1e4*(1 - block_mask)
    p_attn = F.softmax(scores, dim=-1) # [b, n_h, t_t, t_s]
    p_attn = self.drop(p_attn)
    output = torch.matmul(p_attn, value)
    if self.window_size is not None:
      relative_weights = self._absolute_position_to_relative_position(p_attn)
      value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
      output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
    output = output.transpose(2, 3).contiguous().view(b, d, t_t) # [b, n_h, t_t, d_k] -> [b, d, t_t]
    return output, p_attn

  def _matmul_with_relative_values(self, x, y):
    """
    x: [b, h, l, m]
    y: [h or 1, m, d]
    ret: [b, h, l, d]
    """
    ret = torch.matmul(x, y.unsqueeze(0))
    return ret

  def _matmul_with_relative_keys(self, x, y):
    """
    x: [b, h, l, d]
    y: [h or 1, m, d]
    ret: [b, h, l, m]
    """
    ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
    return ret

  def _get_relative_embeddings(self, relative_embeddings, length):
    max_relative_position = 2 * self.window_size + 1
    # Pad first before slice to avoid using cond ops.
    pad_length = max(length - (self.window_size + 1), 0)
    slice_start_position = max((self.window_size + 1) - length, 0)
    slice_end_position = slice_start_position + 2 * length - 1
    if pad_length > 0:
      padded_relative_embeddings = F.pad(
          relative_embeddings,
          commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
    else:
      padded_relative_embeddings = relative_embeddings
    used_relative_embeddings = padded_relative_embeddings[:,slice_start_position:slice_end_position]
    return used_relative_embeddings

  def _relative_position_to_absolute_position(self, x):
    """
    x: [b, h, l, 2*l-1]
    ret: [b, h, l, l]
    """
    batch, heads, length, _ = x.size()
    # Concat columns of pad to shift from relative to absolute indexing.
    x = F.pad(x, commons.convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))

    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    x_flat = x.view([batch, heads, length * 2 * length])
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0,0],[0,0],[0,length-1]]))

    # Reshape and slice out the padded elements.
    x_final = x_flat.view([batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
    return x_final

  def _absolute_position_to_relative_position(self, x):
    """
    x: [b, h, l, l]
    ret: [b, h, l, 2*l-1]
    """
    batch, heads, length, _ = x.size()
    # padd along column
    x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length-1]]))
    x_flat = x.view([batch, heads, length**2 + length*(length -1)])
    # add 0's in the beginning that will skew the elements after reshape
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
    x_final = x_flat.view([batch, heads, length, 2*length])[:,:,:,1:]
    return x_final

  def _attention_bias_proximal(self, length):
    """Bias for self-attention to encourage attention to close positions.
    Args:
      length: an integer scalar.
    Returns:
      a Tensor with shape [1, 1, length, length]
    """
    r = torch.arange(length, dtype=torch.float32)
    diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
    return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
  def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., activation=None):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.activation = activation

    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size//2)
    self.drop = nn.Dropout(p_dropout)

  def forward(self, x, x_mask):
    x = self.conv_1(x * x_mask)
    if self.activation == "gelu":
      x = x * torch.sigmoid(1.702 * x)
    else:
      x = torch.relu(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    return x * x_mask
  