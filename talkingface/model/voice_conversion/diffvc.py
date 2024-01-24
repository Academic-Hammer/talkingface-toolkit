from talkingface.model.abstract_talkingface import AbstractTalkingFace
import torch
import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import torch
import torchaudio
import numpy as np
from librosa.filters import mel as librosa_mel_fn
from talkingface.utils.utils import sequence_mask, fix_len_compatibility, mse_loss,convert_pad_shape


from talkingface.utils.voice_conversion_talkingface.params_model import *
from talkingface.utils.voice_conversion_talkingface.params_data import *
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq
from torch import nn
import numpy as np
import torch


class SpeakerEncoder(nn.Module):
    def __init__(self, device, loss_device):
        super().__init__()
        self.loss_device = loss_device

        # Network defition
        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=model_hidden_size,
                            num_layers=model_num_layers,
                            batch_first=True).to(device)
        self.linear = nn.Linear(in_features=model_hidden_size,
                                out_features=model_embedding_size).to(device)
        self.relu = torch.nn.ReLU().to(device)

        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(loss_device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(loss_device)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)

    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01

        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)

    def forward(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.

        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape
        (batch_size, n_frames, n_channels)
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers,
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
        # and the final cell state.
        out, (hidden, cell) = self.lstm(utterances, hidden_init)

        # We take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))

        # L2-normalize it
        embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

        return embeds

    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, embedding_size)
        :return: the similarity matrix as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, speakers_per_batch)
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / torch.norm(centroids_incl, dim=2, keepdim=True)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / torch.norm(centroids_excl, dim=2, keepdim=True)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(self.loss_device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)

        ## Even more vectorized version (slower maybe because of transpose)
        # sim_matrix2 = torch.zeros(speakers_per_batch, speakers_per_batch, utterances_per_speaker
        #                           ).to(self.loss_device)
        # eye = np.eye(speakers_per_batch, dtype=np.int)
        # mask = np.where(1 - eye)
        # sim_matrix2[mask] = (embeds[mask[0]] * centroids_incl[mask[1]]).sum(dim=2)
        # mask = np.where(eye)
        # sim_matrix2[mask] = (embeds * centroids_excl).sum(dim=2)
        # sim_matrix2 = sim_matrix2.transpose(1, 2)

        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix

    def loss(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]

        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker,
                                         speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        loss = self.loss_fn(sim_matrix, target)

        # EER (not backpropagated)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # Snippet from https://yangcha.github.io/EER-ROC/
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        return loss, eer

#base
class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params


    def relocate_input(self, x: list):
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x
#modules
class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim_out, 3, 
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                               dim_out))

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)            

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                            heads = self.heads, qkv=3)            
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', 
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = 1000.0 * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RefBlock(BaseModule):
    def __init__(self, out_dim, time_emb_dim):
        super(RefBlock, self).__init__()
        base_dim = out_dim // 4
        self.mlp1 = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                                base_dim))
        self.mlp2 = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                                2 * base_dim))
        self.block11 = torch.nn.Sequential(torch.nn.Conv2d(1, 2 * base_dim, 
                      3, 1, 1), torch.nn.InstanceNorm2d(2 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.block12 = torch.nn.Sequential(torch.nn.Conv2d(base_dim, 2 * base_dim, 
                      3, 1, 1), torch.nn.InstanceNorm2d(2 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.block21 = torch.nn.Sequential(torch.nn.Conv2d(base_dim, 4 * base_dim,
                      3, 1, 1), torch.nn.InstanceNorm2d(4 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.block22 = torch.nn.Sequential(torch.nn.Conv2d(2 * base_dim, 4 * base_dim,
                      3, 1, 1), torch.nn.InstanceNorm2d(4 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.block31 = torch.nn.Sequential(torch.nn.Conv2d(2 * base_dim, 8 * base_dim,
                      3, 1, 1), torch.nn.InstanceNorm2d(8 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.block32 = torch.nn.Sequential(torch.nn.Conv2d(4 * base_dim, 8 * base_dim,
                      3, 1, 1), torch.nn.InstanceNorm2d(8 * base_dim, affine=True),
                      torch.nn.GLU(dim=1))
        self.final_conv = torch.nn.Conv2d(4 * base_dim, out_dim, 1)

    def forward(self, x, mask, time_emb):
        y = self.block11(x * mask)
        y = self.block12(y * mask)
        y += self.mlp1(time_emb).unsqueeze(-1).unsqueeze(-1)
        y = self.block21(y * mask)
        y = self.block22(y * mask)
        y += self.mlp2(time_emb).unsqueeze(-1).unsqueeze(-1)
        y = self.block31(y * mask)
        y = self.block32(y * mask)
        y = self.final_conv(y * mask)
        return (y * mask).sum((2, 3)) / (mask.sum((2, 3)) * x.shape[2])

#diffusion
class GradLogPEstimator(BaseModule):
    def __init__(self, dim_base, dim_cond, use_ref_t, dim_mults=(1, 2, 4)):
        super(GradLogPEstimator, self).__init__()
        self.use_ref_t = use_ref_t
        dims = [2 + dim_cond, *map(lambda m: dim_base * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(dim_base)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim_base, dim_base * 4), 
                               Mish(), torch.nn.Linear(dim_base * 4, dim_base))

        cond_total = dim_base + 256
        if use_ref_t:
            self.ref_block = RefBlock(out_dim=dim_cond, time_emb_dim=dim_base)
            cond_total += dim_cond
        self.cond_block = torch.nn.Sequential(torch.nn.Linear(cond_total, 4 * dim_cond),
                                      Mish(), torch.nn.Linear(4 * dim_cond, dim_cond))

        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim_in, dim_out,time_emb_dim=dim_base),
                       ResnetBlock(dim_out, dim_out, time_emb_dim=dim_base),
                       Residual(Rezero(LinearAttention(dim_out))),
                       Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                     ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim_base),
                     ResnetBlock(dim_in, dim_in, time_emb_dim=dim_base),
                     Residual(Rezero(LinearAttention(dim_in))),
                     Upsample(dim_in)]))
        self.final_block = Block(dim_base, dim_base)
        self.final_conv = torch.nn.Conv2d(dim_base, 1, 1)

    def forward(self, x, x_mask, mean, ref, ref_mask, c, t):
        condition = self.time_pos_emb(t)
        t = self.mlp(condition)

        x = torch.stack([mean, x], 1)
        x_mask = x_mask.unsqueeze(1)
        ref_mask = ref_mask.unsqueeze(1)

        if self.use_ref_t:
            condition = torch.cat([condition, self.ref_block(ref, ref_mask, t)], 1)
        condition = torch.cat([condition, c], 1)

        condition = self.cond_block(condition).unsqueeze(-1).unsqueeze(-1)
        condition = torch.cat(x.shape[2]*[condition], 2)
        condition = torch.cat(x.shape[3]*[condition], 3)
        x = torch.cat([x, condition], 1)

        hiddens = []
        masks = [x_mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, x_mask)
        output = self.final_conv(x * x_mask)

        return (output * x_mask).squeeze(1)


class Diffusion(BaseModule):
    def __init__(self, n_feats, dim_unet, dim_spk, use_ref_t, beta_min, beta_max):
        super(Diffusion, self).__init__()
        self.estimator = GradLogPEstimator(dim_unet, dim_spk, use_ref_t)
        self.n_feats = n_feats
        self.dim_unet = dim_unet
        self.dim_spk = dim_spk
        self.use_ref_t = use_ref_t
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_beta(self, t):
        beta = self.beta_min + (self.beta_max - self.beta_min) * t
        return beta

    def get_gamma(self, s, t, p=1.0, use_torch=False):
        beta_integral = self.beta_min + 0.5*(self.beta_max - self.beta_min)*(t + s)
        beta_integral *= (t - s)
        if use_torch:
            gamma = torch.exp(-0.5*p*beta_integral).unsqueeze(-1).unsqueeze(-1)
        else:
            gamma = math.exp(-0.5*p*beta_integral)
        return gamma

    def get_mu(self, s, t):
        a = self.get_gamma(s, t)
        b = 1.0 - self.get_gamma(0, s, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return a * b / c

    def get_nu(self, s, t):
        a = self.get_gamma(0, s)
        b = 1.0 - self.get_gamma(s, t, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return a * b / c

    def get_sigma(self, s, t):
        a = 1.0 - self.get_gamma(0, s, p=2.0)
        b = 1.0 - self.get_gamma(s, t, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return math.sqrt(a * b / c)

    def compute_diffused_mean(self, x0, mask, mean, t, use_torch=False):
        x0_weight = self.get_gamma(0, t, use_torch=use_torch)
        mean_weight = 1.0 - x0_weight
        xt_mean = x0 * x0_weight + mean * mean_weight
        return xt_mean * mask

    def forward_diffusion(self, x0, mask, mean, t):
        xt_mean = self.compute_diffused_mean(x0, mask, mean, t, use_torch=True)
        variance = 1.0 - self.get_gamma(0, t, p=2.0, use_torch=True)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt = xt_mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mean, ref, ref_mask, mean_ref, c, 
                          n_timesteps, mode):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = 1.0 - i*h
            time = t * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            beta_t = self.get_beta(t)
            xt_ref = [self.compute_diffused_mean(ref, ref_mask, mean_ref, t)]
#            for j in range(15):
#                xt_ref += [self.compute_diffused_mean(ref, ref_mask, mean_ref, (j+0.5)/15.0)]
            xt_ref = torch.stack(xt_ref, 1)
            if mode == 'pf':
                dxt = 0.5 * (mean - xt - self.estimator(xt, mask, mean, xt_ref, ref_mask, c, time)) * (beta_t * h)
            else:
                if mode == 'ml':
                    kappa = self.get_gamma(0, t - h) * (1.0 - self.get_gamma(t - h, t, p=2.0))
                    kappa /= (self.get_gamma(0, t) * beta_t * h)
                    kappa -= 1.0
                    omega = self.get_nu(t - h, t) / self.get_gamma(0, t)
                    omega += self.get_mu(t - h, t)
                    omega -= (0.5 * beta_t * h + 1.0)
                    sigma = self.get_sigma(t - h, t)
                else:
                    kappa = 0.0
                    omega = 0.0
                    sigma = math.sqrt(beta_t * h)
                dxt = (mean - xt) * (0.5 * beta_t * h + omega)
                dxt -= self.estimator(xt, mask, mean, xt_ref, ref_mask, c, time) * (1.0 + kappa) * (beta_t * h)
                dxt += torch.randn_like(z, device=z.device) * sigma
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(self, z, mask, mean, ref, ref_mask, mean_ref, c, 
                n_timesteps, mode):
        if mode not in ['pf', 'em', 'ml']:
            print('Inference mode must be one of [pf, em, ml]!')
            return z
        return self.reverse_diffusion(z, mask, mean, ref, ref_mask, mean_ref, c, 
                                      n_timesteps, mode)

    def loss_t(self, x0, mask, mean, x_ref, mean_ref, c, t):
        xt, z = self.forward_diffusion(x0, mask, mean, t)
        xt_ref = [self.compute_diffused_mean(x_ref, mask, mean_ref, t, use_torch=True)]
#        for j in range(15):
#            xt_ref += [self.compute_diffused_mean(x_ref, mask, mean_ref, (j+0.5)/15.0)]
        xt_ref = torch.stack(xt_ref, 1)
        z_estimation = self.estimator(xt, mask, mean, xt_ref, mask, c, t)
        z_estimation *= torch.sqrt(1.0 - self.get_gamma(0, t, p=2.0, use_torch=True))
        loss = torch.sum((z_estimation + z)**2) / (torch.sum(mask)*self.n_feats)
        return loss

    def compute_loss(self, x0, mask, mean, x_ref, mean_ref, c, offset=1e-5):
        b = x0.shape[0]
        t = torch.rand(b, dtype=x0.dtype, device=x0.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, mean, x_ref, mean_ref, c, t)
#encoder
class LayerNorm(BaseModule):
    def __init__(self, channels, eps=1e-4):
        super(LayerNorm, self).__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean)**2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ConvReluNorm(BaseModule):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, 
                 n_layers, p_dropout):
        super(ConvReluNorm, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.conv_layers.append(torch.nn.Conv1d(in_channels, hidden_channels, 
                                                kernel_size, padding=kernel_size//2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(torch.nn.Conv1d(hidden_channels, hidden_channels, 
                                                    kernel_size, padding=kernel_size//2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
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


class MultiHeadAttention(BaseModule):
    def __init__(self, channels, out_channels, n_heads, window_size=None, 
                 heads_share=True, p_dropout=0.0, proximal_bias=False, 
                 proximal_init=False):
        super(MultiHeadAttention, self).__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.heads_share = heads_share
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = torch.nn.Parameter(torch.randn(n_heads_rel, 
                             window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = torch.nn.Parameter(torch.randn(n_heads_rel, 
                             window_size * 2 + 1, self.k_channels) * rel_stddev)
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)
        
    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        
        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
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
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, 
                                                                    dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, 
                                                                value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = torch.nn.functional.pad(
                            relative_embeddings, convert_pad_shape([[0, 0], 
                            [pad_length, pad_length], [0, 0]]))
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:,
                                   slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        x = torch.nn.functional.pad(x, convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = torch.nn.functional.pad(x_flat, convert_pad_shape([[0,0],[0,0],[0,length-1]]))
        x_final = x_flat.view([batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        x = torch.nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length-1]]))
        x_flat = x.view([batch, heads, length**2 + length*(length - 1)])
        x_flat = torch.nn.functional.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2*length])[:,:,:,1:]
        return x_final

    def _attention_bias_proximal(self, length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(BaseModule):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, 
                 p_dropout=0.0):
        super(FFN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, 
                                      padding=kernel_size//2)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size, 
                                      padding=kernel_size//2)
        self.drop = torch.nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class Encoder(BaseModule):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, 
                 kernel_size=1, p_dropout=0.0, window_size=None, **kwargs):
        super(Encoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, 
                                    n_heads, window_size=window_size, p_dropout=p_dropout))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, 
                                       filter_channels, kernel_size, p_dropout=p_dropout))
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


class MelEncoder(BaseModule):
    def __init__(self, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size=None):
        super(MelEncoder, self).__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        print(self.channels)
        print(self.n_feats)
        self.init_proj = torch.nn.Conv1d(n_feats, channels, 1)
        self.prenet = ConvReluNorm(channels, channels, channels, 
                                   kernel_size=5, n_layers=3, p_dropout=0.5)

        self.encoder = Encoder(channels, filters, heads, layers, kernel, 
                               dropout, window_size=window_size)

        self.term_proj = torch.nn.Conv1d(channels, n_feats, 1)

    def forward(self, x, x_mask):
        x = self.init_proj(x * x_mask)
        x = self.prenet(x, x_mask)
        x = self.encoder(x, x_mask)
        x = self.term_proj(x * x_mask)
        return x
    
#layer
class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

#postnet
class Block1(BaseModule):
    def __init__(self, dim, groups=8):
        super(Block1, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv2d(dim, dim, 7, 
                     padding=3), torch.nn.GroupNorm(groups, dim), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1(BaseModule):
    def __init__(self, dim, groups=8):
        super(ResnetBlock1, self).__init__()
        self.block1 = Block1(dim)
        self.block2 = Block1(dim)
        self.res = torch.nn.Conv2d(dim, dim, 1)

    def forward(self, x, mask):
        h = self.block1(x, mask)
        h = self.block2(h, mask)
        output = self.res(x * mask) + h
        return output


class PostNet(BaseModule):
    def __init__(self, dim, groups=8):
        super(PostNet, self).__init__()
        self.init_conv = torch.nn.Conv2d(1, dim, 1)
        self.res_block = ResnetBlock1(dim, groups=groups)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask):
        x = x.unsqueeze(1)
        mask = mask.unsqueeze(1)
        x = self.init_conv(x * mask)
        x = self.res_block(x, mask)
        output = self.final_conv(x * mask)
        return output.squeeze(1)

#utils
class PseudoInversion(BaseModule):
    def __init__(self, n_mels, sampling_rate, n_fft):
        super(PseudoInversion, self).__init__()
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        mel_basis = librosa_mel_fn(sampling_rate, n_fft, n_mels, 0, 8000)
        mel_basis_inverse = np.linalg.pinv(mel_basis)
        mel_basis_inverse = torch.from_numpy(mel_basis_inverse).float()
        self.register_buffer("mel_basis_inverse", mel_basis_inverse)

    def forward(self, log_mel_spectrogram):
        mel_spectrogram = torch.exp(log_mel_spectrogram)
        stftm = torch.matmul(self.mel_basis_inverse, mel_spectrogram)
        return stftm


class InitialReconstruction(BaseModule):
    def __init__(self, n_fft, hop_size):
        super(InitialReconstruction, self).__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        window = torch.hann_window(n_fft).float()
        self.register_buffer("window", window)

    def forward(self, stftm):
        real_part = torch.ones_like(stftm, device=stftm.device)
        imag_part = torch.zeros_like(stftm, device=stftm.device)
        #stft = torch.stack([real_part, imag_part], -1)*stftm.unsqueeze(-1)
        
        stft_complex = torch.complex(real_part * stftm, imag_part * stftm)
        #istft = torchaudio.functional.istft(stft, n_fft=self.n_fft,
                           #hop_length=self.hop_size, win_length=self.n_fft, 
                          # window=self.window, center=True)
        print("0stft.shape:", stft_complex.shape)
        istft = torch.istft(stft_complex, n_fft=self.n_fft, hop_length=self.hop_size, win_length=self.n_fft, 
                           window=self.window,center=True )
        return istft.unsqueeze(1)


# Fast Griffin-Lim algorithm as a PyTorch module
class FastGL(BaseModule):
    def __init__(self, n_mels, sampling_rate, n_fft, hop_size, momentum=0.99):
        super(FastGL, self).__init__()
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.momentum = momentum
        self.pi = PseudoInversion(n_mels, sampling_rate, n_fft)
        self.ir = InitialReconstruction(n_fft, hop_size)
        window = torch.hann_window(n_fft).float()
        self.register_buffer("window", window)

    @torch.no_grad()
    def forward(self, s, n_iters=32):
        c = self.pi(s)
        x = self.ir(c)
        x = x.squeeze(1)
        c = c.unsqueeze(-1)
        prev_angles = torch.zeros_like(c, device=c.device)
        for _ in range(n_iters):        
            s = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_size, 
                           win_length=self.n_fft, window=self.window, 
                           center=True, return_complex=True)
            #real_part, imag_part = s.unbind(-1)
            real_part = s.real
            imag_part = s.imag
            stftm = torch.sqrt(torch.clamp(real_part**2 + imag_part**2, min=1e-8))
            print(s.shape)
            print(stftm.shape)
            stftm_complex = torch.complex(stftm, torch.zeros_like(stftm))
            #angles = s / stftm.unsqueeze(-1)
            angles = s / stftm_complex
            angles=angles.unsqueeze(-1)
            print("1s.shape:", s.shape)
            print("c.shape:", c.shape)
            print("prev_angles.shape:", prev_angles.shape)
            s = c * (angles + self.momentum * (angles - prev_angles))
           # real_part = torch.ones_like(s, device=stftm.device)
           # imag_part = torch.zeros_like(s, device=stftm.device)
           # s_complex = torch.complex(real_part * s, imag_part * s)
            #x = torchaudio.functional.istft(s, n_fft=self.n_fft, hop_length=self.hop_size, 
                                            #win_length=self.n_fft, window=self.window, 
                                            #center=True)
           # print("2s.shape:", s_complex.shape)
            s = s.squeeze(-1)  
            print("3s.shape:", s.shape)
            x = torch.istft(s, n_fft=self.n_fft, hop_length=self.hop_size, 
                                            win_length=self.n_fft, window=self.window, 
                                            center=True)
            prev_angles = angles
        return x.unsqueeze(1)
    
#vc
class FwdDiffusion(BaseModule):
    def __init__(self, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size, dim):
        super(FwdDiffusion, self).__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        self.dim = dim
        print(self.channels)
        self.encoder = MelEncoder(n_feats, channels, filters, heads, layers, 
                                  kernel, dropout, window_size)
        self.postnet = PostNet(dim)
    
    def nparams(self):
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params


    def relocate_input(self, x: list):
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x

    @torch.no_grad()
    def forward(self, x, mask):
        x, mask = self.relocate_input([x, mask])
        z = self.encoder(x, mask)
        z_output = self.postnet(z, mask)
        return z_output

    def compute_loss(self, x, y, mask):
        x, y, mask = self.relocate_input([x, y, mask])
        z = self.encoder(x, mask)
        z_output = self.postnet(z, mask)
        loss = mse_loss(z_output, y, mask, self.n_feats)
        return loss
    

class diffvc(AbstractTalkingFace):
    def __init__(self, config):
            super(diffvc, self).__init__()
            
            self.n_feats=config["n_feats"]
            self.channels= config["channels"]
            self.filters=config["filters"]
            self.heads=config["heads"]
            self.layers=config["layers"]
            self.kernel=config["kernel"]
            self.dropout=config["dropout"]
            self.window_size=config["window_size"]
            self.enc_dim=config["enc_dim"]
            self.spk_dim=config["spk_dim"]
            self.use_ref_t=config["use_ref_t"]
            self.dec_dim=config["dec_dim"]
            self.beta_min=config["beta_min"]
            self.beta_max=config["beta_max"]
            print(config["n_feats"])
            self.encoder = FwdDiffusion(config["n_feats"], config["channels"], config["filters"],
                                config["heads"],config["layers"],config["kernel"],
                                config["dropout"],config["window_size"],config["enc_dim"])
            self.decoder = Diffusion(config["n_feats"], config["dec_dim"],config["spk_dim"],config["use_ref_t"], 
                                    config["beta_min"],config["beta_max"])
    
    def nparams(self):
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params


    def relocate_input(self, x: list):
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x

    def load_encoder(self, enc_path):
        enc_dict = torch.load(enc_path, map_location=lambda loc, storage: loc)
        self.encoder.load_state_dict(enc_dict, strict=False)

    @torch.no_grad()
    def forward(self, x, x_lengths, x_ref, x_ref_lengths, c, n_timesteps, 
                mode='ml'):
        """
        Generates mel-spectrogram from source mel-spectrogram conditioned on
        target speaker embedding. Returns:
            1. 'average voice' encoder outputs
            2. decoder outputs
        
        Args:
            x (torch.Tensor): batch of source mel-spectrograms.
            x_lengths (torch.Tensor): numbers of frames in source mel-spectrograms.
            x_ref (torch.Tensor): batch of reference mel-spectrograms.
            x_ref_lengths (torch.Tensor): numbers of frames in reference mel-spectrograms.
            c (torch.Tensor): batch of reference speaker embeddings
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            mode (string, optional): sampling method. Can be one of:
              'pf' - probability flow sampling (Euler scheme for ODE)
              'em' - Euler-Maruyama SDE solver
              'ml' - Maximum Likelihood SDE solver
        """
        x, x_lengths = self.relocate_input([x, x_lengths])
        x_ref, x_ref_lengths, c = self.relocate_input([x_ref, x_ref_lengths, c])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        x_ref_mask = sequence_mask(x_ref_lengths).unsqueeze(1).to(x_ref.dtype)
        mean = self.encoder(x, x_mask)
        mean_x = self.decoder.compute_diffused_mean(x, x_mask, mean, 1.0)
        mean_ref = self.encoder(x_ref, x_ref_mask)

        b = x.shape[0]
        max_length = int(x_lengths.max())
        max_length_new = fix_len_compatibility(max_length)
        x_mask_new = sequence_mask(x_lengths, max_length_new).unsqueeze(1).to(x.dtype)
        mean_new = torch.zeros((b, self.n_feats, max_length_new), dtype=x.dtype, 
                                device=x.device)
        mean_x_new = torch.zeros((b, self.n_feats, max_length_new), dtype=x.dtype, 
                                  device=x.device)
        for i in range(b):
            mean_new[i, :, :x_lengths[i]] = mean[i, :, :x_lengths[i]]
            mean_x_new[i, :, :x_lengths[i]] = mean_x[i, :, :x_lengths[i]]

        z = mean_x_new
        z += torch.randn_like(mean_x_new, device=mean_x_new.device)

        y = self.decoder(z, x_mask_new, mean_new, x_ref, x_ref_mask, mean_ref, c, 
                         n_timesteps, mode)
        return mean_x, y[:, :, :max_length]

    def compute_loss(self, x, x_lengths, x_ref, c):
        """
        Computes diffusion (score matching) loss.
            
        Args:
            x (torch.Tensor): batch of source mel-spectrograms.
            x_lengths (torch.Tensor): numbers of frames in source mel-spectrograms.
            x_ref (torch.Tensor): batch of reference mel-spectrograms.
            c (torch.Tensor): batch of reference speaker embeddings
        """
        x, x_lengths, x_ref, c = self.relocate_input([x, x_lengths, x_ref, c])
        x_mask = sequence_mask(x_lengths).unsqueeze(1).to(x.dtype)
        mean = self.encoder(x, x_mask).detach()
        mean_ref = self.encoder(x_ref, x_mask).detach()
        diff_loss = self.decoder.compute_loss(x, x_mask, mean, x_ref, mean_ref, c)
        return diff_loss
    

    def calculate_loss(self, interaction):
            x, x_lengths, x_ref, c = interaction
            return {"loss": self.compute_loss(x, x_lengths, x_ref, c)}

    def predict(self, interaction):
            x, x_lengths, x_ref, x_ref_lengths, c, n_timesteps, mode = interaction
            return self.forward(x, x_lengths, x_ref, x_ref_lengths, c, n_timesteps, mode)

    def generate_batch(self):
            # Implement this method based on your requirements
            pass


    

