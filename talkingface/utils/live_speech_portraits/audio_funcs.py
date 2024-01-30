import os
import os.path
import math
# import sox
# import pyworld as pw
import torch
import torch.utils.data
import numpy as np
import librosa

"""
useage
fft = Audio2Mel().cuda()
# audio shape is B x 1 x T, the normalized mel shape is B x D x T
mel = fft(audio)
"""
from librosa.filters import mel as librosa_mel_fn
import torch.nn.functional as F


class Audio2Mel(torch.nn.Module):
    def __init__(
            self,
            n_fft=512,
            hop_length=256,
            win_length=1024,
            sampling_rate=16000,
            n_mel_channels=80,
            mel_fmin=90,
            mel_fmax=7600.0,
    ):
        super(Audio2Mel, self).__init__()
        ##############################################
        # FFT Parameters                             #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.min_mel = math.log(1e-5)
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

    """
    input audio signal (-1,1): B x 1 x T
    output mel signal: B x D x T', T' is a reduction of T
    """

    def forward(self, audio, normalize=True):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=False
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=1e-5))

        # normalize to the range [0,1]
        if normalize:
            log_mel_spec = (log_mel_spec - self.min_mel) / -self.min_mel
        return log_mel_spec

    def mel_to_audio(self, mel):
        mel = torch.exp(mel * (-self.min_mel) + self.min_mel) ** 2
        mel_np = mel.cpu().numpy()
        audio = librosa.feature.inverse.mel_to_audio(mel_np, sr=self.sampling_rate, n_fft=self.n_fft,
                                                     hop_length=self.hop_length, win_length=self.win_length,
                                                     window='hann', center=False,
                                                     pad_mode='reflect', power=2.0, n_iter=32, fmin=self.mel_fmin,
                                                     fmax=self.mel_fmax)
        return audio

    """
    here we will get per frame energy to replace mc0 in the corresponding prosody representation
    the audio is already in the gpu card for accerelate the computation speed
    input audio signal: B x 1 x T
    output energy: B x 1 x T'
    """

    def get_energy(self, audio, normalize=True):
        # B x 1 x T
        p = (self.n_fft - self.hop_length) // 2
        audio_new = F.pad(audio, (p, p), "reflect").squeeze(1)
        # audio_new = audio.squeeze(1)
        audio_fold = audio_new.unfold(1, self.win_length, self.hop_length)
        audio_energy = torch.sqrt(torch.mean(audio_fold ** 2, dim=-1))
        audio_energy = torch.log(torch.clamp(audio_energy, min=1e-5))
        if normalize:
            audio_energy = (audio_energy - self.min_mel) / -self.min_mel
        return audio_energy

    # we can get the energy of mels here, B*D*T
    def get_energy_mel(self, mels, normalize=True):
        m = mels.exp().mean(dim=1)
        audio_energy = torch.log(m)
        # audio_energy = torch.log(torch.clamp(m,min=1e-5))
        # if normalize:
        #     audio_energy = (audio_energy - self.min_mel) / -self.min_mel
        return audio_energy


def mu_law_encoding(data, mu=255):
    '''encode the original audio via mu-law companding and mu-bits quantization
    '''
    # mu-law companding
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    # mu-bits quantization from [-1, 1] to [0, mu]
    mu_x = (mu_x + 1) / 2 * mu + 0.5
    return mu_x.astype(np.int32)


# %timeit mu_x = mu_law_encoding(x, 255)  305 µs ± 554 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each)


def mu_law_decoding(data, mu=255):
    '''inverse the mu-law compressed and quantized data.
    '''
    # dequantization
    y = 2 * (data.astype(np.float32) / mu) - 1
    # inverse mu-law companding
    x = np.sign(y) * (1.0 / mu) * ((1.0 + mu) ** abs(y) - 1.0)
    return x


## audio augmentation
def inject_gaussian_noise(data, noise_factor, use_torch=False):
    ''' inject random gaussian noise (mean=0, std=1) to audio clip
    In my test, a reasonable factor region could be [0, 0.01]
    larger will be too large and smaller could be ignored.
    Args:
        data: [n,] original audio sequence
        noise_factor(float): scaled factor
        use_torch(bool): optional, if use_torch=True, input data and implementation will
            be torch methods.
    Returns:
        augmented_data: [n,] noised audio clip

    '''
    if use_torch == False:
        augmented_data = data + noise_factor * np.random.normal(0, 1, len(data))
        # Cast back to same data type
        augmented_data = augmented_data.astype(type(data[0]))
    # use torch
    else:
        augmented_data = data + noise_factor * torch.randn(1).cuda()

    return augmented_data


# pitch shifting
def pitch_shifting(data, sampling_rate=48000, factor=5):
    ''' shift the audio pitch.
    '''
    # Permissible factor values = -5 <= x <= 5
    pitch_factor = np.random.rand(1) * 2 * factor - factor
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


def speed_change(data, landmark=None):
    ''' change the speed of input audio. Note that we return the speed_rate to
    change the speed of landmarks or videos.
    Args:
        data: [n,] audio clip
        landmark: [m, pts, 2] aligned landmarks with audio if existed.
    '''
    # Permissible factor values = 0.7 <= x <= 1.3 (higher is faster)
    # resulted audio length: np.round(n/rate)
    speed_rate = np.random.uniform(0.7, 1.3)
    # only augment audio
    if landmark == None:
        return librosa.effects.time_stretch(data, speed_rate), speed_rate
    else:
        #        n_after = np.round(data.shape[0]/speed_rate)
        pass


def prepare_noises(scp_file, root=None, sampline_rate=None, ignore_class=None):
    noises = []
    print('Loading augmentation noises...')
    with open(scp_file, 'r') as fp:
        for line in fp.readlines():
            line = line.rstrip('\n')
            if ignore_class is not None and ignore_class in line:
                continue

            noise, sr = librosa.load(os.path.join(root, line), sr=sampline_rate)
            noises.append(noise)
    print('Augmentation noises loaded!')
    return noises, sr


def add_gauss_noise(wav, noise_std=0.03, max_wav_value=1.0):
    if isinstance(wav, np.ndarray):
        wav = torch.tensor(wav.copy())

    real_std = np.random.random() * noise_std
    wav_new = wav.float() / max_wav_value + torch.randn(wav.size()) * real_std
    wav_new = wav_new * max_wav_value
    wav_new = wav_new.clamp_(-max_wav_value, max_wav_value)

    return wav_new.float().numpy()


def add_background_noise(wav, noises, min_snr=2, max_snr=15):
    def mix_noise(wav, noise, scale):
        x = wav + scale * noise
        x = x.clip(-1, 1)
        return x

    def voice_energy(wav):
        wav_float = np.copy(wav)
        return np.sum(wav_float ** 2) / (wav_float.shape[0] + 1e-5)

    def voice_energy_ratio(wav, noise, target_snr):
        wav_eng = voice_energy(wav)
        noise_eng = voice_energy(noise)
        target_noise_eng = wav_eng / (10 ** (target_snr / 10.0))
        ratio = target_noise_eng / (noise_eng + 1e-5)
        return ratio

    total_id = len(noises)
    # 0 is no need to generate the noise
    idx = np.random.choice(range(0, total_id))
    noise_wav = noises[idx]
    if noise_wav.shape[0] > wav.shape[0]:
        sel_range_id = np.random.choice(range(0, noise_wav.shape[0] - wav.shape[0]))
        n = noise_wav[sel_range_id:sel_range_id + wav.shape[0]]
    else:
        n = np.zeros(wav.shape[0])
        sel_range_id = np.random.choice(range(0, wav.shape[0] - noise_wav.shape[0] + 1))
        n[sel_range_id:sel_range_id + noise_wav.shape[0]] = noise_wav
    #
    target_snr = np.random.random() * (max_snr - min_snr) + min_snr
    scale = voice_energy_ratio(wav, n, target_snr)
    wav_new = mix_noise(wav, n, scale)
    return wav_new


def noise_augment(wav, wav_noises, gaussian_prob=0.5):
    if np.random.random() > gaussian_prob:  # add gauss noise
        noise_std = np.random.uniform(low=0.001, high=0.02)
        aug_wave_data = add_gauss_noise(wav, noise_std=noise_std)
    else:  # add background noise
        aug_wave_data = add_background_noise(wav, wav_noises, min_snr=2, max_snr=15)

    return aug_wave_data







