
import warnings
import librosa
import numpy as np
import soundfile as sf



class StarganAudio:
    def __init__(self, config):
        # 初始化配置参数。
        self.kwargs = config

    def logmelfilterbank(self, audio,
                     sampling_rate,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     num_mels=80,
                     fmin=None,
                     fmax=None,
                     eps=1e-10,
                     ):
        """Compute log-Mel filterbank feature.
        Args:
            audio (ndarray): Audio signal (T,).
            sampling_rate (int): Sampling rate.
            fft_size (int): FFT size.
            hop_size (int): Hop size.
            win_length (int): Window length. If set to None, it will be the same as fft_size.
            window (str): Window function type.
            num_mels (int): Number of mel basis.
            fmin (int): Minimum frequency in mel basis calculation.
            fmax (int): Maximum frequency in mel basis calculation.
            eps (float): Epsilon value to avoid inf in log calculation.
        Returns:
            ndarray: Log Mel filterbank feature (#frames, num_mels).
        """
        # 获取振幅频谱
        x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                            win_length=win_length, window=window, pad_mode="reflect")
        spc = np.abs(x_stft).T  # (#frames, #bins)

        # 获取梅尔基数
        fmin = 0 if fmin is None else fmin
        fmax = sampling_rate / 2 if fmax is None else fmax
        mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)

        return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

    def extract_melspec(self, src_filepath):
        try:
            warnings.filterwarnings('ignore')

            # 从配置中提取参数。
            trim_silence = self.kwargs['trim_silence']
            top_db = self.kwargs['top_db']
            flen = self.kwargs['flen']
            fshift = self.kwargs['fshift']
            fmin = self.kwargs['fmin']
            fmax = self.kwargs['fmax']
            num_mels = self.kwargs['num_mels']
            fs = self.kwargs['fs']
            
            # 读取音频文件。
            audio, fs_ = sf.read(src_filepath)
            if trim_silence:
                # 如果需要，剪切静音部分。
                audio, _ = librosa.effects.trim(audio, top_db=top_db, frame_length=2048, hop_length=512)
            if fs != fs_:
                # 如果需要，进行重采样。
                audio = librosa.resample(audio, fs_, fs)
            # 提取梅尔频谱。
            melspec_raw = self.logmelfilterbank(audio,fs, fft_size=flen,hop_size=fshift,
                                            fmin=fmin, fmax=fmax, num_mels=num_mels)
            melspec_raw = melspec_raw.astype(np.float32)
            melspec_raw = melspec_raw.T # n_mels x n_frame
            return melspec_raw

        except:
            print(f"{src_filepath}...failed.")
            return None

