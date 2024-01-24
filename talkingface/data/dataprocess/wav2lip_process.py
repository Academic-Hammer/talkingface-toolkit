import os
#import cv2
import numpy as np
import subprocess
from tqdm import tqdm
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from talkingface.utils import face_detection
import librosa
import librosa.filters
from scipy import signal
from scipy.io import wavfile

class Wav2LipAudio:
    """This class is used for audio processing of wav2lip

    这个类提供了从音频到mel谱的方法
    """

    def __init__(self, config):
        self.config = config
            # Conversions
        self._mel_basis = None
    
    def load_wav(self, path, sr):
        return librosa.core.load(path, sr=sr)[0]

    def save_wav(self, wav, path, sr):
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        #proposed by @dsmiller
        wavfile.write(path, sr, wav.astype(np.int16))

    def save_wavenet_wav(self, wav, path, sr):
        librosa.output.write_wav(path, wav, sr=sr)

    def preemphasis(self, wav, k, preemphasize=True):
        if preemphasize:
            return signal.lfilter([1, -k], [1], wav)
        return wav

    def inv_preemphasis(self, wav, k, inv_preemphasize=True):
        if inv_preemphasize:
            return signal.lfilter([1], [1, -k], wav)
        return wav

    def get_hop_size(self):
        hop_size = self.config['hop_size']
        if hop_size is None:
            assert self.config['frame_shift_ms'] is not None
            hop_size = int(self.config['frame_shift_ms'] / 1000 * self.config['sample_rate'])
        return hop_size

    def linearspectrogram(self, wav):
        D = self._stft(self.preemphasis(wav, self.config['preemphasis'], self.config['preemphasize']))
        S = self._amp_to_db(np.abs(D)) - self.config['ref_level_db']

        if self.config['signal_normalization']:
            return self._normalize(S)
        return S

    def melspectrogram(self, wav):
        D = self._stft(self.preemphasis(wav, self.config['preemphasis'], self.config['preemphasize']))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.config['ref_level_db']
        
        if self.config['signal_normalization']:
            return self._normalize(S)
        return S

    def _lws_processor(self):
        import lws
        return lws.lws(self.config['n_fft'], self.get_hop_size(), fftsize=self.config['win_size'], mode="speech")

    def _stft(self, y):
        if self.config['use_lws']:
            return self._lws_processor().stft(y).T
        else:
            return librosa.stft(y=y, n_fft=self.config['n_fft'], hop_length=self.get_hop_size(), win_length=self.config['win_size'])

    ##########################################################
    #Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
    def num_frames(self, length, fsize, fshift):
        """Compute number of time frames of spectrogram
        """
        pad = (fsize - fshift)
        if length % fshift == 0:
            M = (length + pad * 2 - fsize) // fshift + 1
        else:
            M = (length + pad * 2 - fsize) // fshift + 2
        return M


    def pad_lr(self, x, fsize, fshift):
        """Compute left and right padding
        """
        M = self.num_frames(len(x), fsize, fshift)
        pad = (fsize - fshift)
        T = len(x) + 2 * pad
        r = (M - 1) * fshift + fsize - T
        return pad, pad + r
    ##########################################################
    #Librosa correct padding
    def librosa_pad_lr(self, x, fsize, fshift):
        return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

    def _linear_to_mel(self, spectogram):
        if self._mel_basis is None:
            self._mel_basis = self._build_mel_basis()
        return np.dot(self._mel_basis, spectogram)

    def _build_mel_basis(self):
        assert self.config['fmax'] <= self.config['sample_rate'] // 2
        return librosa.filters.mel(sr=self.config['sample_rate'], n_fft=self.config['n_fft'], n_mels=self.config['num_mels'],
                                fmin=self.config['fmin'], fmax=self.config['fmax'])

    def _amp_to_db(self, x):
        min_level = np.exp(self.config['min_level_db'] / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _db_to_amp(x):
        return np.power(10.0, (x) * 0.05)

    def _normalize(self, S):
        if self.config['allow_clipping_in_normalization']:
            if self.config['symmetric_mels']:
                return np.clip((2 * self.config['max_abs_value']) * ((S - self.config['min_level_db']) / (-self.config['min_level_db'])) - self.config['max_abs_value'],
                            -self.config['max_abs_value'], self.config['max_abs_value'])
            else:
                return np.clip(self.config['max_abs_value'] * ((S - self.config['min_level_db']) / (-self.config['min_level_db'])), 0, self.config['max_abs_value'])
        
        assert S.max() <= 0 and S.min() - self.config['min_level_db'] >= 0
        if self.config['symmetric_mels']:
            return (2 * self.config['max_abs_value']) * ((S - self.config['min_level_db']) / (-self.config['min_level_db'])) - self.config['max_abs_value']
        else:
            return self.config['max_abs_value'] * ((S - self.config['min_level_db']) / (-self.config['min_level_db']))

    def _denormalize(self, D):
        if self.config['allow_clipping_in_normalization']:
            if self.config['symmetric_mels']:
                return (((np.clip(D, -self.config['max_abs_value'],
                                self.config['max_abs_value']) + self.config['max_abs_value']) * -self.config['min_level_db'] / (2 * self.config['max_abs_value']))
                        + self.config['symmetric_mels'])
            else:
                return ((np.clip(D, 0, self.config['max_abs_value']) * -self.config['min_level_db'] / self.config['max_abs_value']) + self.config['min_level_db'])
        
        if self.config['symmetric_mels']:
            return (((D + self.config['max_abs_value']) * -self.config['min_level_db'] / (2 * self.config['max_abs_value'])) + self.config['min_level_db'])
        else:
            return ((D * -self.config['min_level_db'] / self.config['max_abs_value']) + self.config['min_level_db'])



class Wav2LipPreprocessForInference:
    """This class is used for preprocessing of wav2lip inference

    face_detect: detect face in the image
    datagen: generate data for inference

    """
    def __init__(self, config):
        self.config = config
        self.fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
                                                device=config['device'])
    
    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes
    
    def face_detect(self, images):
        batch_size = self.config['batch_size']
        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size), desc='Running face detection', leave=False):
                    predictions.extend(self.fa.get_detections_for_batch(np.array(images[i:i + batch_size])))
            except RuntimeError:
                if batch_size == 1: 
                    raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break
        results = []
        pady1, pady2, padx1, padx2 = self.config['pads']
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            
            results.append([x1, y1, x2, y2])
        boxes = np.array(results)
        if not self.config['nosmooth']: boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        return results
    

    def datagen(self, frames, face_det_results, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.config['box'][0] == -1:
            if not self.config['static']:
                face_det_results = self.face_detect(frames) # BGR2RGB for CNN face detection
            else:
                face_det_results = self.face_detect([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.config['box']
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = 0 if self.config['static'] else i%len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.config['img_size'], self.config['img_size']))
                
            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.config['wav2lip_batch_size']:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.config['img_size']//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.config['img_size']//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch