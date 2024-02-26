import sys
import os
import cv2
import math
import subprocess
import random
import shutil
from os import listdir, path
from glob import glob
from tqdm import tqdm
import numpy as np
import mediapipe as mp
from talkingface.utils import face_detection
import librosa
import librosa.filters
from scipy import signal
from scipy.io import wavfile

class HyperLipsBaseAudio:
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
        # proposed by @dsmiller
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
            return librosa.stft(y=y, n_fft=self.config['n_fft'], hop_length=self.get_hop_size(),
                                win_length=self.config['win_size'])

    ##########################################################
    # Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
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
    # Librosa correct padding
    def librosa_pad_lr(self, x, fsize, fshift):
        return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]

    def _linear_to_mel(self, spectogram):
        if self._mel_basis is None:
            self._mel_basis = self._build_mel_basis()
        return np.dot(self._mel_basis, spectogram)

    def _build_mel_basis(self):
        assert self.config['fmax'] <= self.config['sample_rate'] // 2
        return librosa.filters.mel(sr=self.config['sample_rate'], n_fft=self.config['n_fft'],
                                   n_mels=self.config['num_mels'],
                                   fmin=self.config['fmin'], fmax=self.config['fmax'])

    def _amp_to_db(self, x):
        min_level = np.exp(self.config['min_level_db'] / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _db_to_amp(x):
        return np.power(10.0, (x) * 0.05)

    def _normalize(self, S):
        if self.config['allow_clipping_in_normalization']:
            if self.config['symmetric_mels']:
                return np.clip((2 * self.config['max_abs_value']) * (
                            (S - self.config['min_level_db']) / (-self.config['min_level_db'])) - self.config[
                                   'max_abs_value'],
                               -self.config['max_abs_value'], self.config['max_abs_value'])
            else:
                return np.clip(
                    self.config['max_abs_value'] * ((S - self.config['min_level_db']) / (-self.config['min_level_db'])),
                    0, self.config['max_abs_value'])

        assert S.max() <= 0 and S.min() - self.config['min_level_db'] >= 0
        if self.config['symmetric_mels']:
            return (2 * self.config['max_abs_value']) * (
                        (S - self.config['min_level_db']) / (-self.config['min_level_db'])) - self.config[
                'max_abs_value']
        else:
            return self.config['max_abs_value'] * ((S - self.config['min_level_db']) / (-self.config['min_level_db']))

    def _denormalize(self, D):
        if self.config['allow_clipping_in_normalization']:
            if self.config['symmetric_mels']:
                return (((np.clip(D, -self.config['max_abs_value'],
                                  self.config['max_abs_value']) + self.config['max_abs_value']) * -self.config[
                    'min_level_db'] / (2 * self.config['max_abs_value']))
                        + self.config['symmetric_mels'])
            else:
                return ((np.clip(D, 0, self.config['max_abs_value']) * -self.config['min_level_db'] / self.config[
                    'max_abs_value']) + self.config['min_level_db'])

        if self.config['symmetric_mels']:
            return (((D + self.config['max_abs_value']) * -self.config['min_level_db'] / (
                        2 * self.config['max_abs_value'])) + self.config['min_level_db'])
        else:
            return ((D * -self.config['min_level_db'] / self.config['max_abs_value']) + self.config['min_level_db'])

class HyperLipsBasePreprocessForInference:
    def __init__(self, config):
        self.config = config
        self.fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False,
                                              device='cuda:{}'.format(config.gpu_id))

    def face_detect(self, images):
        batch_size = self.config['batch_size']
        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size), desc='Running face detection', leave=False):
                    predictions.extend(self.fa.get_detections_for_batch(np.array(images[i:i + batch_size])))
            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError(
                        'Image too big to run face detection on GPU. Please use the --resize_factor argument')
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break
        results = []
        pady1, pady2, padx1, padx2 = self.config['pads']
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
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
                face_det_results = self.face_detect(frames)  # BGR2RGB for CNN face detection
            else:
                face_det_results = self.face_detect([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.config['box']
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = 0 if self.config['static'] else i % len(frames)
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
                img_masked[:, self.config['img_size'] // 2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.config['img_size'] // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def split_video_5s(self):
        print("Starting to divide videos")
        path = self.config.origin_data_root
        video_list = os.listdir(path)
        save_path = os.path.join(self.config.hyperlips_train_dataset, "video_clips", path.split("/")[-1])
        os.makedirs(save_path, exist_ok=True)
        delta_X = 5
        mark = 0

        def get_length(filename):
            result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                     "format=duration", "-of",
                                     "default=noprint_wrappers=1:nokey=1", filename],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            return float(result.stdout)

        for file_name in video_list:
            min = int(get_length(os.path.join(path, file_name))) // 60
            second = int(get_length(os.path.join(path, file_name))) % 60
            video_name = str(file_name.split('.mp4')[0] + 'video_')

            for i in range(min + 1):
                if (second + min * 60) >= delta_X:
                    start_time = 0
                    end_time = start_time + delta_X

                    for j in range((second) + 1):
                        min_temp = str(i)
                        start = str(start_time)
                        end = str(end_time)

                        if len(str(min_temp)) == 1:
                            min_temp = '0' + str(min_temp)
                        if len(str(start_time)) == 1:
                            start = '0' + str(start_time)
                        if len(str(end_time)) == 1:
                            end = '0' + str(end_time)

                        if len(str(mark)) < 6:
                            name = '0' * (6 - len(str(mark)) - 1) + str(mark)
                        else:
                            name = str(mark)

                        command = 'ffmpeg -i {} -ss 00:{}:{} -to 00:{}:{} -strict -2 -ar 16000 -r 25 -qscale 0.001 {}'.format(
                            os.path.join(path, file_name),
                            min_temp, start, min_temp, end,
                            os.path.join(save_path, video_name + 'id' + str(name)) + '.mp4')
                        print(command)

                        mark += 1
                        os.system(command)

                        if i != min or (i == min and (end_time + delta_X) < second):
                            start_time += delta_X
                            end_time += delta_X
                        elif (end_time + delta_X) <= second:
                            start_time += delta_X
                            end_time += delta_X
                        elif (end_time + delta_X) > second:
                            break

    def get_sketch(self, hight, width, image, savepath):
        FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                                   (17, 314), (314, 405), (405, 321), (321, 375),
                                   (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                                   (37, 0), (0, 267),
                                   (267, 269), (269, 270), (270, 409), (409, 291),
                                   (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                                   (14, 317), (317, 402), (402, 318), (318, 324),
                                   (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                                   (82, 13), (13, 312), (312, 311), (311, 310),
                                   (310, 415), (415, 308)])

        FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                                       (374, 380), (380, 381), (381, 382), (382, 362),
                                       (263, 466), (466, 388), (388, 387), (387, 386),
                                       (386, 385), (385, 384), (384, 398), (398, 362)])

        FACEMESH_LEFT_IRIS = frozenset([(474, 475), (475, 476), (476, 477),
                                        (477, 474)])

        FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                           (295, 285), (300, 293), (293, 334),
                                           (334, 296), (296, 336)])

        FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                        (145, 153), (153, 154), (154, 155), (155, 133),
                                        (33, 246), (246, 161), (161, 160), (160, 159),
                                        (159, 158), (158, 157), (157, 173), (173, 133)])

        FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                            (70, 63), (63, 105), (105, 66), (66, 107)])

        FACEMESH_RIGHT_IRIS = frozenset([(469, 470), (470, 471), (471, 472),
                                         (472, 469)])

        FACEMESH_FACE_OVAL = frozenset([(389, 356), (356, 454),
                                        (454, 323), (323, 361), (361, 288), (288, 397),
                                        (397, 365), (365, 379), (379, 378), (378, 400),
                                        (400, 377), (377, 152), (152, 148), (148, 176),
                                        (176, 149), (149, 150), (150, 136), (136, 172),
                                        (172, 58), (58, 132), (132, 93), (93, 234),
                                        (234, 127), (127, 162)])

        FACEMESH_NOSE = frozenset([(168, 6), (6, 197), (197, 195), (195, 5), (5, 4),
                                   (4, 45), (45, 220), (220, 115), (115, 48),
                                   (4, 275), (275, 440), (440, 344), (344, 278), ])
        ROI = frozenset().union(*[FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW,
                                  FACEMESH_RIGHT_EYE, FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL, FACEMESH_NOSE])

        with mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:

            results = face_mesh.process(image)
            if results.multi_face_landmarks == None:
                print("no sketch:" + savepath)
            else:
                face_landmarks = results.multi_face_landmarks[0]
                output = np.zeros((hight, width, 3), np.uint8)
                mp.solutions.drawing_utils.draw_landmarks(
                    image=output,
                    landmark_list=face_landmarks,
                    connections=ROI,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(thickness=6, circle_radius=1, color=(255, 255, 255))
                )
                cv2.imwrite(savepath, output)

    def get_landmarks(self, image, face_mesh, hight, width):
        landmarks = []
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                i = 0
                points = {}
                for landmark in face_landmarks.landmark:
                    x = math.floor(landmark.x * width)
                    y = math.floor(landmark.y * hight)
                    points[i] = (x, y)
                    i += 1
                landmarks.append(points)
        return landmarks

    def get_mask(self, hight, width, image, savepath):
        lip_index = [164, 167, 165, 92, 186, 57, 43, 106, 182, 83, 18, 313, 406, 335, 273, 287, 410, 322, 391, 393]
        with mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:

            face_landmark = self.get_landmarks(image, face_mesh, hight, width)
            if face_landmark == []:
                print("no mask:" + savepath)
            else:
                lip_landmark = []
                for i in lip_index:
                    lip_landmark.append(face_landmark[0][i])
                lip_landmark = np.array(lip_landmark)
                points = lip_landmark.reshape(-1, 1, 2).astype(np.int32)
                matrix = np.zeros((hight, width), dtype=np.int32)
                cv2.drawContours(matrix, [points], -1, (1), thickness=-1)
                list_of_points_indices = np.nonzero(matrix)
                mask = np.zeros((hight, width), np.uint8)
                mask[list_of_points_indices] = 255
                cv2.imwrite(savepath, mask)

    def data_process_hyper_base(self):
        template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
        if self.config.clip_flag == 0:
            filelist = glob(os.path.join(self.config.origin_data_root, '*.mp4'))
            save_dir = os.path.join(self.config.hyperlips_train_dataset, 'video_clips')
            os.makedirs(save_dir, exist_ok=True)

            for video in filelist:
                dirname, filename = os.path.split(video)
                relative_path = os.path.relpath(video, self.config.origin_data_root)
                save_path = os.path.join(self.config.hyperlips_train_dataset, 'video_clips', relative_path)
                print(save_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                shutil.copy(video, save_path)

        filelist = glob(os.path.join(self.config.hyperlips_train_dataset, "video_clips", '*.mp4'))
        filelist_new = []

        for i in filelist:
            res = i.replace('\\', '/')
            filelist_new.append(res)

        for i in tqdm(range(len(filelist_new))):
            vfile = filelist_new[i]
            video_stream = cv2.VideoCapture(vfile)
            frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                frames.append(frame)

            vidname = os.path.basename(vfile).split('.')[0]
            dirname = vfile.split('/')[-2]
            fulldir = path.join(self.config.hyperlips_train_dataset, "imgs", "MEAD", vidname)
            os.makedirs(fulldir, exist_ok=True)

            batches = [frames[i:i + self.config.batch_size] for i in range(0, len(frames), self.config.batch_size)]
            i = -1
            for fb in batches:
                preds = self.fa.get_detections_for_batch(np.asarray(fb))

                for j, f in enumerate(preds):
                    i += 1
                    if f is None:
                        print(vfile + " is wrong")
                        continue
                    x1, y1, x2, y2 = f
                    cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])

            wavpath = path.join(fulldir, 'audio.wav')
            command = template.format(vfile, wavpath)
            subprocess.call(command, shell=True)

    def split_train_test_text(self):
        path = os.path.join(self.config.hyperlips_train_dataset, "imgs")
        train_txt = '../../../dataset/MEAD/filelist/train.txt'
        val_txt = '../../../dataset/MEAD/filelist/val.txt'
        os.makedirs(train_txt.split('/')[-2], exist_ok=True)

        video_list = os.listdir(path)
        list1 = glob(os.path.join(path, "*/*"))
        extor_cout = int(len(list1) * 0.1)
        extor_list = []

        for cout in range(0, extor_cout):
            val_single = random.choice(list1)
            print(val_single)
            val_single = os.path.join(val_single.split('\\')[-2], val_single.split('\\')[-1])
            extor_list.append(val_single)

        with open(train_txt, 'w') as f:
            with open(val_txt, 'w') as f1:
                for item in video_list:
                    path2 = (path) + '/' + item
                    video_list2 = os.listdir(path2)
                    for vilist2 in video_list2:
                        item2 = os.path.join(item, vilist2)
                        if item2 not in extor_list:
                            f.write(item2)
                            f.write('\n')
                        else:
                            f1.write(item2)
                            f1.write('\n')
