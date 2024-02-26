import multiprocessing as mp
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
import mediapipe as mp
import math
import shutil
import sys
from talkingface.utils import face_detection
from os import path

class HyperLipsHRPreprocessForInference:
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

    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

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

    def data_process_hyper_hq_module(self):
        # 获取所有高质量视频文件的路径
        filelist = glob(path.join(self.config['preprocessed_root'], 'video_clips', '*/*.mp4'))
        filelist_new = []
        # 构建高分辨率训练数据集的根目录
        hyperlipsHR_train_dataset = os.path.join(self.config['preprocessed_root'], "HR_Train_Dateset")
        os.makedirs(hyperlipsHR_train_dataset, exist_ok=True)

        # 转换路径分隔符，并将路径添加到新的列表中
        for i in filelist:
            res = i.replace('\\', '/')
            filelist_new.append(res)
        print(filelist_new)
        for i in tqdm(range(len(filelist_new))):
            vfile_h = filelist_new[i]
            vidname = os.path.basename(vfile_h).split('.')[0]
            dirname = vfile_h.split('/')[-2]

            # 构建原始数据和HyperLips训练数据的视频文件路径
            vfile_o = path.join(self.config['preprocessed_root'], 'video_clips',
                                vidname + ".mp4")

            # 构建高分辨率训练数据集的子目录
            fulldir_origin_data_img = path.join(hyperlipsHR_train_dataset, 'GT_IMG', dirname, vidname)
            os.makedirs(fulldir_origin_data_img, exist_ok=True)
            fulldir_hyper_img = path.join(hyperlipsHR_train_dataset, 'HYPER_IMG', dirname, vidname)
            os.makedirs(fulldir_hyper_img, exist_ok=True)
            fulldir_origin_mask = path.join(hyperlipsHR_train_dataset, 'GT_MASK', dirname, vidname)
            os.makedirs(fulldir_origin_mask, exist_ok=True)
            fulldir_origin_sketch = path.join(hyperlipsHR_train_dataset, 'GT_SKETCH', dirname, vidname)
            os.makedirs(fulldir_origin_sketch, exist_ok=True)
            fulldir_hyper_sketch = path.join(hyperlipsHR_train_dataset, 'HYPER_SKETCH', dirname, vidname)
            os.makedirs(fulldir_hyper_sketch, exist_ok=True)

            # 读取高质量视频的帧
            video_stream_h = cv2.VideoCapture(vfile_h)
            video_stream_o = cv2.VideoCapture(vfile_o)
            frames_h = []
            frames_o = []
            while 1:
                still_reading_o, frame_o = video_stream_o.read()
                still_reading_h, frame_h = video_stream_h.read()
                if not still_reading_h:
                    video_stream_h.release()
                    video_stream_o.release()
                    break
                frames_h.append(frame_h)
                frames_o.append(frame_o)

            # 划分帧为小批次
            batches_h = [frames_h[i:i + self.config["batch_size"]] for i in range(0, len(frames_h), self.config["batch_size"])]
            batches_o = [frames_o[i:i + self.config["batch_size"]] for i in range(0, len(frames_o), self.config["batch_size"])]
            num = -1

            # 遍历每个小批次
            for i in range(len(batches_h)):
                f_o = batches_o[i]
                f_h = batches_h[i]
                preds = self.fa.get_detections_for_batch(np.asarray(batches_h[i]))

                # 遍历每帧
                for j, f in enumerate(preds):
                    num += 1
                    if f is None:
                        continue
                    x1, y1, x2, y2 = f

                    # 保存原始数据和HyperLips训练数据的图片
                    cv2.imwrite(path.join(fulldir_origin_data_img, '{}.jpg'.format(num)), f_o[j][y1:y2, x1:x2])
                    cv2.imwrite(path.join(fulldir_hyper_img, '{}.jpg'.format(num)), f_h[j][y1:y2, x1:x2])

                    # 计算口罩区域的高度和宽度，并保存原始数据的口罩掩码
                    hight = y2 - y1
                    width = x2 - x1
                    savepath_origin_mask = path.join(fulldir_origin_mask, '{}.jpg'.format(num))
                    self.get_mask(hight, width, f_o[j][y1:y2, x1:x2], savepath_origin_mask)

                    # 保存原始数据和HyperLips训练数据的口罩轮廓
                    savepath_origin_sketch = path.join(fulldir_origin_sketch, '{}.jpg'.format(num))
                    self.get_sketch(hight, width, f_o[j][y1:y2, x1:x2], savepath_origin_sketch)
                    savepath_hyper_sketch = path.join(fulldir_hyper_sketch, '{}.jpg'.format(num))
                    self.get_sketch(hight, width, f_h[j][y1:y2, x1:x2], savepath_hyper_sketch)

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