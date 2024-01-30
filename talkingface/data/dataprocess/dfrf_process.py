import cv2
import numpy as np
import face_alignment
from skimage import io
import torch
import torch.nn.functional as F
import json
import os
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import argparse
from talkingface.data.dataprocess.DFRD_process_helper.face_tracking.geo_transform import euler2rot

class DFRF_process():
    """
    # Class: DFRF_process
    准备训练DFRF模型所需的数据方法组成的类    
    """
    def __init__(self, config):
        self.config = config

    def steps(self, step):
        """
        # Method: steps
        ## 作用
        输入想要步骤对应的编号，即可执行对应的步骤
        ## 原理说明
        包含了数据预处理中每个步骤所需的具体代码
        ## 注意
        数据预处理的过程每个步骤之间有先后制约关系，因此必须按照顺序依次执行，否则会报错
        """
        id = 0
        running_step = step
        vid_file = os.path.join('dataset', 'vids', id+'.mp4')
        if running_step in [0,1]:
            if not os.path.isfile(vid_file):
                print('no video')
                exit()


        id_dir = os.path.join('dataset', id)
        Path(id_dir).mkdir(parents=True, exist_ok=True)
        id_dir = os.path.join('dataset', id, '0')
        Path(id_dir).mkdir(parents=True, exist_ok=True)
        ori_imgs_dir = os.path.join(id_dir, 'ori_imgs')
        Path(ori_imgs_dir).mkdir(parents=True, exist_ok=True)
        parsing_dir = os.path.join(id_dir, 'parsing')
        Path(parsing_dir).mkdir(parents=True, exist_ok=True)
        head_imgs_dir = os.path.join(id_dir, 'head_imgs')
        Path(head_imgs_dir).mkdir(parents=True, exist_ok=True)
        com_imgs_dir = os.path.join(id_dir, 'com_imgs')
        Path(com_imgs_dir).mkdir(parents=True, exist_ok=True)
        torso_imgs = os.path.join(id_dir, 'torso_imgs')
        Path(torso_imgs).mkdir(parents=True, exist_ok=True)


        # # Step 0: extract wav & deepspeech feature, better run in terminal to parallel with
        # below commands since this may take a few minutes
        if running_step == 0:
            print('--- Step0: extract deepspeech feature ---')
            wav_file = os.path.join(id_dir, 'aud.wav')
            extract_wav_cmd = 'ffmpeg -i ' + vid_file + ' -f wav -ar 16000 ' + wav_file
            os.system(extract_wav_cmd)
            extract_ds_cmd = 'python data_util/deepspeech_features/extract_ds_features.py --input=' + id_dir
            os.system(extract_ds_cmd)
            exit()

        # Step 1: extract images
        if running_step == 1:
            print('--- Step1: extract images from vids ---')
            cap = cv2.VideoCapture(vid_file)
            frame_num = 0
            while(True):
                _, frame = cap.read()
                if frame is None:
                    break
                cv2.imwrite(os.path.join(ori_imgs_dir, str(frame_num) + '.jpg'), frame)
                frame_num = frame_num + 1
            cap.release()
            exit()

        # Step 2: detect lands
        if running_step == 2:
            print('--- Step 2: detect landmarks ---')
            fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, flip_input=False)
            # print(ori_imgs_dir)
            for image_path in os.listdir(ori_imgs_dir):
                    if image_path.endswith('.jpg'):
                        input = io.imread(os.path.join(ori_imgs_dir, image_path))[:, :, :3]
                        preds = fa.get_landmarks(input)
                        if len(preds) > 0:
                            lands = preds[0].reshape(-1, 2)[:,:2]
                            np.savetxt(os.path.join(ori_imgs_dir, image_path[:-3] + 'lms'), lands, '%f')


        max_frame_num = 100000
        valid_img_ids = []
        for i in range(max_frame_num):
            if os.path.isfile(os.path.join(ori_imgs_dir, str(i) + '.lms')):
                valid_img_ids.append(i)
        valid_img_num = len(valid_img_ids)
        print(valid_img_ids)
        tmp_img = cv2.imread(os.path.join(ori_imgs_dir, str(valid_img_ids[0])+'.jpg'))
        h, w = tmp_img.shape[0], tmp_img.shape[1]



        # Step 3: face parsing
        if running_step == 3:
            print('--- Step 3: face parsing ---')
            face_parsing_cmd = 'python data_util/face_parsing/test.py --respath=' + \
                id_dir + '/parsing --imgpath=' + id_dir + '/ori_imgs'
            os.system(face_parsing_cmd)

        # Step 4: extract bc image
        if running_step == 4:
            print('--- Step 4: extract background image ---')
            sel_ids = np.array(valid_img_ids)[np.arange(0, valid_img_num, 20)]
            all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()
            distss = []
            for i in sel_ids:
                parse_img = cv2.imread(os.path.join(id_dir, 'parsing', str(i) + '.png'))
                bg = (parse_img[..., 0] == 255) & (
                    parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
                fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
                dists, _ = nbrs.kneighbors(all_xys)
                distss.append(dists)
            distss = np.stack(distss)
            print(distss.shape)
            max_dist = np.max(distss, 0)
            max_id = np.argmax(distss, 0)
            bc_pixs = max_dist > 5
            bc_pixs_id = np.nonzero(bc_pixs)
            bc_ids = max_id[bc_pixs]
            imgs = []
            num_pixs = distss.shape[1]
            for i in sel_ids:
                img = cv2.imread(os.path.join(ori_imgs_dir, str(i) + '.jpg'))
                imgs.append(img)
            imgs = np.stack(imgs).reshape(-1, num_pixs, 3)
            bc_img = np.zeros((h*w, 3), dtype=np.uint8)
            bc_img[bc_pixs_id, :] = imgs[bc_ids, bc_pixs_id, :]
            bc_img = bc_img.reshape(h, w, 3)
            max_dist = max_dist.reshape(h, w)
            bc_pixs = max_dist > 5
            bg_xys = np.stack(np.nonzero(~bc_pixs)).transpose()
            fg_xys = np.stack(np.nonzero(bc_pixs)).transpose()
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
            distances, indices = nbrs.kneighbors(bg_xys)
            bg_fg_xys = fg_xys[indices[:, 0]]
            print(fg_xys.shape)
            print(np.max(bg_fg_xys), np.min(bg_fg_xys))
            bc_img[bg_xys[:, 0], bg_xys[:, 1],
                :] = bc_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]
            cv2.imwrite(os.path.join(id_dir, 'bc.jpg'), bc_img)

        # Step 5: save training images
        if running_step == 5:
            print('--- Step 5: save training images ---')
            bc_img = cv2.imread(os.path.join(id_dir, 'bc.jpg'))

            def dilate_head_part(img, bc_img, parse_img, kernel_size=11, verbose=False):
                head_part = (parse_img[..., 0] == 255) & (
                        parse_img[..., 1] == 0) & (parse_img[..., 2] == 0)
                neck_part = (parse_img[..., 0] == 0) & (
                        parse_img[..., 1] == 255) & (parse_img[..., 2] == 0)
                torso_part = (parse_img[..., 0] == 0) & (
                        parse_img[..., 1] == 0) & (parse_img[..., 2] == 255)

                torso_img = img.copy()
                torso_img[head_part] = bc_img[head_part]
                if verbose:
                    cv2.imwrite("old_torso.png", torso_img)

                dialte_head_torso_img = img.copy()
                KERNEL_SIZE = kernel_size
                kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE))
                head_part_dilate = cv2.dilate(head_part.astype('uint8'), kernel) > 0
                dialte_head_torso_img[head_part_dilate] = bc_img[head_part_dilate]

                for part in (neck_part, torso_part):
                    dialte_head_torso_img[part] = img[part]

                KERNEL_SIZE = 4
                kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE))
                head_part_dilate = cv2.dilate(head_part.astype('uint8'), kernel) > 0
                dialte_head_torso_img[head_part_dilate] = bc_img[head_part_dilate]
                new_neck_part = neck_part.copy()
                new_neck_part[head_part_dilate]=False
                if verbose:
                    cv2.imwrite("mid_torso.png", dialte_head_torso_img)

                return dialte_head_torso_img, new_neck_part


            def get_xys(x):
                return np.stack(np.nonzero(x)).transpose()

            def inpaint_neck_part2(neck_torso_img, parse_img, new_neck_part, inpaint_upper_len=50, erode_neck_kernel_size=5, num_neighbor=5, verbose=False):
                # 1. get neck_part
                # 2. the coord of neck upper bound -- left, right, up
                # 3. let upper bound rig to 1
                # 4. use KNN find coresponence
                head_part = (parse_img[..., 0] == 255) & (
                        parse_img[..., 1] == 0) & (parse_img[..., 2] == 0)
                neck_part = (parse_img[..., 0] == 0) & (
                        parse_img[..., 1] == 255) & (parse_img[..., 2] == 0)
                neck_part = new_neck_part
                torso_part = (parse_img[..., 0] == 0) & (
                        parse_img[..., 1] == 0) & (parse_img[..., 2] == 255)
                h, w = neck_torso_img.shape[:2]

                INPAINT_UPPER_LEN = inpaint_upper_len
                ERODE_NECK_KERNEL_SIZE = erode_neck_kernel_size
                HEAD_NECK_SIZE = ERODE_NECK_KERNEL_SIZE
                NUM_NEIGHBOR = num_neighbor
                neck_part_idx = get_xys(neck_part)

                left_, right_ = neck_part_idx[:, 1].min(), neck_part_idx[:, 1].max()

                _neck_part = neck_part.copy() * 1.0
                _neck_part[_neck_part == 0] = _neck_part.max() + 1
                upper_idx = np.argmin(_neck_part, axis=0)

                left = left_
                right = right_
                for i in range(left_, right_+1):
                    if parse_img[upper_idx[i]-5, i, 2] == 255:
                        if i < (left_+right_)/2:
                            left = left + 1
                        else:
                            right = right - 1

                upper_part = np.zeros_like(neck_part) * 1.0
                dialte_upper_part = np.zeros_like(neck_part) * 1.0
                for i in range(left, right + 1):
                    anchor = upper_idx[i]
                    neck_torso_img[anchor - INPAINT_UPPER_LEN: anchor, i] = neck_torso_img[anchor, i]

                if verbose:
                    cv2.imwrite("new_torso.png", neck_torso_img)
                return neck_torso_img


            for i in valid_img_ids:
                parsing_img = cv2.imread(os.path.join(parsing_dir, str(i) + '.png'))
                head_part = (parsing_img[:, :, 0] == 255) & (
                    parsing_img[:, :, 1] == 0) & (parsing_img[:, :, 2] == 0)
                bc_part = (parsing_img[:, :, 0] == 255) & (
                    parsing_img[:, :, 1] == 255) & (parsing_img[:, :, 2] == 255)
                img = cv2.imread(os.path.join(ori_imgs_dir, str(i) + '.jpg'))

                torso_img_newbc = img.copy()
                torso_img_newbc[bc_part] = bc_img[bc_part]
                neck_torso_img, new_neck_part = dilate_head_part(torso_img_newbc, bc_img, parsing_img)

                neck_torso_img = inpaint_neck_part2(neck_torso_img, parsing_img, new_neck_part)
                cv2.imwrite(os.path.join(torso_imgs, str(i) + '.jpg'), neck_torso_img)

                img[bc_part] = bc_img[bc_part]
                cv2.imwrite(os.path.join(com_imgs_dir, str(i) + '.jpg'), img)
                img[~head_part] = bc_img[~head_part]
                cv2.imwrite(os.path.join(head_imgs_dir, str(i) + '.jpg'), img)


        # Step 6: estimate head pose
        if running_step == 6:
            print('--- Estimate Head Pose ---')
            est_pose_cmd = 'python data_util/face_tracking/face_tracker.py --idname=' + \
                id + '/0' + ' --img_h=' + str(h) + ' --img_w=' + str(w) + \
                ' --frame_num=' + str(max_frame_num)
            os.system(est_pose_cmd)

        # Step 7: save transform param & write config file
        if running_step == 7:
            print('--- Step 7: Save Transform Param ---')
            params_dict = torch.load(os.path.join(id_dir, 'track_params.pt'))
            focal_len = params_dict['focal']
            euler_angle = params_dict['euler']
            trans = params_dict['trans'] / 10.0
            #trans = params_dict['trans']
            valid_num = euler_angle.shape[0]
            train_val_split = int(valid_num*0.5)
            train_ids = torch.arange(0, train_val_split)
            val_ids = torch.arange(train_val_split, valid_num)
            rot = euler2rot(euler_angle)
            rot_inv = rot.permute(0, 2, 1)
            trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(2))
            pose = torch.eye(4, dtype=torch.float32)
            save_ids = ['train', 'val']
            train_val_ids = [train_ids, val_ids]
            mean_z = -float(torch.mean(trans[:, 2]).item())
            for i in range(2):
                transform_dict = dict()
                transform_dict['focal_len'] = float(focal_len[0])
                transform_dict['cx'] = float(w/2.0)
                transform_dict['cy'] = float(h/2.0)
                transform_dict['frames'] = []
                ids = train_val_ids[i]
                save_id = save_ids[i]
                for i in ids:
                    i = i.item()
                    frame_dict = dict()
                    frame_dict['img_id'] = int(valid_img_ids[i])
                    frame_dict['aud_id'] = int(valid_img_ids[i])
                    pose[:3, :3] = rot_inv[i]
                    pose[:3, 3] = trans_inv[i, :, 0]
                    frame_dict['transform_matrix'] = pose.numpy().tolist()
                    lms = np.loadtxt(os.path.join(
                        ori_imgs_dir, str(valid_img_ids[i]) + '.lms'))
                    min_x, max_x = np.min(lms, 0)[0], np.max(lms, 0)[0]
                    import pdb
                    cx = int((min_x+max_x)/2.0)
                    cy = int(lms[27, 1])
                    h_w = int((max_x-cx)*1.5)
                    h_h = int((lms[8, 1]-cy)*1.15)
                    rect_x = cx - h_w
                    rect_y = cy - h_h
                    if rect_x < 0:
                        rect_x = 0
                    if rect_y < 0:
                        rect_y = 0
                    rect_w = min(w-1-rect_x, 2*h_w)
                    rect_h = min(h-1-rect_y, 2*h_h)
                    rect = np.array((rect_x, rect_y, rect_w, rect_h), dtype=np.int32)
                    frame_dict['face_rect'] = rect.tolist()

                    min_x, max_x = np.min(lms, 0)[0], np.max(lms, 0)[0]
                    cx = int((min_x+max_x)/2.0)
                    cy = int(lms[66, 1])
                    h_w = int((lms[54, 0]-cx)*1.2)
                    h_h = int((lms[57, 1]-cy)*1.2)
                    rect_x = cx - h_w
                    rect_y = cy - h_h
                    if rect_x < 0:
                        rect_x = 0
                    if rect_y < 0:
                        rect_y = 0
                    rect_w = min(w-1-rect_x, 2*h_w)
                    rect_h = min(h-1-rect_y, 2*h_h)
                    rect = np.array((rect_x, rect_y, rect_w, rect_h), dtype=np.int32)
                    frame_dict['lip_rect'] = rect.tolist()

                    transform_dict['frames'].append(frame_dict)
                with open(os.path.join(id_dir, 'transforms_' + save_id + '.json'), 'w') as fp:
                    json.dump(transform_dict, fp, indent=2, separators=(',', ': '))

            dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            testskip = int(val_ids.shape[0]/7)

            config_file = os.path.join(id_dir, 'config.txt')
            with open(config_file, 'w') as file:
                file.write('expname = ' + id + '\n')
                file.write('expname_finetune = ' + id + '\n')
                file.write('datadir = dataset/' + id + '\n')
                file.write('basedir = dataset/finetune_models' + '\n')
                file.write('ft_path = dataset/base_model/base.tar' + '\n')
                file.write('near = ' + str(mean_z-0.2) + '\n')
                file.write('far = ' + str(mean_z+0.4) + '\n')
                file.write('testskip = 1 ' + '\n')
                file.write('L2loss_weight = 5e-8' + '\n')
                file.write('train_length = 15' + '\n')
                file.write('bc_type=torso_imgs' + '\n')

            print(id + ' data processed done!')

    def run(self):
        """
        # Method: run
        ## 作用
        执行数据预处理的全过程
        ## 原理说明
        执行这个run函数即可完成所有训练这个模型所需的数据的数据预处理准备工作,
        函数将依次执行8个数据预处理过程0-7，
        每个步骤执行的内容在steps函数中
        """
        for i in range(8):
            self.steps(i)