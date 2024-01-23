import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, constant_
from tqdm import tqdm
from os import listdir, path
import numpy as np
import os, subprocess
from glob import glob
import cv2
from ...data.dataprocess.wav2lip_process import *


from talkingface.model.layers import Conv2d, Conv2dTranspose, nonorm_Conv2d
from talkingface.model.abstract_talkingface import AbstractTalkingFace
from talkingface.utils import ensure_dir

class SyncNet_color(nn.Module):
    def __init__(self):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)


        return audio_embedding, face_embedding
    






class Wav2Lip(AbstractTalkingFace):
    """wav2lip is a GAN-based model that predict the final with audio and image"""
    def __init__(self, config):
        super(Wav2Lip, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)), # 96,96

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 48,48
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # 24,24
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 12,12
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),       # 6,6
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),     # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),
            
            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     # 1, 1
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0), # 3,3
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),), # 12, 12

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 24, 24

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 48, 48

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),),]) # 96,96

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()) 
        self.config = config
        self.l1loss = nn.L1Loss()
        self.bceloss = nn.BCELoss()
    def forward(self, audio_sequences, face_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
            
            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x
            
        return outputs
    
    def predict(self, audio_sequences, face_sequences):
        return self.forward(audio_sequences, face_sequences)

    def calculate_loss(self, interaction, valid=False):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        indiv_mels = interaction['indiv_mels'].to(self.config['device'])
        input_frames = interaction['input_frames'].to(self.config['device'])
        mel = interaction['mels'].to(self.config['device'])
        gt = interaction['gt'].to(self.config['device'])
        g_frames = self.forward(indiv_mels, input_frames)
        l1loss = self.l1loss(g_frames, gt)
        if self.config['syncnet_wt'] > 0 or valid:
            sync_loss = self.syncnet_loss(mel, g_frames)
        else:
            sync_loss = 0
        
        loss = self.config['syncnet_wt'] * sync_loss + (1- self.config['syncnet_wt']) * l1loss
        return {"loss":loss, "l1loss":l1loss, "sync_loss":sync_loss} 
    
    def syncnet_loss(self, mel, g_frames):
        syncnet = self.load_syncnet()
        syncnet.eval()
        g = g_frames[:, :, :, g_frames.size(3)//2:]
        g = torch.cat([g[:, :, i] for i in range(self.config['syncnet_T'])], dim=1)
        # B, 3 * T, H//2, W
        a, v = syncnet(mel, g)
        y = torch.ones(g.size(0), 1).float().to(self.config['device'])
        return self.cosine_loss(a, v, y)
    
    def cosine_loss(self, a, v, y):
        d = nn.functional.cosine_similarity(a, v)
        loss = self.bceloss(d.unsqueeze(1), y)

        return loss
    
    def load_syncnet(self):
        syncnet = SyncNet_color().to(self.config['device'])
        for p in syncnet.parameters():
            p.requires_grad = False
        checkpoint = torch.load(self.config["syncnet_checkpoint_path"])
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        syncnet.load_state_dict(new_s)
        return syncnet

    def generate_batch(self):
        audio_processor = Wav2LipAudio(self.config)
        video_processor = Wav2LipPreprocessForInference(self.config)

        with open(self.config['test_filelist'], 'r') as filelist:
            lines = filelist.readlines()

        file_dict = {'generated_video': [], 'real_video': []}
        for idx, line in enumerate(tqdm(lines, desc='generate video')):
            file_src = line.split()[0]

            audio_src = os.path.join(self.config['data_root'], file_src) + '.mp4'
            video = os.path.join(self.config['data_root'], file_src) + '.mp4'


            ensure_dir(os.path.join(self.config['temp_dir']))

            command = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'.format(audio_src, os.path.join(self.config['temp_dir'], 'temp')+'.wav')
            subprocess.call(command, shell=True)

            temp_audio = os.path.join(self.config['temp_dir'], 'temp')+'.wav'
            wav = audio_processor.load_wav(temp_audio, 16000)
            mel = audio_processor.melspectrogram(wav)

            if np.isnan(mel.reshape(-1)).sum() > 0:
                continue

            mel_idx_multiplier = 80./self.config['fps']
            mel_chunks = []
            i = 0
            while 1:
                start_idx = int(i *  mel_idx_multiplier)
                if start_idx + self.config['mel_step_size'] > len(mel[0]):
                    break
                mel_chunks.append(mel[:, start_idx : start_idx + self.config['mel_step_size']])
                i += 1

            video_stream = cv2.VideoCapture(video)
            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading or len(full_frames) > len(mel_chunks):
                    video_stream.release()
                    break
                full_frames.append(frame)

            if len(full_frames) < len(mel_chunks):
                continue

            full_frames = full_frames[:len(mel_chunks)]

            try:
                face_det_results = video_processor.face_detect(full_frames.copy())
            except ValueError as e:
                continue

            batch_size = self.config['wav2lip_batch_size']
            gen = video_processor.datagen(full_frames.copy(), face_det_results, mel_chunks)

            for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
                if i == 0:
                    frame_h, frame_w = full_frames[0].shape[:-1]
                    output_video_path = os.path.join(self.config['temp_dir'], 'temp')+'.mp4'
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者尝试 'avc1'
                    out = cv2.VideoWriter(output_video_path, fourcc, 25, (frame_w, frame_h))


                img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.config['device'])
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.config['device'])

                with torch.no_grad():
                    pred = self.predict(mel_batch, img_batch)
                        

                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
                
                for pl, f, c in zip(pred, frames, coords):
                    y1, y2, x1, x2 = c
                    pl = cv2.resize(pl.astype(np.uint8), (x2 - x1, y2 - y1))
                    f[y1:y2, x1:x2] = pl
                    out.write(f)

            out.release()

            vid = os.path.join(self.config['temp_dir'], file_src) + '.mp4'
            vid_directory = os.path.dirname(vid)
            if not os.path.exists(vid_directory):
                os.makedirs(vid_directory)

            command = 'ffmpeg -loglevel panic -y -i {} -i {} -strict -2 -q:v 1 {}'.format(temp_audio, 
                                    os.path.join(self.config['temp_dir'], 'temp')+'.mp4', vid)
            process_status = subprocess.call(command, shell=True)
            if process_status == 0:
                file_dict['generated_video'].append(vid)
                file_dict['real_video'].append(video)
            else:
                continue
        return file_dict

