import os
import cv2
import numpy as np
import subprocess
from tqdm import tqdm
from glob import glob
import zipfile

import talkingface.utils.audio
from talkingface.utils.audio.stft import TacotronSTFT
from concurrent.futures import ThreadPoolExecutor, as_completed
from talkingface.utils import face_detection
import traceback
import librosa
import librosa.filters
from scipy import signal
from scipy.io import wavfile


class lrs2Preprocess:
    def __init__(self, config):
        self.config = config
        self.fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
                                                device=f'cuda:{id}') for id in range(config['ngpu'])]
        self.template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'

    def process_video_file(self, vfile, gpu_id):
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

        fulldir = os.path.join(self.config['preprocessed_root'], dirname, vidname)
        os.makedirs(fulldir, exist_ok=True)

        batches = [frames[i:i + self.config['preprocess_batch_size']] for i in range(0, len(frames), self.config['preprocess_batch_size'])]

        i = -1
        for fb in batches:
            preds = self.fa[gpu_id].get_detections_for_batch(np.asarray(fb))

            for j, f in enumerate(preds):
                i += 1
                if f is None:
                    continue

                x1, y1, x2, y2 = f
                cv2.imwrite(os.path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])


    def process_audio_file(self, vfile):
        vidname = os.path.basename(vfile).split('.')[0]
        dirname = vfile.split('/')[-2]

        fulldir = os.path.join(self.config['preprocessed_root'], dirname, vidname)
        os.makedirs(fulldir, exist_ok=True)

        wavpath = os.path.join(fulldir, 'audio.wav')

        command =self.template.format(vfile, wavpath)
        subprocess.call(command, shell=True)

    def mp_handler(self, job):
        vfile, gpu_id = job
        try:
            self.process_video_file(vfile, gpu_id)
        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc()

    def run(self):
        print(f'Started processing for {self.config["data_root"]} with {self.config["ngpu"]} GPUs')
        
        filelist = glob(os.path.join(self.config["data_root"], '*/*.mp4'))

        # jobs = [(vfile, i % self.config["ngpu"]) for i, vfile in enumerate(filelist)]
        # with ThreadPoolExecutor(self.config["ngpu"]) as p:
        #     futures = [p.submit(self.mp_handler, j) for j in jobs]
        #     _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

        print('Dumping audios...')
        for vfile in tqdm(filelist):
            try:
                self.process_audio_file(vfile)
            except KeyboardInterrupt:
                exit(0)
            except:
                traceback.print_exc()
                continue


class LJSpeechPreprocess:
    #进行数据预处理：1.音频转mel谱。2.csv文本数据抽取
    def __init__(self, config):
        self.config = config
        self._stft = TacotronSTFT(
    self.config['filter_length'], self.config['hop_length'], self.config['win_length'],
    self.config['n_mel_channels'], self.config['sampling_rate'], self.config['mel_fmin'],
    self.config['mel_fmax'])

    def run(self):
        #数据预处理入口
        in_dir = self.config['data_root']
        out_dir = self.config['preprocessed_root']
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        metadata = self.build_from_path(in_dir, out_dir)
        #写数据切分用的
        #self.write_metadata(metadata, out_dir)
        self.unzip_alignments()

    def unzip_alignments(self):


        # 指定要解压的 ZIP 文件路径
        zip_file_path = os.path.join(self.config['data_root'],'alignments.zip')

        # 指定解压后的目标文件夹
        extract_folder = self.config['preprocessed_root']

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # 解压所有文件到目标文件夹
            zip_ref.extractall(extract_folder)


    def write_metadata(self,metadata, out_dir):
        if not os.path.exists(self.config['train_filelist']):
            with open(self.config['train_filelist'], 'w', encoding='utf-8') as train_file, \
                    open(self.config['val_filelist'], 'w', encoding='utf-8') as val_file, \
                    open(self.config['test_filelist'], 'w', encoding='utf-8') as test_file:
                index=1
                for m in metadata:
                    if index<=self.config['train_data_num']:
                        train_file.write(''.join(map(str, m)) + '\n')
                        index=index+1
                    elif index<=(self.config['train_data_num']+self.config['val_data_num']):
                        val_file.write(''.join(map(str, m)) + '\n')
                        index = index + 1
                    else:
                        test_file.write(''.join(map(str, m)) + '\n')
                        index = index + 1



    def build_from_path(self,in_dir, out_dir):
        #生成 mel 谱图，并返回语音文本列表
        index = 1
        texts = []
        print("数据预处理：")
        with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                parts = line.strip().split('|')
                wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
                text = parts[2]
                texts.append((parts[0],'|','{0:05d}'.format(index),'|',self._process_utterance(out_dir, index, wav_path, text)))
                index = index + 1
        print('数据预处理完成')
        return texts

    def _process_utterance(self,out_dir, index, wav_path, text):
        # 计算mel谱图并写入磁盘
        mel_spectrogram = talkingface.utils.audio.tools.get_mel(wav_path).numpy().astype(np.float32)
        # 将mel谱图以.npy格式写入磁盘
        mel_filename = 'LJSpeech-mel-%05d.npy' % index
        mel_path=os.path.join(out_dir, 'mels')
        if not os.path.exists(mel_path):
            os.makedirs(mel_path)
        np.save(os.path.join(mel_path, mel_filename),
                mel_spectrogram.T, allow_pickle=False)
        return text


if __name__ == "__main__":
    # TEST
    config = {
        'max_wav_value': 32768.0,
        'sampling_rate': 22050,
        'filter_length': 1024,
        'hop_length': 256,
        'win_length': 1024,
        'n_mel_channels': 80,
        'mel_fmin': 0.0,
        'mel_fmax': 8000.0,
        'train_data_num':12000,
        'preprocessed_root':"../../dataset/LJSpeech/ljspeech_preprocessed",
        "train_filelist":'../../dataset/LJSpeech/filelist/train.txt',
        "val_filelist":'../../dataset/LJSpeech/filelist/val.txt'
    }
    pre_pro = LJSpeechPreprocess(config=config)
    data_root='../../dataset/LJSpeech/data'
    file_path = os.path.join(data_root, 'metadata.csv')
    pre_pro.run(data_root)

