import os
import cv2
import numpy as np
import subprocess
from tqdm import tqdm
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from talkingface.utils import face_detection
import traceback
import librosa
import librosa.filters
from scipy import signal
from scipy.io import wavfile
from text import _clean_text

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

import os
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from text import _clean_text

class FastSpeech2Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["corpus_path"]
        self.out_dir = config["path"]["raw_path"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
        self.cleaners = config["preprocessing"]["text"]["text_cleaners"]
        self.speaker = "LJSpeech"

    def prepare_align(self):
        with open(os.path.join(self.in_dir, "metadata.csv"), encoding="utf-8") as f:
            for line in tqdm(f):
                parts = line.strip().split("|")
                base_name = parts[0]
                text = parts[2]
                text = _clean_text(text, self.cleaners)

                wav_path = os.path.join(self.in_dir, "wavs", "{}.wav".format(base_name))
                if os.path.exists(wav_path):
                    os.makedirs(os.path.join(self.out_dir, self.speaker), exist_ok=True)
                    wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
                    wav = wav / max(abs(wav)) * self.max_wav_value
                    wavfile.write(
                        os.path.join(self.out_dir, self.speaker, "{}.wav".format(base_name)),
                        self.sampling_rate,
                        wav.astype(np.int16),
                    )
                    with open(
                        os.path.join(self.out_dir, self.speaker, "{}.lab".format(base_name)),
                        "w",
                    ) as f1:
                        f1.write(text)

    def run(self):
        self.prepare_align()
