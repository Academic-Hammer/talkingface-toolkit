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
from skimage import transform as tf
import python_speech_features
import dlib
import imageio
from pathlib import Path


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

class meadPreprocess:
    def __init__(self, config):
        self.config = config
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('checkpoints/EAMM/shape_predictor_68_face_landmarks.dat')
        
    def save(self, path, frames, format):
        if format == '.mp4':
            imageio.mimsave(path, frames)
        elif format == '.png':
            if not os.path.exists(path):
                os.makedirs(path)
            for j, frame in enumerate(frames):
                cv2.imwrite(path+'/'+str(j)+'.png',frame)
        else:
            print ("Unknown format %s" % format)
            exit()
            
    def shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        # return the list of (x, y)-coordinates
        return coords
            
    def crop_image(self, image_path, out_path):
        template = np.load('./M003_template.npy')
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)  #detect human face
        if len(rects) != 1:
            return 0
        for (j, rect) in enumerate(rects):
            shape = self.predictor(gray, rect) #detect 68 points
            shape = self.shape_to_np(shape)
        pts2 = np.float32(template[:47,:])
        # pts2 = np.float32(template[17:35,:])
        # pts1 = np.vstack((landmark[27:36,:], landmark[39,:],landmark[42,:],landmark[45,:]))
        pts1 = np.float32(shape[:47,:]) #eye and nose
        # pts1 = np.float32(landmark[17:35,:])
        tform = tf.SimilarityTransform()
        tform.estimate( pts2, pts1) #Set the transformation matrix with the explicit parameters.
        dst = tf.warp(image, tform, output_shape=(256, 256))
        dst = np.array(dst * 255, dtype=np.uint8)
        cv2.imwrite(out_path,dst)

    def crop_image_tem(self, video_path, out_path):
        """
        video alignment
        """
        image_all = []
        videoCapture = cv2.VideoCapture(video_path)
        success, frame = videoCapture.read()
        n = 0
        while success :
            image_all.append(frame)
            n = n + 1
            success, frame = videoCapture.read()   
        if len(image_all)!=0 :
            template = np.load('checkpoints/EAMM/M003_template.npy')
            image=image_all[0]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 1)  #detect human face
            if len(rects) != 1:
                return 0
            for (j, rect) in enumerate(rects):
                shape = self.predictor(gray, rect) #detect 68 points
                shape = self.shape_to_np(shape)
            pts2 = np.float32(template[:47,:])
            pts1 = np.float32(shape[:47,:]) #eye and nose
            tform = tf.SimilarityTransform()
            tform.estimate( pts2, pts1) #Set the transformation matrix with the explicit parameters.
            out = []
            for i in range(len(image_all)):
                image = image_all[i]
                dst = tf.warp(image, tform, output_shape=(256, 256))
                dst = np.array(dst * 255, dtype=np.uint8)
                out.append(dst)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            self.save(out_path,out,'.png')
    
    def proc_audio(self, src_mouth_path, dst_audio_path):
        audio_command = 'ffmpeg -i \"{}\" -loglevel error -y -f wav -acodec pcm_s16le ' \
                        '-ar 16000 \"{}\"'.format(src_mouth_path, dst_audio_path)
        # os.system(audio_command)
        subprocess.call(audio_command, shell=True)
    
    def audio2mfcc(self, audio_file, save, name):
        speech, sr = librosa.load(audio_file, sr=16000)
        speech = np.insert(speech, 0, np.zeros(1920))
        speech = np.append(speech, np.zeros(1920))
        mfcc = python_speech_features.mfcc(speech,16000,winstep=0.01)
        if not os.path.exists(save):
            os.makedirs(save)
        time_len = mfcc.shape[0]
        mfcc_all = []
        for input_idx in range(int((time_len-28)/4)+1):
            input_feat = mfcc[4*input_idx:4*input_idx+28,:]
            mfcc_all.append(input_feat)
        np.save(os.path.join(save,name+'.npy'), mfcc_all)
        # print(input_idx)
    
    def prepare_3dpose(self, filepath, save_path):
        from talkingface.utils.pose_3ddfa.FaceBoxes import FaceBoxes
        from talkingface.utils.pose_3ddfa.TDDFA import TDDFA
        from talkingface.utils.pose_3ddfa.utils.pose import get_pose
        import yaml
        
        pathDir = os.listdir(filepath)
        cfg = yaml.load(open('./talkingface/properties/model/EAMM/mb1_120x120.yml'), Loader=yaml.SafeLoader)
        
        for i in range(len(pathDir)):
            image= cv2.imread(os.path.join(filepath,pathDir[i]))
            
            # Init FaceBoxes and TDDFA, recommend using onnx flag
            tddfa = TDDFA(gpu_mode=False, **cfg)
            face_boxes = FaceBoxes()
            
            # Detect faces, get 3DMM params and roi boxes
            boxes = face_boxes(image)
            n = len(boxes)
            if n == 0:
                print(f'No face detected, exit')
                return None
            # print(f'Detect {n} faces')

            param_lst, roi_box_lst = tddfa(image, boxes)
            ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
            all_pose = get_pose(image, param_lst, ver_lst, show_flag=False, wfp=None, wnp = None)
            pose = all_pose.reshape(1,7)

            save = Path(save_path)
            if not save.exists():
                save.mkdir(parents=True, exist_ok=True)
            np.save((save / (pathDir[i].split('.')[0]+'.npy')),pose)
            print(i,pathDir[i])
        

    def run(self):
        print(f'Started processing for {self.config["data_root"]} with {self.config["ngpu"]} GPUs')
        
        data_root = Path(self.config['data_root'])
        preprocessed_root = Path(self.config['_preprocessed_root'])
        
        vfilelist = list((data_root / 'crop').glob("*.mp4"))
        print('video alignment...')
        for vfile in tqdm(vfilelist):    
            try:
                odir = (preprocessed_root / 'crop' / vfile.stem)
                if odir.exists():
                    continue
                self.crop_image_tem(str(vfile), str(odir))
            except KeyboardInterrupt:
                exit(0)
            except:
                traceback.print_exc()
                continue
        
        afilelist = list((data_root / 'audio').glob("*.m4a"))
        print('audio2mfcc...')
        for afile in tqdm(afilelist):
            try:
                save_path = (preprocessed_root / 'MEAD_MFCC')
                if (save_path / (afile.stem + '.npy')).exists():
                    continue
                self.audio2mfcc(str(afile), str(save_path), afile.stem)
            except KeyboardInterrupt:
                exit(0)
            except:
                traceback.print_exc()
                continue
        
        
        pfilelist = list((preprocessed_root / 'crop').glob('*'))
        # print(pfilelist)
        print('3d pose...')
        for pfile in tqdm(pfilelist):
            try:
                save_path = (preprocessed_root / 'pose' / pfile.stem)
                print(save_path)
                if save_path.exists():
                    continue
                self.prepare_3dpose(str(pfile), str(save_path))
            except KeyboardInterrupt:
                exit(0)
            except:
                traceback.print_exc()
                continue
        
        
