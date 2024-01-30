import torch
import numpy
import time, pdb, argparse, subprocess, os, math, glob
import cv2
import python_speech_features
from tqdm import tqdm
from scipy import signal
from scipy.io import wavfile
from talkingface.evaluator.metric_models import *
from shutil import rmtree
from skimage.metrics import structural_similarity as ssim
from talkingface.evaluator.base_metric import AbstractMetric, SyncMetric, VideoQMetric
from talkingface.utils.logger import set_color

class LSE(SyncMetric):

    '''
    '''
    def __init__(self, config, num_layers_in_fc_layers = 1024):
        super(LSE, self).__init__(config)
        self.config = config
        self.syncnet = S(num_layers_in_fc_layers = num_layers_in_fc_layers)

    def metric_info(self, videofile):
        self.loadParameters(self.config['lse_checkpoint_path'])
        self.syncnet.to(self.config["device"])
        self.syncnet.eval()

        if os.path.exists(os.path.join(self.config['temp_dir'], self.config['lse_reference_dir'])):
            rmtree(os.path.join(self.config['temp_dir'], self.config['lse_reference_dir']))
        
        os.makedirs(os.path.join(self.config['temp_dir'], self.config['lse_reference_dir']))

        command = ("ffmpeg -loglevel error -y -i %s -threads 1 -f image2 %s" % (videofile,os.path.join(self.config['temp_dir'], self.config['lse_reference_dir'],'%06d.jpg'))) 
        output = subprocess.call(command, shell=True, stdout=None)

        command = ("ffmpeg -loglevel error -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (videofile,os.path.join(self.config['temp_dir'], self.config['lse_reference_dir'],'audio.wav'))) 
        output = subprocess.call(command, shell=True, stdout=None)

        images = []
        
        flist = glob.glob(os.path.join(self.config['temp_dir'], self.config['lse_reference_dir'],'*.jpg'))
        flist.sort()

        for fname in flist:
            img_input = cv2.imread(fname)
            img_input = cv2.resize(img_input, (224, 224))
            images.append(img_input)

        im = numpy.stack(images, axis=3)
        im = numpy.expand_dims(im, axis=0)
        im = numpy.transpose(im, (0, 3, 4, 1, 2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        sample_rate, audio = wavfile.read(os.path.join(self.config['temp_dir'], self.config['lse_reference_dir'],'audio.wav'))
        mfcc = zip(*python_speech_features.mfcc(audio,sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])

        cc = numpy.expand_dims(numpy.expand_dims(mfcc,axis=0),axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
        
        # ========== ==========
        # Check audio and video input length
        # ========== ==========

        #if (float(len(audio))/16000) != (float(len(images))/25) :
        #    print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different."%(float(len(audio))/16000,float(len(images))/25))

        min_length = min(len(images),math.floor(len(audio)/640))
        
        # ========== ==========
        # Generate video and audio feats
        # ========== ==========

        lastframe = min_length-5
        im_feat = []
        cc_feat = []

        tS = time.time()
        for i in range(0,lastframe,self.config['evaluate_batch_size']):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe, i+self.config['evaluate_batch_size'])) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.syncnet.forward_lip(im_in.cuda())
            im_feat.append(im_out.data.cpu())

            cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe, i+self.config['evaluate_batch_size'])) ]
            cc_in = torch.cat(cc_batch,0)
            cc_out  = self.syncnet.forward_aud(cc_in.cuda())
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0)

        # ========== ==========
        # Compute offset
        # ========== ==========
            
        #print('Compute time %.3f sec.' % (time.time()-tS))

        dists = self.calc_pdist(im_feat,cc_feat,vshift=self.config['vshift'])
        mdist = torch.mean(torch.stack(dists,1),1)

        minval, minidx = torch.min(mdist,0)

        offset = self.config['vshift']-minidx
        conf   = torch.median(mdist) - minval

        fdist   = numpy.stack([dist[minidx].numpy() for dist in dists])
        # fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
        fconf   = torch.median(mdist).numpy() - fdist
        fconfm  = signal.medfilt(fconf,kernel_size=9)
        
        numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        return conf.numpy(), minval.numpy()



    def calc_pdist(self, feat1, feat2, vshift=10):
    
        win_size = vshift*2+1

        feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))

        dists = []

        for i in range(0,len(feat1)):

            dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))

        return dists
    
    def calculate_metric(self, dataobject):
        video_list = self.get_videolist(dataobject)

        LSE_dict = {}

        iter_data = (
            tqdm(
                video_list,
                total=len(video_list),
                desc=set_color("calculate for lip-audio sync", "yellow")
            )
            if self.config['show_progress']
            else video_list
        )

        for video in iter_data:
            LSE_C, LSE_D = self.metric_info(video)
            if "LSE_C" not in LSE_dict:
                LSE_dict["LSE_C"] = [LSE_C]
            else:
                LSE_dict["LSE_C"].append(LSE_C)
            if "LSE_D" not in LSE_dict:
                LSE_dict["LSE_D"] = [LSE_D]
            else:
                LSE_dict["LSE_D"].append(LSE_D)

        return {"LSE-C: {}".format(sum(LSE_dict["LSE_C"])/len(LSE_dict["LSE_C"])), "LSE-D: {}".format(sum(LSE_dict["LSE_D"])/len(LSE_dict["LSE_D"]))}

    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage)

        self_state = self.syncnet.state_dict()

        for name, param in loaded_state.items():

            self_state[name].copy_(param)


class SSIM(VideoQMetric):

    metric_need = ["generated_video", "real_video"]

    def __init__(self, config):
        super(SSIM, self).__init__(config)
        self.config = config

    def metric_info(self, g_videofile, r_videofile):
        g_frames = []
        g_video = cv2.VideoCapture(g_videofile)
        while g_video.isOpened():
            ret, frame = g_video.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))  
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g_frames.append(gray)
        g_video.release()

        r_frames = []
        r_video = cv2.VideoCapture(r_videofile)
        while r_video.isOpened():
            ret, frame = r_video.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))  
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            r_frames.append(gray)
        r_video.release()

        min_frames = min(len(g_frames), len(r_frames))

        g_frames = g_frames[:min_frames]
        r_frames = r_frames[:min_frames]

        ssim_scores = []
        for frame1, frame2 in zip(g_frames, r_frames):
            # 计算两帧之间的SSIM
            score, _ = ssim(frame1, frame2, full=True)
            ssim_scores.append(score)
        
        return numpy.mean(ssim_scores)
        





    def calculate_metric(self, dataobject):

        pair_list = self.get_videopair(dataobject)

        ssim_score_total = []
        print(pair_list)
        iter_data = (
            tqdm(
                pair_list,
                total=len(pair_list),
                desc=set_color("calculate for video quality ssim", "yellow")
            )
            if self.config['show_progress']
            else pair_list
        )
        for pair in iter_data:
            g_video = pair[0]
            r_video = pair[1]
            ssim_score = self.metric_info(g_video, r_video)
            ssim_score_total.append(ssim_score)

        return sum(ssim_score_total)/len(ssim_score_total)
