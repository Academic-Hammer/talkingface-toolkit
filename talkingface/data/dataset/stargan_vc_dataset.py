# 导入必要的模块和函数。

import os
from talkingface.data.dataprocess.stargan_vc_process import StarganAudio
from talkingface.data.dataset.dataset import Dataset
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle


class StarganDataset(Dataset):
    def __init__(self, config, file_list):
        self.config = config
        # 从配置中获取预处理数据的根目录，并列出其中的所有目录。
        feat_dirs = os.listdir(self.config['preprocessed_root'])

        with open(file_list) as f:
            text = f.readlines()
        file_list_ = []
        for line in text:
            line = line.strip()
            file_list_.append(line)
        # root下spk数量
        # 为每个目录中的文件创建一个文件名列表。
        self.filenames_all = [
            [
                os.path.join(self.config['preprocessed_root'],d,t) for t in sorted(os.listdir(os.path.join(self.config['preprocessed_root'], d))) if t in file_list_
            ] for d in feat_dirs
        ]
        self.n_domain = len(self.filenames_all)
        self.feat_dirs = feat_dirs
        # 创建 StarganAudio 对象以处理音频数据。
        self.process_audio = StarganAudio(config)

        self.melspec_scaler = StandardScaler()
        if os.path.exists(config['stat_path']):
            with open(config['stat_path'], mode='rb') as f:
                self.melspec_scaler = pickle.load(f)
        else:
            print("Melspec_scaler is None.")
            self.melspec_scaler = None
    def __len__(self):
        return min(len(f) for f in self.filenames_all)

    def __getitem__(self, idx):
        melspec_list = []
        # 遍历每个spk的平行wav。
        for d in range(self.n_domain):
            # 处理音频文件并获取梅尔频谱。
            # print(self.filenames_all[d][idx])
            melspec = self.process_audio.extract_melspec(self.filenames_all[d][idx])  # n_freq x n_time
            if self.melspec_scaler is not None:
                melspec = self.melspec_scaler.transform(melspec.T)
            # 将梅尔频谱添加到列表中。
            melspec_list.append(melspec.T)
        # 返回包含所有领域梅尔频谱的列表。
        return melspec_list
        # return {"melspec_list": melspec_list}
    
    def collate_fn(self, batch):
        #batch[b][s]: melspec (n_freq x n_frame)
        #b: batch size
        #s: speaker ID

        batchsize = len(batch)
        n_spk = len(batch[0])
        melspec_list = [[batch[b][s] for b in range(batchsize)] for s in range(n_spk)]
        #melspec_list[s][b]: melspec (n_freq x n_frame)
        #s: speaker ID
        #b: batch size

        n_freq = melspec_list[0][0].shape[0]

        X_list = []
        for s in range(n_spk):
            maxlen=0
            for b in range(batchsize):
                if maxlen<melspec_list[s][b].shape[1]:
                    maxlen = melspec_list[s][b].shape[1]
            maxlen = math.ceil(maxlen/4)*4
        
            X = np.zeros((batchsize,n_freq,maxlen))
            for b in range(batchsize):
                melspec = melspec_list[s][b]
                melspec = np.tile(melspec,(1,math.ceil(maxlen/melspec.shape[1])))
                X[b,:,:] = melspec[:,0:maxlen]
            #X = torch.tensor(X)
            X_list.append(X)

        return {"X_list": X_list}