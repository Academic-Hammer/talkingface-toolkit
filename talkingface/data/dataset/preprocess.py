# -*- coding: utf-8 -*-

import os
import sys

import json
import pickle
import librosa
import numpy as np
import python_speech_features
from pathlib import Path


def audio2mfcc(audio_file, save):
    try:
        speech, sr = librosa.load(audio_file, sr=16000)
        #  mfcc = python_speech_features.mfcc(speech ,16000,winstep=0.01)
        speech = np.insert(speech, 0, np.zeros(1920))
        speech = np.append(speech, np.zeros(1920))
        mfcc = python_speech_features.mfcc(speech, 16000, winstep=0.01)
        if not os.path.exists(save):
            os.makedirs(save)
        time_len = mfcc.shape[0]

        for input_idx in range(int((time_len - 28) / 4) + 1):
            #   target_idx = input_idx + sample_delay #14

            input_feat = mfcc[4 * input_idx:4 * input_idx + 28, :]

            np.save(os.path.join(save, str(input_idx) + '.npy'), input_feat)

        print(input_idx)
    except Exception as e:
        print(f"发生了异常: {e}")

MEAD = {'angry':0, 'contempt':1, 'disgusted':2, 'fear':3, 'happy':4, 'neutral':5,
        'sad':6, 'surprised':7}
filepath = '/data/tuluwei/nlp1/dataset/train/landmark/dataset_M030/landmark'
save_path = '/data/tuluwei/nlp1/dataset/train/MFCC/M030/'
pathDir = os.listdir(filepath)
allp=[]
for i in range(len(pathDir)):
    emotion = pathDir[i]
    path = os.path.join(filepath,emotion)
    Dir = os.listdir(path)
    for j in range(len(Dir)):
        audio_file = os.path.join(path,Dir[j])
        index = Dir[j].split('.')[0]
        # print(index,Dir[j],emotion,"+=====")
        # emotion=emotion.split('')
        save = os.path.join(save_path,emotion+'_'+index)
        audio2mfcc(audio_file, save)
        print(i, emotion, j, index)

#create list
train_list = []
val_list = []
a = Path(save_path)
print(a)
for b in a.iterdir():
    for c in b.iterdir():
        print(b.name)
        if int(b.name.split('_')[-2]) < 10:
            val_list.append(b.name+'/'+c.name)
        else:
            train_list.append(b.name+'/'+c.name)

with open('dataset/train/mfcc_data/train_M030.pkl', 'wb') as f:
    pickle.dump(train_list, f)
with open('dataset/train/mfcc_data/val_M030.pkl', 'wb') as f:
    pickle.dump(val_list, f)

'''
allp=[]
for allDir in pathDir:
    if (allDir.split('_')[2] == '3'):

        child = os.path.join(filepath, allDir)
        for i in os.listdir(child):

            allp.append(allDir+'/'+i)
    if (int(allDir.split('_')[1]) > 61):
        child = os.path.join(filepath, allDir)
        for i in os.listdir(child):

            allp.append(allDir+'/'+i)

with open('/home/thea/data/MEAD/SER_oneintense/list.pkl', 'wb') as f:
    pickle.dump(allp, f)
'''