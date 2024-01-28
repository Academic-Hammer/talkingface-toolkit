from talkingface.data.dataset.dataset import Dataset
import torch.nn.functional as F
import numpy as np
import torch
import os
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import time
import os

from tqdm import tqdm

from talkingface.utils import text_to_sequence


class FastSpeechDataset(Dataset):
    def __init__(self, config, datasplit):
        self.config = config
        self.datasplit=datasplit
        self.buffer = self.get_data_to_buffer()

    def get_data_to_buffer(self):
        buffer = list()
        txt_data = process_text(self.datasplit)

        start = time.perf_counter()
        for text,i in tqdm(txt_data,total=len(txt_data)):
            mel_gt_name = os.path.join(
                self.config['preprocessed_root'],'mels', "LJSpeech-mel-%05d.npy" % i)
            mel_gt_target = np.load(mel_gt_name)
            duration = np.load(os.path.join(
                self.config['preprocessed_root'],'alignments', str(i-1) + ".npy"))
            character = text
            character = np.array(
                text_to_sequence(character, self.config['text_cleaners']))

            character = torch.from_numpy(character)
            duration = torch.from_numpy(duration)
            mel_gt_target = torch.from_numpy(mel_gt_target)

            buffer.append({"text": character, "duration": duration,
                           "mel_target": mel_gt_target})

        end = time.perf_counter()
        print("cost {:.2f}s to load all data into buffer.".format(end - start))
        return buffer

    def collate_fn(self,batch):
        len_arr = np.array([d["text"].size(0) for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = batchsize // self.config['batch_expand_size']

        cut_list = list()
        for i in range(self.config['batch_expand_size']):
            cut_list.append(index_arr[i * real_batchsize:(i + 1) * real_batchsize])

        output = list()
        for i in range(self.config['batch_expand_size']):
            output.append(reprocess_tensor(batch, cut_list[i]))

        return output

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]


def reprocess_tensor(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    durations = [batch[ind]["duration"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i + 1 for i in range(int(length_src_row))],
                              (0, max_len - int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i + 1 for i in range(int(length_mel_row))],
                              (0, max_mel_len - int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    mel_targets = pad_2D_tensor(mel_targets)

    out = {"text": texts,
           "mel_target": mel_targets,
           "duration": durations,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_len": max_mel_len}

    return out



def pad_1D_tensor(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D_tensor(inputs, maxlen=None):

    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len - x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output

def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt_list=[]
        for line in f.readlines():
            parts = line.strip().split('|')
            txt_list.append([parts[2],int(parts[1])])
        return txt_list


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
        'text_cleaners': ['english_cleaners'],
        'preprocessed_root':'../../../dataset/LJSpeech/ljspeech_preprocessed',
        "train_filelist": '../../../dataset/LJSpeech/filelist/train.txt',
        "test_filelist": '../../../dataset/LJSpeech/filelist/test.txt',
        "val_filelist": '../../../dataset/LJSpeech/filelist/val.txt',
    }
    ds=FastSpeechDataset(config,config['test_filelist'])
    for real_wav in ds.buffer:
        print(real_wav['mel_target'])


