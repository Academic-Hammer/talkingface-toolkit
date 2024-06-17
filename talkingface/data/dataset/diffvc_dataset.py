import os
import random
import numpy as np
import torch
import tgt

from talkingface.data.dataset.dataset import Dataset

random_seed = 37
n_mels = 80
train_frames = 128

# 返回测试说话人的列表。
def get_test_speakers():
    test_speakers = ['1401', '2238', '3723', '4014', '5126', 
                     '5322', '587', '6415', '8057', '8534']
    return test_speakers

# 返回VCTK数据集中未知说话人的列表。
def get_vctk_unseen_speakers():
    unseen_speakers = ['p252', 'p261', 'p241', 'p238', 'p243',
                       'p294', 'p334', 'p343', 'p360', 'p362']
    return unseen_speakers

# 返回VCTK数据集中未知句子的列表。
def get_vctk_unseen_sentences():
    unseen_sentences = ['001', '002', '003', '004', '005']
    return unseen_sentences

# 用于排除MFA不能识别某些单词的语音。
def exclude_spn(data_dir, spk, mel_ids):
    res = []
    for mel_id in mel_ids:
        textgrid = mel_id + '.TextGrid'
        t = tgt.io.read_textgrid(os.path.join(data_dir, 'textgrids', spk, textgrid))
        t = t.get_tier_by_name('phones')
        spn_found = False
        for i in range(len(t)):
            if t[i].text == 'spn':
                spn_found = True
                break
        if not spn_found:
            res.append(mel_id)
    return res

# 用于训练 "average voice" 编码器的 LibriTTS 数据集。
class VCEncDataset(Dataset):
    def __init__(self, config, datasplit, data_dir, exc_file, avg_type):
        super().__init__(config, datasplit)
        self.mel_x_dir = os.path.join(data_dir, 'mels')
        self.mel_y_dir = os.path.join(data_dir, 'mels_%s' % avg_type)

        self.test_speakers = get_test_speakers()
        self.speakers = [spk for spk in os.listdir(self.mel_x_dir) 
                         if spk not in self.test_speakers]
        with open(exc_file) as f:
            exceptions = f.readlines()
        self.exceptions = [e.strip() + '_mel.npy' for e in exceptions]
        self.test_info = []
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.mel_x_dir, spk))
            mel_ids = [m[:-8] for m in mel_ids if m not in self.exceptions]
            mel_ids = exclude_spn(data_dir, spk, mel_ids)
            self.train_info += [(m, spk) for m in mel_ids]
        for spk in self.test_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_x_dir, spk))
            mel_ids = [m[:-8] for m in mel_ids]
            self.test_info += [(m, spk) for m in mel_ids]
        print("Total number of test wavs is %d." % len(self.test_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, mel_id, spk):
        mel_x_path = os.path.join(self.mel_x_dir, spk, mel_id + '_mel.npy')
        mel_y_path = os.path.join(self.mel_y_dir, spk, mel_id + '_avgmel.npy')
        mel_x = np.load(mel_x_path)
        mel_y = np.load(mel_y_path)
        mel_x = torch.from_numpy(mel_x).float()
        mel_y = torch.from_numpy(mel_y).float()
        return (mel_x, mel_y)

    def __getitem__(self, index):
        mel_id, spk = self.train_info[index]
        mel_x, mel_y = self.get_vc_data(mel_id, spk)
        return {"x": mel_x, "y": mel_y}

    def __len__(self):
        return len(self.train_info)

    def get_test_dataset(self):
        pairs = []
        for i in range(len(self.test_info)):
            mel_id, spk = self.test_info[i]
            mel_x, mel_y = self.get_vc_data(mel_id, spk)
            pairs.append((mel_x, mel_y))
        return [{"x": pair[0], "y": pair[1]} for pair in pairs]
    
class VCDecDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, val_file, exc_file):
        self.mel_dir = os.path.join(data_dir, 'mels')
        #self.mel_dir = self.mel_dir.replace('\\','/')
        #self.mel_dir="./data/mels"
        self.emb_dir = os.path.join(data_dir, 'embeds')
        self.test_speakers = get_test_speakers()
        self.speakers = [spk for spk in os.listdir(self.mel_dir)
                         if spk not in self.test_speakers]
        self.speakers = [spk for spk in self.speakers
                         if len(os.listdir(os.path.join(self.mel_dir, spk))) >= 10]
        random.seed(random_seed)
        random.shuffle(self.speakers)
        with open(exc_file) as f:
            exceptions = f.readlines()
        self.exceptions = [e.strip() + '_mel.npy' for e in exceptions]
        with open(val_file) as f:
            valid_ids = f.readlines()
        self.valid_ids = set([v.strip() + '_mel.npy' for v in valid_ids])
        self.exceptions += self.valid_ids

        self.valid_info = [(v[:-8], v.split('_')[0]) for v in self.valid_ids]
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m not in self.exceptions]
            self.train_info += [(i[:-8], spk) for i in mel_ids]
        print("Total number of validation wavs is %d." % len(self.valid_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        print("Total number of training speakers is %d." % len(self.speakers))
        random.seed(random_seed)
        random.shuffle(self.train_info) 
    def get_vc_data(self, audio_info):
        audio_id, spk = audio_info
        mels = self.get_mels(audio_id, spk)
        embed = self.get_embed(audio_id, spk)
        return (mels, embed)

    def get_mels(self, audio_id, spk):
        mel_path = os.path.join(self.mel_dir, spk, audio_id + '_mel.npy')
        mels = np.load(mel_path)
        mels = torch.from_numpy(mels).float()
        return mels

    def get_embed(self, audio_id, spk):
        embed_path = os.path.join(self.emb_dir, spk, audio_id + '_embed.npy')
        embed = np.load(embed_path)
        embed = torch.from_numpy(embed).float()
        return embed

    def __getitem__(self, index):
        mels, embed = self.get_vc_data(self.train_info[index])
        item = {'mel': mels, 'c': embed}
        return item

    def __len__(self):
        return len(self.train_info)

    def get_valid_dataset(self):
        pairs = []
        for i in range(len(self.valid_info)):
            mels, embed = self.get_vc_data(self.valid_info[i])
            pairs.append((mels, embed))
        return pairs   

# 用于训练 "average voice" 编码器的 VCTK 数据集。
class VCTKEncDataset(Dataset):
    def __init__(self, config, datasplit):
        super().__init__(config, datasplit)
        data_dir=config['data_dir']
        exc_file=config['exc_file']
        avg_type=config['avg_type']
        self.mel_x_dir = os.path.join(data_dir, 'mels')
        self.mel_y_dir = os.path.join(data_dir, 'mels_%s' % avg_type)

        self.unseen_speakers = get_vctk_unseen_speakers()
        self.unseen_sentences = get_vctk_unseen_sentences()
        self.speakers = [spk for spk in os.listdir(self.mel_x_dir) 
                         if spk not in self.unseen_speakers]
        with open(exc_file) as f:
            exceptions = f.readlines()
        self.exceptions = [e.strip() + '_mel.npy' for e in exceptions]
        self.test_info = []
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.mel_x_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            mel_ids = [m[:-8] for m in mel_ids if m not in self.exceptions]
            mel_ids = exclude_spn(data_dir, spk, mel_ids)
            self.train_info += [(m, spk) for m in mel_ids]
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_x_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            mel_ids = [m[:-8] for m in mel_ids if m not in self.exceptions]
            self.test_info += [(m, spk) for m in mel_ids]
        print("Total number of test wavs is %d." % len(self.test_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, mel_id, spk):
        mel_x_path = os.path.join(self.mel_x_dir, spk, mel_id + '_mel.npy')
        mel_y_path = os.path.join(self.mel_y_dir, spk, mel_id + '_avgmel.npy')
        mel_x = np.load(mel_x_path)
        mel_y = np.load(mel_y_path)
        mel_x = torch.from_numpy(mel_x).float()
        mel_y = torch.from_numpy(mel_y).float()
        return (mel_x, mel_y)

    def __getitem__(self, index):
        mel_id, spk = self.train_info[index]
        mel_x, mel_y = self.get_vc_data(mel_id, spk)
        return {"x": mel_x, "y": mel_y}

    def __len__(self):
        return len(self.train_info)

    def get_test_dataset(self):
        pairs = []
        for i in range(len(self.test_info)):
            mel_id, spk = self.test_info[i]
            mel_x, mel_y = self.get_vc_data(mel_id, spk)
            pairs.append((mel_x, mel_y))
        return [{"x": pair[0], "y": pair[1]} for pair in pairs]

# 用于训练 speaker-conditional diffusion-based 解码器的 LibriTTS 数据集。(train)
class diffvcDataset(Dataset):
    def __init__(self, config, datasplit):
        super().__init__(config, datasplit)
        data_dir=config['data_dir']
        exc_file=config['exc_file']
        val_file=config['val_file']
        self.mel_dir = os.path.join(data_dir, 'mels')
        self.emb_dir = os.path.join(data_dir, 'embeds')
        self.test_speakers = get_test_speakers()
        self.speakers = [spk for spk in os.listdir(self.mel_dir)
                         if spk not in self.test_speakers]
        self.speakers = [spk for spk in self.speakers
                         if len(os.listdir(os.path.join(self.mel_dir, spk))) >= 10]
        random.seed(random_seed)
        random.shuffle(self.speakers)
        with open(exc_file) as f:
            exceptions = f.readlines()
        self.exceptions = [e.strip() + '_mel.npy' for e in exceptions]
        with open(val_file) as f:
            valid_ids = f.readlines()
        self.valid_ids = set([v.strip() + '_mel.npy' for v in valid_ids])
        self.exceptions += self.valid_ids

        self.valid_info = [(v[:-8], v.split('_')[0]) for v in self.valid_ids]
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m not in self.exceptions]
            self.train_info += [(i[:-8], spk) for i in mel_ids]
        print("Total number of validation wavs is %d." % len(self.valid_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        print("Total number of training speakers is %d." % len(self.speakers))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, audio_info):
        audio_id, spk = audio_info
        mels = self.get_mels(audio_id, spk)
        embed = self.get_embed(audio_id, spk)
        return (mels, embed)

    def get_mels(self, audio_id, spk):
        mel_path = os.path.join(self.mel_dir, spk, audio_id + '_mel.npy')
        mels = np.load(mel_path)
        mels = torch.from_numpy(mels).float()
        return mels

    def get_embed(self, audio_id, spk):
        embed_path = os.path.join(self.emb_dir, spk, audio_id + '_embed.npy')
        embed = np.load(embed_path)
        embed = torch.from_numpy(embed).float()
        return embed

    def __getitem__(self, index):
        mels, embed = self.get_vc_data(self.train_info[index])
        return {'mel': mels, 'c': embed}

    def __len__(self):
        return len(self.train_info)

    def get_valid_dataset(self):
        pairs = []
        for i in range(len(self.valid_info)):
            mels, embed = self.get_vc_data(self.valid_info[i])
            pairs.append((mels, embed))
        return [{"mel": pair[0], "c": pair[1]} for pair in pairs]
    



# 用于训练 speaker-conditional diffusion-based 解码器的 VCTK 数据集。
class VCTKDecDataset(Dataset):
    def __init__(self,config, datasplit, data_dir):
        super().__init__(config, datasplit)
        self.mel_dir = os.path.join(data_dir, 'mels')
        self.emb_dir = os.path.join(data_dir, 'embeds')
        self.unseen_speakers = get_vctk_unseen_speakers()
        self.unseen_sentences = get_vctk_unseen_sentences()
        self.speakers = [spk for spk in os.listdir(self.mel_dir)
                         if spk not in self.unseen_speakers]
        random.seed(random_seed)
        random.shuffle(self.speakers)
        self.train_info = []
        for spk in self.speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            self.train_info += [(i[:-8], spk) for i in mel_ids]
        self.valid_info = []
        for spk in self.unseen_speakers:
            mel_ids = os.listdir(os.path.join(self.mel_dir, spk))
            mel_ids = [m for m in mel_ids if m.split('_')[1] not in self.unseen_sentences]
            self.valid_info += [(i[:-8], spk) for i in mel_ids]
        print("Total number of validation wavs is %d." % len(self.valid_info))
        print("Total number of training wavs is %d." % len(self.train_info))
        print("Total number of training speakers is %d." % len(self.speakers))
        random.seed(random_seed)
        random.shuffle(self.train_info)

    def get_vc_data(self, audio_info):
        audio_id, spk = audio_info
        mels = self.get_mels(audio_id, spk)
        embed = self.get_embed(audio_id, spk)
        return (mels, embed)

    def get_mels(self, audio_id, spk):
        mel_path = os.path.join(self.mel_dir, spk, audio_id + '_mel.npy')
        mels = np.load(mel_path)
        mels = torch.from_numpy(mels).float()
        return mels

    def get_embed(self, audio_id, spk):
        embed_path = os.path.join(self.emb_dir, spk, audio_id + '_embed.npy')
        embed = np.load(embed_path)
        embed = torch.from_numpy(embed).float()
        return embed

    def __getitem__(self, index):
        mels, embed = self.get_vc_data(self.train_info[index])
        return {'mel': mels, 'c': embed}

    def __len__(self):
        return len(self.train_info)

    def get_valid_dataset(self):
        pairs = []
        for i in range(len(self.valid_info)):
            mels, embed = self.get_vc_data(self.valid_info[i])
            pairs.append((mels, embed))
        return [{"mel": pair[0], "c": pair[1]} for pair in pairs]

class VCDecBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        mels1 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        mels2 = torch.zeros((B, n_mels, train_frames), dtype=torch.float32)
        max_starts = [max(item['mel'].shape[-1] - train_frames, 0)
                      for item in batch]
        starts1 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        starts2 = [random.choice(range(m)) if m > 0 else 0 for m in max_starts]
        mel_lengths = []
        for i, item in enumerate(batch):
            mel = item['mel']
            if mel.shape[-1] < train_frames:
                mel_length = mel.shape[-1]
            else:
                mel_length = train_frames
            mels1[i, :, :mel_length] = mel[:, starts1[i]:starts1[i] + mel_length]
            mels2[i, :, :mel_length] = mel[:, starts2[i]:starts2[i] + mel_length]
            mel_lengths.append(mel_length)
        mel_lengths = torch.LongTensor(mel_lengths)
        embed = torch.stack([item['c'] for item in batch], 0)
        return {'mel1': mels1, 'mel2': mels2, 'mel_lengths': mel_lengths, 'c': embed}

    
