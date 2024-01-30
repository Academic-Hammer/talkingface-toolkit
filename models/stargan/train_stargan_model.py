import itertools
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import joblib
import logging
import os
import warnings
import json

import librosa
import soundfile as sf
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pickle
from dataset import MultiDomain_Dataset, collate_fn
import net

def walk_files(root, extension):
    for path, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(extension):
                yield os.path.join(path, file)

def logmelfilterbank(audio,
                     sampling_rate,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     num_mels=80,
                     fmin=None,
                     fmax=None,
                     eps=1e-10,
                     ):
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    # mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)
    mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)

    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

def extract_melspec(src_filepath, dst_filepath, kwargs):
    # print('what')
    # try:
    warnings.filterwarnings('ignore')

    trim_silence = kwargs['trim_silence']
    top_db = kwargs['top_db']
    flen = kwargs['flen']
    fshift = kwargs['fshift']
    fmin = kwargs['fmin']
    fmax = kwargs['fmax']
    num_mels = kwargs['num_mels']
    fs = kwargs['fs']

    audio, fs_ = sf.read(src_filepath)
    if trim_silence:
        # print('trimming.')
        audio, _ = librosa.effects.trim(audio, top_db=top_db, frame_length=2048, hop_length=512)
    # print('xzz')
    if fs != fs_:
        # print('resampling.')
        # audio = librosa.resample(audio, fs_, fs)
        audio = librosa.resample(y=audio, orig_sr=fs_, target_sr=fs)

    melspec_raw = logmelfilterbank(audio,fs, fft_size=flen,hop_size=fshift,
                                    fmin=fmin, fmax=fmax, num_mels=num_mels)
    melspec_raw = melspec_raw.astype(np.float32)
    melspec_raw = melspec_raw.T # n_mels x n_frame
    if not os.path.exists(os.path.dirname(dst_filepath)):
        os.makedirs(os.path.dirname(dst_filepath), exist_ok=True)
    with h5py.File(dst_filepath, "w") as f:
        f.create_dataset("melspec", data=melspec_raw)
    logging.info(f"{dst_filepath}...[{melspec_raw.shape}].")


def makedirs_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def comb(N, r):
    iterable = list(range(0, N))
    return list(itertools.combinations(iterable, 2))


src = './dataset/vctk/data'
dst = './models/stargan/dump/arctic/feat/train'
ext = '.wav'

fmt = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
datafmt = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datafmt)

data_config = {
    'num_mels': 80,
    'fs': 16000,
    'flen': 1024,
    'fshift': 128,
    'fmin': 80,
    'fmax': 7600,
    'trim_silence': True,
    'top_db': 30
}
configpath = './models/stargan/dump/arctic/data_config.json'
if not os.path.exists(os.path.dirname(configpath)):
    os.makedirs(os.path.dirname(configpath))
with open(configpath, 'w') as outfile:
    json.dump(data_config, outfile, indent=4)

fargs_list = [
    [
        f,
        f.replace(src, dst).replace(ext, ".h5"),
        data_config,
    ]
    for f in walk_files(src, ext)
]
print(fargs_list)
results = joblib.Parallel(n_jobs=1)(
    joblib.delayed(extract_melspec)(*f) for f in tqdm(fargs_list)
)


def walk_files(root, extension):
    for path, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(extension):
                yield os.path.join(path, file)


def melspec_transform(melspec, scaler):
    melspec = scaler.transform(melspec.T)
    melspec = melspec.T
    return melspec

def normalize_features(src_filepath, dst_filepath, melspec_transform):
    try:
        with h5py.File(src_filepath, "r") as f:
            melspec = f["melspec"][()]
        melspec = melspec_transform(melspec)

        if not os.path.exists(os.path.dirname(dst_filepath)):
            os.makedirs(os.path.dirname(dst_filepath), exist_ok=True)
        with h5py.File(dst_filepath, "w") as f:
            f.create_dataset("melspec", data=melspec)

        # logging.info(f"{dst_filepath}...[{melspec.shape}].")
        return melspec.shape

    except:
        logging.info(f"{dst_filepath}...failed.")

# parser.add_argument('--src', type=str,
#                     default='./models/stargan/dump/arctic/feat/train',
#                     help='data folder that contains the raw features extracted from VoxCeleb2 Dataset')
# parser.add_argument('--dst', type=str, default='./models/stargan/dump/arctic/norm_feat/train',
#                     help='data folder where the normalized features are stored')
# parser.add_argument('--stat', type=str, default='./models/stargan/dump/arctic/stat.pkl',
#                     help='state file used for normalization')
# parser.add_argument('--ext', type=str, default='.h5')


src = './models/stargan/dump/arctic/feat/train'
dst = './models/stargan/dump/arctic/norm_feat/train'
ext = '.h5'
stat_filepath = './models/stargan/dump/arctic/stat.pkl'

fmt = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
datafmt = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datafmt)

melspec_scaler = StandardScaler()
if os.path.exists(stat_filepath):
    with open(stat_filepath, mode='rb') as f:
        melspec_scaler = pickle.load(f)
    print('Loaded mel-spectrogram statistics successfully.')
else:
    print('Stat file not found.')

root = src
fargs_list = [
    [
        f,
        f.replace(src, dst),
        lambda x: melspec_transform(x, melspec_scaler),
    ]
    for f in walk_files(root, ext)
]
print(fargs_list)
results = joblib.Parallel(n_jobs=1)(
    joblib.delayed(normalize_features)(*f) for f in tqdm(fargs_list)
)


def walk_files(root, extension):
    for path, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(extension):
                yield os.path.join(path, file)

def read_melspec(filepath):
    with h5py.File(filepath, "r") as f:
        melspec = f["melspec"][()]  # n_mels x n_frame
    # import pdb;pdb.set_trace() # Breakpoint
    return melspec


def compute_statistics(src, stat_filepath):
    melspec_scaler = StandardScaler()

    filepath_list = list(walk_files(src, '.h5'))
    for filepath in tqdm(filepath_list):
        melspec = read_melspec(filepath)
        # import pdb;pdb.set_trace() # Breakpoint
        melspec_scaler.partial_fit(melspec.T)

    with open(stat_filepath, mode='wb') as f:
        pickle.dump(melspec_scaler, f)


fmt = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
datafmt = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datafmt)

src = './models/stargan/dump/arctic/feat/train'
stat_filepath = './models/stargan/dump/arctic/stat.pkl'
if not os.path.exists(os.path.dirname(stat_filepath)):
    os.makedirs(os.path.dirname(stat_filepath))

compute_statistics(src, stat_filepath)

gpu = -1
data_rootdir = './models/stargan/dump/arctic/norm_feat/train'
epochs = 2000
snapshot = 200
batch_size = 12
num_mels = 80
arch_type = 'conv'
loss_type = 'wgan'
zdim = 16
hdim = 64
mdim = 32
sdim = 16
lrate_g = 0.0005
lrate_d = 5e-6
gradient_clip = 1.0
w_adv = 1.0
w_grad = 1.0
w_cls = 1.0
w_cyc = 1.0
w_rec = 1.0
normtype = 'IN'
src_conditioning = 0
resume = 0
model_rootdir = './model/arctic/'
log_dir = './logs/arctic/'
experiment_name = 'experiment1'




def train_stargan():
    # Set up GPU
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device('cuda:%d' % gpu)
    else:
        device = torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)


    spk_list = sorted(os.listdir(data_rootdir))
    n_spk = len(spk_list)
    melspec_dirs = [os.path.join(data_rootdir, spk) for spk in spk_list]

    config = {
        'num_mels': num_mels,
        'arch_type': arch_type,
        'loss_type': loss_type,
        'zdim': zdim,
        'hdim': hdim,
        'mdim': mdim,
        'sdim': sdim,
        'w_adv': 1.0,
        'w_grad': 1.0,
        'w_cls': 1.0,
        'w_cyc': 1.0,
        'w_rec': 1.0,
        'lrate_g': lrate_g,
        'lrate_d': lrate_d,
        'gradient_clip': 1.0,
        'normtype': normtype,
        'epochs': epochs,
        'BatchSize': batch_size,
        'n_spk': n_spk,
        'spk_list': spk_list,
        'src_conditioning': src_conditioning
    }

    model_dir = os.path.join(model_rootdir, experiment_name)
    makedirs_if_not_exists(model_dir)
    log_path = os.path.join(log_dir, experiment_name, 'train_{}.log'.format(experiment_name))

    # Save configuration as a json file
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'w') as outfile:
        json.dump(config, outfile, indent=4)

    if arch_type == 'conv':
        gen = net.Generator1(num_mels, n_spk, zdim, hdim, sdim, normtype, src_conditioning)
    elif arch_type == 'rnn':
        net.Generator2(num_mels, n_spk, zdim, hdim, sdim, src_conditioning=src_conditioning)
    dis = net.Discriminator1(num_mels, n_spk, mdim, normtype)
    models = {
        'gen': gen,
        'dis': dis
    }
    models['stargan'] = net.StarGAN(models['gen'], models['dis'], n_spk, loss_type)

    optimizers = {
        'gen': optim.Adam(models['gen'].parameters(), lr=lrate_g, betas=(0.9, 0.999)),
        'dis': optim.Adam(models['dis'].parameters(), lr=lrate_d, betas=(0.5, 0.999))
    }

    for tag in ['gen', 'dis']:
        models[tag].to(device).train(mode=True)

    train_dataset = MultiDomain_Dataset(*melspec_dirs)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              # num_workers=os.cpu_count(),
                              drop_last=True,
                              collate_fn=collate_fn)

    fmt = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    datafmt = '%m/%d/%Y %I:%M:%S'
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    logging.basicConfig(filename=log_path, filemode='a', level=logging.INFO, format=fmt, datefmt=datafmt)
    writer = SummaryWriter(os.path.dirname(log_path))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for tag in ['gen', 'dis']:
        checkpointpath = os.path.join(model_dir, '{}.{}.pt'.format(resume, tag))
        if os.path.exists(checkpointpath):
            checkpoint = torch.load(checkpointpath, map_location=device)
            models[tag].load_state_dict(checkpoint['model_state_dict'])
            optimizers[tag].load_state_dict(checkpoint['optimizer_state_dict'])
            print('{} loaded successfully.'.format(checkpointpath))

    w_adv = config['w_adv']
    w_grad = config['w_grad']
    w_cls = config['w_cls']
    w_cyc = config['w_cyc']
    w_rec = config['w_rec']
    gradient_clip = config['gradient_clip']

    print("===================================Training Started===================================")
    n_iter = 0
    for epoch in range(resume + 1, epochs + 1):
        b = 0
        for X_list in train_loader:
            n_spk = len(X_list)
            xin = []
            for s in range(n_spk):
                xin.append(torch.tensor(X_list[s]).to(device, dtype=torch.float))

            # List of speaker pairs
            spk_pair_list = comb(n_spk, 2)
            n_spk_pair = len(spk_pair_list)

            gen_loss_mean = 0
            dis_loss_mean = 0
            advloss_d_mean = 0
            gradloss_d_mean = 0
            advloss_g_mean = 0
            clsloss_d_mean = 0
            clsloss_g_mean = 0
            cycloss_mean = 0
            recloss_mean = 0
            # Iterate through all speaker pairs
            for m in range(n_spk_pair):
                s0 = spk_pair_list[m][0]
                s1 = spk_pair_list[m][1]

                AdvLoss_g, ClsLoss_g, CycLoss, RecLoss = models['stargan'].calc_gen_loss(xin[s0], xin[s1], s0, s1)
                gen_loss = (w_adv * AdvLoss_g + w_cls * ClsLoss_g + w_cyc * CycLoss + w_rec * RecLoss)

                models['gen'].zero_grad()
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(models['gen'].parameters(), gradient_clip)
                optimizers['gen'].step()

                AdvLoss_d, GradLoss_d, ClsLoss_d = models['stargan'].calc_dis_loss(xin[s0], xin[s1], s0, s1)
                dis_loss = w_adv * AdvLoss_d + w_grad * GradLoss_d + w_cls * ClsLoss_d

                models['dis'].zero_grad()
                dis_loss.backward()
                torch.nn.utils.clip_grad_norm_(models['dis'].parameters(), gradient_clip)
                optimizers['dis'].step()

                gen_loss_mean += gen_loss.item()
                dis_loss_mean += dis_loss.item()
                advloss_d_mean += AdvLoss_d.item()
                gradloss_d_mean += GradLoss_d.item()
                advloss_g_mean += AdvLoss_g.item()
                clsloss_d_mean += ClsLoss_d.item()
                clsloss_g_mean += ClsLoss_g.item()
                cycloss_mean += CycLoss.item()
                recloss_mean += RecLoss.item()

            gen_loss_mean /= n_spk_pair
            dis_loss_mean /= n_spk_pair
            advloss_d_mean /= n_spk_pair
            gradloss_d_mean /= n_spk_pair
            advloss_g_mean /= n_spk_pair
            clsloss_d_mean /= n_spk_pair
            clsloss_g_mean /= n_spk_pair
            cycloss_mean /= n_spk_pair
            recloss_mean /= n_spk_pair

            logging.info(
                'epoch {}, mini-batch {}: AdvLoss_d={:.4f}, AdvLoss_g={:.4f}, GradLoss_d={:.4f}, ClsLoss_d={:.4f}, ClsLoss_g={:.4f}'
                .format(epoch, b + 1, w_adv * advloss_d_mean, w_adv * advloss_g_mean, w_grad * gradloss_d_mean,
                        w_cls * clsloss_d_mean, w_cls * clsloss_g_mean))
            logging.info(
                'epoch {}, mini-batch {}: CycLoss={:.4f}, RecLoss={:.4f}'.format(epoch, b + 1, w_cyc * cycloss_mean,
                                                                                 w_rec * recloss_mean))
            writer.add_scalars('Loss/Total_Loss', {'adv_loss_d': w_adv * advloss_d_mean,
                                                   'adv_loss_g': w_adv * advloss_g_mean,
                                                   'grad_loss_d': w_grad * gradloss_d_mean,
                                                   'cls_loss_d': w_cls * clsloss_d_mean,
                                                   'cls_loss_g': w_cls * clsloss_g_mean,
                                                   'cyc_loss': w_cyc * cycloss_mean,
                                                   'rec_loss': w_rec * recloss_mean}, n_iter)
            n_iter += 1
            b += 1

        if epoch % snapshot == 0:
            for tag in ['gen', 'dis']:
                print('save {} at {} epoch'.format(tag, epoch))
                torch.save({'epoch': epoch,
                            'model_state_dict': models[tag].state_dict(),
                            'optimizer_state_dict': optimizers[tag].state_dict()},
                           os.path.join(model_dir, '{}.{}.pt'.format(epoch, tag)))

    print("===================================Training Finished===================================")


train_stargan()
# Train(models, epochs, train_dataset, train_loader, optimizers, device, model_dir, log_path, model_config, snapshot,
#       resume)