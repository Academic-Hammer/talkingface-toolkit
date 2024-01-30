import os
import h5py
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pickle

def walk_files(root, extension):
    for path, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(extension):
                yield os.path.join(path, file)

def compute_statistics(src, processor, stat_filepath="stat.pkl"):
    melspec_scaler = StandardScaler()
    filenames_all = [[os.path.join(src,d,t) for t in sorted(os.listdir(os.path.join(src,d)))] for d in os.listdir(src)]
    filepath_list = list(walk_files(src, '.h5'))
    for filepath_list in tqdm(filenames_all):
        for filepath in filepath_list:
            # 读取
            melspec = processor.extract_melspec(filepath)
            #import pdb;pdb.set_trace() # Breakpoint
            melspec_scaler.partial_fit(melspec.T)

    with open(stat_filepath, mode='wb') as f:
        pickle.dump(melspec_scaler, f)
    print("Saved.")
    

if __name__ == '__main__':
    pass