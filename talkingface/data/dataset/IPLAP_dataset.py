import torch
import subprocess


class IPLAPDataset(torch.utils.data.Dataset):
    def __init__(self, config, datasplit):
        self.config = config
        self.split = datasplit
        self.preprocess_data()

    def preprocess_data(self):
        # 定义预处理脚本的路径
        audio_script_path = "talkingface/data/dataprocess/IPLAP_process/preprocess_audio.py"
        video_script_path = "talkingface/data/dataprocess/IPLAP_process/preprocess_video.py"

        # 运行预处理音频脚本
        subprocess.run(["python", audio_script_path], check=True)

        # 运行预处理视频脚本
        subprocess.run(["python", video_script_path], check=True)

    def __getitem__(self, index):
        return {}

