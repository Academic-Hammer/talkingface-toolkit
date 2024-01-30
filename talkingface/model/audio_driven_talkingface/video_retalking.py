import os
import sys
import subprocess
import platform
from talkingface.model.abstract_talkingface import AbstractTalkingFace
# def get_virtual_environment_name():
#     # 获取 Python 解释器的安装路径
#     python_path = sys.executable

#     # 提取虚拟环境的名称
#     venv_name = os.path.basename(os.path.dirname(python_path))
#     #print('当前虚拟环境为：',venv_name)
#     return venv_name

def function_inference():
    # 获取当前工作目录

    # 构建inference.py文件的路径
    cd_path = os.path.join(os.path.dirname(__file__), 'videoretalking')

    inference_script_path = os.path.join(os.path.dirname(__file__), 'videoretalking', 'inference.py')

    # 获取当前虚拟环境的名称
    # virtual_environment_name = get_virtual_environment_name()
    
    print('下面开始将原始数据转化为待评估的视频：')
    # command_to_run = f'activate {virtual_environment_name} && cd {cd_path} && python {inference_script_path}'
    command_to_run = f'cd {cd_path} && python {inference_script_path}'
    try:
        # 使用subprocess.run()执行命令，并等待其完成
        result = subprocess.run(command_to_run, shell=True, check=True, text=True)
        print("命令执行成功，输出：", result.stdout)
    except subprocess.CalledProcessError as e:
        print("命令执行失败，错误信息：", e.output)

class video_retalking(AbstractTalkingFace):
    def __init__(self, config):
        super(video_retalking, self).__init__()
    def generate_batch(self):

        function_inference()

        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建 x.py 文件的完整路径
        x_py_path = os.path.join(current_dir, 'video-retalking', 'inference.py')
        # 执行 x.py 文件
        # os.system(f'python {x_py_path}')
        # 获取父目录的路径
        parent_dir = os.path.dirname(current_dir)
        # 获取父目录的父目录的路径
        grandparent_dir = os.path.dirname(parent_dir)
        # 获取父目录的父目录的父目录的路径
        great_grandparent_dir = os.path.dirname(grandparent_dir)
        generated_video_path = os.path.join(great_grandparent_dir, "results", "1_1.mp4")
        real_video_path = os.path.join(great_grandparent_dir, "dataset", "video-retalking", "val", "face", "1.mp4")

        print("评估使用的模型生成的视频位于:", generated_video_path)
        print("评估使用的数据集的原视频位于:", real_video_path)
        # 返回固定的数据字典
        file_dict = {'generated_video': [], 'real_video': []}
        file_dict['generated_video'].append(generated_video_path)
        file_dict['real_video'].append(real_video_path)

        return file_dict