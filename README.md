# talkingface-toolkit小组作业
小组成员名单:高艺芙 贺芳琪 陈清扬（姓名排序为各自工作的前后逻辑顺序，与工作量无关）
# 模型选择：
我们选择复现的模型是[StarGAN-VC](https://github.com/kamepong/StarGAN-VC)模型，是专为语音转换任务设计的模型。它是StarGAN框架的延伸，该框架最初用于图像到图像的转换任务。StarGAN-VC专注于将一个说话者的语音特征转换为另一个说话者。
# 作业环境
python3.8

开发环境：PyCharm2022.1.3

框架：PyTorch2.1

操作系统：Windows11，macOs

语音识别库：Librosa0.10.1

数据处理库：NumPy1.20.3
# 数据集
包含p225、226、227，分为test、train、val三种，详见dataset文件夹。
# 运行指令
总指令：python run_talkingface.py --model stargan --dataset vctk

前期数据准备指令：./recipes/run_train.sh

使用数据的指令：例如，当训练时要用对应的数据集，输入StarganDataset(config, config['train_filelist'])能调用train数据集
# 实现功能
总体功能：输入python run_talkingface.py --model stargan --dataset vctk可以直接进行数据处理、训练建模。

前期数据准备：得到了StarGAN模型的配置文件参数，将其运用在toolkit中。

数据部分：按照stargan的源码进行数据预处理，提取mel谱，转化为可用形式；划分数据集，按照80%、10%、10%的比例划分了三个数据集，生成了test.txt、train.txt、val.txt等文件，保存在dataset文件夹中。
# 结果截图
![](./md_img/1.jpg)
![](./md_img/2.jpg)
![](./md_img/3.jpg)
![](./md_img/4.jpg)

# 成员分工（按任务先后顺序编写）
高艺芙-1120213132-07022102:负责配置好实验环境，使用pip install -r requirements.txt语句更新安装包、解决报错问题，训练得到程序运行后的结果。准备实验数据，运行StarGan模型，将得到配置文件Arctic.json,StarGAN.json，将其转换为Arctic.yaml,StarGAN.yaml并整合到相应的文件结构中，整合至talkingface-toolkit/talkingface/properties下，最终打印调试。

贺芳琪-1120210640-08012101:部分配置调整，主要负责data，数据预处理部分，进行了数据集划分。
  将stargan-vc中的dataset.py、compute_statistics.py、extract_features.py、normalize_features.py中有关数据预处理的代码整合到talkingface-toolkit/talkingface/data的dataset和dataprocess文件夹中，并修改了talkingface-toolkit中yaml里面有关数据预处理的参数。详细解释可见data部分的readme文件中。
  
陈清扬-1120213599-07112106:模型代码重构，训练代码重构，推理文件重构，配置文件调整，测试代码与bug修复，撰写实验报告

