# talkingface-toolkit-VideoReTalking

<a href='https://arxiv.org/abs/2211.14758'><img src='https://img.shields.io/badge/ArXiv-2211.14758-red'></a> <a href='https://vinthony.github.io/video-retalking/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

项目链接：https://github.com/OpenTalker/video-retalking

论文链接：https://arxiv.org/abs/2211.14758

------

## 项目介绍

目前关于音频驱动唇形同步的工作中，存在一些缺点。首先一方面的工作，不是通用模型，需要对目标说话人进行单独训练；另外一方面的工作，当前的通用模型生成出来的唇形模糊；并且它们都不支持情感编辑。作者解决了上述问题并提出一个新的模型。

作者受到wav2lip的启发提出了一种新的模型来用音频驱动嘴唇合成。在wav2lip中，需要输入两组（每组5帧）图像帧，一组是GroudTruth下半部分被mask住的，另外一组是原始视频中的随机5帧（与GroudTruth是不一样的）。Mask图像帧显而易见是我们音频需要驱动生成嘴唇的图像，而随机图像帧是为了给嘴唇生成提供姿势参考。作者提到，模型对于姿势参考的图像帧是十分敏感的，因为这些图像帧中含有的嘴唇信息会泄露给模型作为先验知识。如果直接使用随机图像帧作为姿势参考，生成的图像常常会产生不同步的结果。因此作者对于姿势参考的随机图像帧做了修改，中和了面部表情，再输入到模型作为姿势参考。按照这个思想，也可以使用修改后的高兴或者悲伤的面部表情作为姿势参考，自然而然可以产生相应情感的说话视频。

![pipeline](https://github.com/OpenTalker/video-retalking/raw/main/docs/static/images/pipeline.png?raw=true)

------

## 环境配置

```
conda create -n talkingface python=3.8
activate talkingface

conda install ffmpeg

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html  

pip install cmake -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
pip install -r requirements.txt -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
```

------

## 使用依赖

```
basicsr==1.4.2
kornia==0.5.1
face-alignment==1.3.4
ninja==1.10.2.3
einops==0.4.1
facexlib==0.2.5
librosa==0.9.2
dlib==19.24.0
gradio>=3.7.0
numpy==1.23.4
scipy>=1.2.1
scenedetect==0.5.1
opencv-contrib-python
python_speech_features
ray==2.6.3
colorlog==6.7.0
texttable==1.7.0
```

------

## 快速开始

### 下载预训练模型

下载预训练模型 [pre-trained models](https://pan.baidu.com/s/1WYWb1BYEz0Sbh0UwHjYLUQ?pwd=ga6o) 并放到`./checkpoints`路径下.

### 数据集介绍

作者在两个不同分辨率的数据集上评估了他们的框架：低分辨率数据集LRS2和高分辨率数据集HDTF。

HDTF数据集包含来自YouTube的720p或1080p视频。

根据Prajwal等人在2020年描述的未配对评估设置，作者选择了一个视频和另一个不同视频的音频片段来合成结果，这意味着视频和音频是不匹配的，以此来评估他们框架的性能。

**由于HDTF数据集和LRS2数据集过大，而且本次项目仓库没有给出开源训练代码，因此主要是进行评估任务，所以只选择了部分数据**

### 命令运行

```
python run_talkingface.py --model=video_retalking --dataset=video-retalking
```

------

## 实现功能

- [x] 模型重构嵌入框架
- [x] 数据预处理
- [x] 基于预训练模型生成推理结果
- [x] 仅在命令行输入模型和数据集名称就可以完成推理并对结果进行评估

**因为原论文代码并未给出训练代码，并且已经给出了预训练网络模型，因此本次任务只利用论文代码在talkingface框架内完成推理并对结果进行评估**

### 推理结果

**原视频**

https://github.com/dndnda/talkingface-toolkit/raw/main/dataset/video-retalking/val/face/1.mp4

**推理结果**

https://github.com/dndnda/talkingface-toolkit/raw/main/results/1_1.mp4

### 命令运行截图（包括评估结果）

**具体过程在演示视频中体现**

![运行截图1](https://github.com/dndnda/talkingface-toolkit/blob/main/%E8%BF%90%E8%A1%8C%E6%88%AA%E5%9B%BE/%E8%BF%90%E8%A1%8C%E6%88%AA%E5%9B%BE1.jpg?raw=true)

![运行截图2](https://github.com/dndnda/talkingface-toolkit/blob/main/%E8%BF%90%E8%A1%8C%E6%88%AA%E5%9B%BE/%E8%BF%90%E8%A1%8C%E6%88%AA%E5%9B%BE2.jpg?raw=true)



## 成员分工

张泽渊：论文阅读，推理文件重构，测试代码与纠错，参与相关讨论，参与撰写开发文档

陈则瑜：论文阅读，源项目代码梳理分析，数据整理，参与相关讨论，撰写开发文档

李明：论文阅读，研究评估部分代码，可行性分析，参与相关讨论

