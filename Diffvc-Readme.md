# Diffvc-Readme

## 成员分工及工作量描述

吕春吉（贡献度：50%）：组长，负责阅读论文源码，理清架构，进行了model主模型的嵌入、dataset数据处理部分的嵌入、trainer部分的嵌入、推理部分的嵌入、模型的训练和调整、参数的调整等工作
蔡昕怡（贡献度：25%）：负责数据集的查找、数据集的处理、模型的训练和调整、hifi_gan模块的嵌入
唐欣欣（贡献度：10%）：负责encoder训练部分的添加
马翊程（贡献度：10%）：参与数据处理部分的嵌入
孙嘉成（贡献度：5%）：负责数据集的查找

## 完成功能

可以进行diffvc模型的相关训练，可将数据集处理为所需要的形式(进行相关的特征提取和生成)，在命令行输入指定命令可进行模型的训练并返回正确的训练结果。

## 训练截图

参见 readme pic

## 所使用的的依赖

参见 diffvc_requierments.txt文件

## 训练过程

1.下载我们所需要的预训练模型hifi-gan声码器。

取自官方的hifi-gan存储库：https://drive.google.com/file/d/10khlrM645pTbQ4rc2aNEYPba8RFDBkW-/view?usp=sharing

放在checkpts/vocoder/下

2.下载在LibriTTS和VCTK上训练好的模型：

LibriTTS：https://drive.google.com/file/d/18Xbme0CTVo58p2vOHoTQm8PBGW7oEjAy/view?usp=sharing

VCTK：https://drive.google.com/file/d/12s9RPmwp9suleMkBCVetD8pub7wsDAy4/view?usp=sharing

放在checkpts/vc/下

3.进行环境配置，参考diffvc_requirements.txt文件。

4.数据集获取和数据处理部分：

数据集链接：https://www.openslr.org/60/

数据处理部分：

①首先建立一个数据集文件夹data，包含“wavs”、“mels”和“embeds”三个文件夹，并将原数据集上的wav文件按照原文件夹放入wav文件中，在filelist中加入训练数据名称

（以下涉及到的代码块在talkingface/data/dataprocess/下

②运行inference文件的第一部分，配置相应的环境（注：librosa版本为0.9.2）
参见readme pic/pic1

③对inference文件的第二个代码块中的get_mel函数和get_embed函数进行阅读，编写相关代码分别对原wav文件运行这两个函数。对原始数据运行get_mel函数生成mel文件，运行get_embed函数生成embed文件，分别保存在两个文件夹中。（注意，这里需要调用spk_encoder的预训练模型，下载之后引用其路径即可）

参见readme pic/pic2

④在原代码中补充引用预训练模型、提取wav文件并执行两个函数，并分别保存到相应的两个文件夹中且正确命名的代码：

参见readme pic/pic3 

⑤运行该代码块，即可得到处理后的embed和mel文件。

5.创建logs_enc文件夹，并下载训练好的编码器放在该文件夹下

编码器下载地址：https://drive.google.com/file/d/1JdoC5hh7k6Nz_oTcumH0nXNEib-GDbSq/view?usp=sharing

6.新建log_dec文件夹

7.进行训练

## 框架整体介绍

### checkpoints

主要保存的是训练和评估模型所需要的额外的预训练模型，在对应文件夹的[README](https://github.com/Academic-Hammer/talkingface-toolkit/blob/main/checkpoints/README.md)有更详细的介绍

### datset

存放数据集以及数据集预处理之后的数据，详细内容见dataset里的[README](https://github.com/Academic-Hammer/talkingface-toolkit/blob/main/dataset/README.md)

### saved

存放训练过程中保存的模型checkpoint, 训练过程中保存模型时自动创建

### talkingface

主要功能模块，包括所有核心代码

#### config

根据模型和数据集名称自动生成所有模型、数据集、训练、评估等相关的配置信息

```
config/

├── configurator.py

```

#### data

- dataprocess：该模型主要涉及到inference.ipynb和get_avg_mels.ipynb文件，用于生成数据集中的mels和embed文件夹和文件夹中的内容。
- dataset：数据处理部分，主要为diffvc_dataset，其他文件为训练encoder部分所需的数据处理，可不关注。

```
data/

├── dataprocess

| ├── wav2lip_process.py

| ├── inference.ipynb

| ├── get_avg_mels.ipynb

├── dataset

| ├── wav2lip_dataset.py

| ├── diffvc_dataset.py
```

#### evaluate

主要涉及模型评估的代码
LSE metric 需要的数据是生成的视频列表
SSIM metric 需要的数据是生成的视频和真实的视频列表

#### model

实现的模型的网络和对应的方法 

diffvc存储在voice_conversion文件夹下的diffvc.py中，hifi-gan为训练encoder所需模块，可不关注。

```
model/

├── audio_driven_talkingface

| ├── wav2lip.py

├── image_driven_talkingface

| ├── xxxx.py

├── nerf_based_talkingface

| ├── xxxx.py

├── voice_conversion

| ├── diffvc.py

├── abstract_talkingface.py

```

#### properties

保存默认配置文件，包括diffvc.yaml，diffvc_dataset.yaml,diffvc_encoder.yaml和diffvc_encoder_dataset.yaml,其中diffvc_encoder.yaml和diffvc_encoder_dataset.yaml为训练encoder所需要的配置文件，可不关注。

```
properties/

├── dataset

| ├── diffvc_dataset.yaml

| ├── diffvc_encoder_dataset.yaml

├── model

| ├── diffvc.yaml

| ├── diffvc_encoder.yaml

├── overall.yaml

```

#### quick_start

通用的启动文件，根据传入参数自动配置数据集和模型，然后训练和评估

```
quick_start/

├── quick_start.py

```

#### trainer

训练、评估函数的主类。

```
trainer/

├── trainer.py

```

#### utils

公用的工具类。

## 使用方法

### 环境要求

参见 diffvc_requierments.txt，可运行以下代码配置。

```
pip install -r diffvc_requierments.txt
```



### 训练和评估

```bash
python run_talkingface.py --model=diffvc -–dataset=diffvc_data
```



## 论文及源代码仓库链接：

论文：https://arxiv.org/abs/2109.13821

源代码仓库：https://github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC



