# talkingface-toolkit
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
- dataprocess：模型特有的数据处理代码，（可以是对方仓库自己实现的音频特征提取、推理时的数据处理）。如果实现的模型有这个需求，就要建立一对应的文件
- dataset：每个模型都要重载`torch.utils.data.Dataset` 用于加载数据。每个模型都要有一个`model_name+'_dataset.py'`文件. `__getitem__()`方法的返回值应处理成字典类型的数据。 <span style="color:red">(核心部分)</span>
```
data/

├── dataprocess

| ├── wav2lip_process.py

| ├── xxxx_process.py

├── dataset

| ├── wav2lip_dataset.py

| ├── xxx_dataset.py
```

#### evaluate
主要涉及模型评估的代码
LSE metric 需要的数据是生成的视频列表
SSIM metric 需要的数据是生成的视频和真实的视频列表

#### model
实现的模型的网络和对应的方法 <span style="color:red">（核心部分）</span>

主要分三类：
- audio-driven (音频驱动)
- image-driven （图像驱动）
- nerf-based （基于神经辐射场的方法）

```
model/

├── audio_driven_talkingface

| ├── wav2lip.py

├── image_driven_talkingface

| ├── xxxx.py

├── nerf_based_talkingface

| ├── xxxx.py

├── abstract_talkingface.py

```

#### properties
保存默认配置文件，包括：
- 数据集配置文件
- 模型配置文件
- 通用配置文件

需要根据对应模型和数据集增加对应的配置文件，通用配置文件`overall.yaml`一般不做修改
```
properties/

├── dataset

| ├── xxx.yaml

├── model

| ├── xxx.yaml

├── overall.yaml

```

#### quick_start
通用的启动文件，根据传入参数自动配置数据集和模型，然后训练和评估（一般不需要修改）
```
quick_start/

├── quick_start.py

```

#### trainer
训练、评估函数的主类。在trainer中，如果可以使用基类`Trainer`实现所有功能，则不需要写一个新的。如果模型训练有一些特有部分，则需要重载`Trainer`。需要重载部分可能主要集中于: `_train_epoch()`, `_valid_epoch()`。 重载的`Trainer`应该命名为：`{model_name}Trainer`
```
trainer/

├── trainer.py

```

#### utils
公用的工具类，包括`s3fd`人脸检测，视频抽帧、视频抽音频方法。还包括根据参数配置找对应的模型类、数据类等方法。
一般不需要修改，但可以适当添加一些必须的且相对普遍的数据处理文件。

## 使用方法
### 环境要求
- `python=3.8`
- `torch==1.13.1+cu116`（gpu版，若设备不支持cuda可以使用cpu版）
- `numpy==1.20.3`
- `librosa==0.10.1`

尽量保证上面几个包的版本一致

提供了两种配置其他环境的方法：
```
pip install -r requirements.txt

or

conda env create -f environment.yml
```

建议使用conda虚拟环境！！！

### 训练和评估

```bash
python run_talkingface.py --model=xxxx --dataset=xxxx (--other_parameters=xxxxxx)
```

### 权重文件

- LSE评估需要的权重: syncnet_v2.model [百度网盘下载](https://pan.baidu.com/s/1vQoL9FuKlPyrHOGKihtfVA?pwd=32hc)
- wav2lip需要的lip expert 权重：lipsync_expert.pth [百度网下载](https://pan.baidu.com/s/1vQoL9FuKlPyrHOGKihtfVA?pwd=32hc)

## 可选论文：
### Aduio_driven talkingface
| 模型简称 | 论文 | 代码仓库 |
|:--------:|:--------:|:--------:|
| MakeItTalk | [paper](https://arxiv.org/abs/2004.12992) | [code](https://github.com/yzhou359/MakeItTalk) |
| MEAD | [paper](https://wywu.github.io/projects/MEAD/support/MEAD.pdf) | [code](https://github.com/uniBruce/Mead) |
| RhythmicHead | [paper](https://arxiv.org/pdf/2007.08547v1.pdf) | [code](https://github.com/lelechen63/Talking-head-Generation-with-Rhythmic-Head-Motion) |
| PC-AVS | [paper](https://arxiv.org/abs/2104.11116) | [code](https://github.com/Hangz-nju-cuhk/Talking-Face_PC-AVS) |
| EVP | [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ji_Audio-Driven_Emotional_Video_Portraits_CVPR_2021_paper.pdf) | [code](https://github.com/jixinya/EVP) |
| LSP | [paper](https://arxiv.org/abs/2109.10595) | [code](https://github.com/YuanxunLu/LiveSpeechPortraits) |
| EAMM | [paper](https://arxiv.org/pdf/2205.15278.pdf) | [code](https://github.com/jixinya/EAMM/) |
| DiffTalk | [paper](https://arxiv.org/abs/2301.03786) | [code](https://github.com/sstzal/DiffTalk) |
| TalkLip | [paper](https://arxiv.org/pdf/2303.17480.pdf) | [code](https://github.com/Sxjdwang/TalkLip) |
| EmoGen | [paper](https://arxiv.org/pdf/2303.11548.pdf) | [code](https://github.com/sahilg06/EmoGen) |
| SadTalker | [paper](https://arxiv.org/abs/2211.12194) | [code](https://github.com/OpenTalker/SadTalker) |
| HyperLips | [paper](https://arxiv.org/abs/2310.05720) | [code](https://github.com/semchan/HyperLips) |
| PHADTF | [paper](http://arxiv.org/abs/2002.10137) | [code](https://github.com/yiranran/Audio-driven-TalkingFace-HeadPose) |
| VideoReTalking | [paper](https://arxiv.org/abs/2211.14758) | [code](https://github.com/OpenTalker/video-retalking#videoretalking--audio-based-lip-synchronization-for-talking-head-video-editing-in-the-wild-)
|                                 |



### Image_driven talkingface
| 模型简称 | 论文 | 代码仓库 |
|:--------:|:--------:|:--------:|
| PIRenderer | [paper](https://arxiv.org/pdf/2109.08379.pdf) | [code](https://github.com/RenYurui/PIRender) |
| StyleHEAT | [paper](https://arxiv.org/pdf/2203.04036.pdf) | [code](https://github.com/OpenTalker/StyleHEAT) |
| MetaPortrait | [paper](https://arxiv.org/abs/2212.08062) | [code](https://github.com/Meta-Portrait/MetaPortrait) |
|                                 |
### Nerf-based talkingface
| 模型简称 | 论文 | 代码仓库 |
|:--------:|:--------:|:--------:|
| AD-NeRF | [paper](https://arxiv.org/abs/2103.11078) | [code](https://github.com/YudongGuo/AD-NeRF) |
| GeneFace | [paper](https://arxiv.org/abs/2301.13430) | [code](https://github.com/yerfor/GeneFace) |
| DFRF | [paper](https://arxiv.org/abs/2207.11770) | [code](https://github.com/sstzal/DFRF) |
|                                 |
### text_to_speech
| 模型简称 | 论文 | 代码仓库 |
|:--------:|:--------:|:--------:|
| VITS | [paper](https://arxiv.org/abs/2106.06103) | [code](https://github.com/jaywalnut310/vits) |
| Glow TTS | [paper](https://arxiv.org/abs/2005.11129) | [code](https://github.com/jaywalnut310/glow-tts) |
| FastSpeech2 | [paper](https://arxiv.org/abs/2006.04558v1) | [code](https://github.com/ming024/FastSpeech2) |
| StyleTTS2 | [paper](https://arxiv.org/abs/2306.07691) | [code](https://github.com/yl4579/StyleTTS2) |
| Grad-TTS | [paper](https://arxiv.org/abs/2105.06337) | [code](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS) | 
| FastSpeech | [paper](https://arxiv.org/abs/1905.09263) | [code](https://github.com/xcmyz/FastSpeech) |
|                                 |
### voice_conversion
| 模型简称 | 论文 | 代码仓库 |
|:--------:|:--------:|:--------:|
| StarGAN-VC | [paper](http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/stargan-vc2/index.html) | [code](https://github.com/kamepong/StarGAN-VC) |
| Emo-StarGAN | [paper](https://www.researchgate.net/publication/373161292_Emo-StarGAN_A_Semi-Supervised_Any-to-Many_Non-Parallel_Emotion-Preserving_Voice_Conversion) | [code](https://github.com/suhitaghosh10/emo-stargan) |
| adaptive-VC | [paper](https://arxiv.org/abs/1904.05742) | [code](https://github.com/jjery2243542/adaptive_voice_conversion) |
| DiffVC | [paper](https://arxiv.org/abs/2109.13821) | [code](https://github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC) |
| Assem-VC | [paper](https://arxiv.org/abs/2104.00931) | [code](https://github.com/maum-ai/assem-vc) |
|               |

## 作业要求
- 确保可以仅在命令行输入模型和数据集名称就可以训练、验证。（部分仓库没有提供训练代码的，可以不训练）
- 每个组都要提交一个README文件，写明完成的功能、最终实现的训练、验证截图、所使用的依赖、成员分工等。


##小组README文件
###张卓远
编写实现了模型的数据预处理和数据加载代码，编写实现了模型emo_disc部分模型。
运行步骤：
进入talkingface/data/dataprocess/文件夹下运行控制台，输入命令
python emogen_dataprocess.py --inputfolder <folder_of_dataset> --preprocessed_root <output_folder_for_preprocessed_dataset/>
预处理会先将视频转化为25帧格式，此过程需要安装ffmpeg并添加为环境变量。
并将转化好的视频存入./modified_videos文件夹下
<img width="1280" alt="数据预处理第一步转换FPS演示结果" src="https://github.com/zhangyuanyuan02/talkingface-toolkit/assets/103866519/11d41884-5fc6-4af1-8664-2bb58e54db30">
<img width="1276" alt="转换视频FPS结果" src="https://github.com/zhangyuanyuan02/talkingface-toolkit/assets/103866519/1b751036-dbea-47b6-b3a3-fa82a8c711a7">
程序会自动运行数据预处理第二步：
<img width="1280" alt="数据预处理第二步演示" src="https://github.com/zhangyuanyuan02/talkingface-toolkit/assets/103866519/a2eba2fb-ca15-419e-896d-fe7665474ca9">
由于本机内存空间不足转为使用其他云服务器测试，完成数据预处理的过程。
<img width="1280" alt="预处理内存空间不足报错" src="https://github.com/zhangyuanyuan02/talkingface-toolkit/assets/103866519/d393514c-8107-495a-a4f9-d7da15acc063">
<img width="578" alt="预处理完成演示" src="https://github.com/zhangyuanyuan02/talkingface-toolkit/assets/103866519/16264f1c-188c-463a-a2f0-db03aa9faf03">

接下来训练情绪鉴别器：

<img width="572" alt="训练情绪鉴别器结果演示" src="https://github.com/zhangyuanyuan02/talkingface-toolkit/assets/103866519/3c7e3c19-0939-4c41-a4b8-7f4489026fdc">

由于训练过程过长，可以在中途输入Ctrl^C以停止训练










