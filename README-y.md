# Grad-TTS提交说明

## 项目
Grad-TTS, text to speech
成员：宋楚齐 杨昆龙 张子瑞 张文锦 钟宇鹏

## 完成的功能
- [x] 数据载入融入框架
- [x] 模型载入融入框架
- [x] 训练过程融入框架
- [ ] 验证过程融入框架 (源代码未提供验证方法)
- [x] 仅在命令行输入模型和数据集名称就可以训练
```
python run_talkingface.py --model=GradTTS --dataset=LJSpeech1.1
```
## 添加的文件
|                                文件名称                                 |                      添加位置                       |            文件功能            |
|:-------------------------------------------------------------------:|:-----------------------------------------------:|:--------------------------:|
|                 [LJSpeech1.1](dataset/LJSpeech1.1)                  |               dataset/LJSpeech1.1               |      LJSpeech1.1数据集存放      |
|             [gradtts](talkingface/data/dataset/gradtts)             |        talkingface/data/dataset/gradtts         | gradtts_dataset.py所需要的一些方法 |
|  [gradtts_dataset.py](talkingface/data/dataset/gradtts_dataset.py)  |   talkingface/data/dataset/gradtts_dataset.py   |           数据集载入            |
|             [tts](talkingface/model/text_to_speech/tts)             |      talkingface/model/text_to_speech/tts       |     gradtts.py 需要的一些方法     |
|      [gradtts.py](talkingface/model/text_to_speech/gradtts.py)      |   talkingface/model/text_to_speech/gradtts.py   |       gradtts模型网络结构        |
| [LJSpeech1.1.yaml](talkingface/properties/dataset/LJSpeech1.1.yaml) | talkingface/properties/dataset/LJSpeech1.1.yaml |    LJSpeech1.1数据集的配置文件     |
|      [gradtts.yaml](talkingface/properties/model/GradTTS.yaml)      |    talkingface/properties/model/GradTTS.yaml    |       gradtts模型的配置文件       |
|     [overall.yaml](talkingface/properties/overall.yaml)     |     talkingface/properties/overall.yaml-行1      |      修改为本地GPU的ID：0，1       |
|                     [README-y.md](README-y.md)                      |                       根目录                       |         本文档，做一些说明          |
## 修改的文件
|                                文件名称                                 |                      修改位置                       |                    修改内容                    |
|:-------------------------------------------------------------------:|:-----------------------------------------------:|:------------------------------------------:|
|                 [__init__.py](talkingface/data/dataset/__init__.py)                  |     talkingface/data/dataset/__init__.py-行2     |            添加对GradTTSDataset的引用            |
|             [__init__.py](talkingface/model/text_to_speech/__init__.py)             | talkingface/model/text_to_speech/__init__.py-行1 |               添加对GradTTS的引用                |
|  [trainer.py](talkingface/trainer/trainer.py)  |     talkingface/trainer/trainer.py-行545-710     | 新写了GradTTSTrainer类，重写了fit方法和_valid_epoch方法 |
## 运行结果截图

## 所使用的依赖
- `python=3.8`（如果需要其他版本的python，请按照解决方案中的A1步骤进行）
### 额外第三方库需求
- blurhash==1.1.4
- boost==0.1
- blurhash==1.1.4
- boost==0.1
- Cython==0.29.23
- einops==0.7.0
- greenlet==3.0.3
- greenlet==3.0.3
- Mastodon.py==1.8.1
- python-magic-bin==0.4.14
- resampy==0.4.2
- SQLAlchemy==2.0.25
- Unidecode==1.1.2
### 冲突的第三方库
- kornia==0.7.1
- 主项目依赖的kornia版本为0.5.11，而本项目依赖的kornia版本为0.7.1，因此需要将主项目依赖的kornia版本修改为0.7.1
### 所有需求打包在[requirements-y.txt](requirements-y.txt)中，在安装完主项目依赖后可通过以下命令安装
````
pip install -r requirements-y.txt
````


## 成员工作量
- 宋楚齐：跑通项目，模型载入、训练过程融入框架，readme文档编写，项目管理A，组长
- 杨昆龙：跑通项目，数据集载入、训练过程融入框架、fit方法，readme文档编写，项目管理B
- 张文锦：跑通项目，数据集载入，readme文档编写
- 张子瑞：跑通项目，训练过程融入框架
- 钟宇鹏：跑通项目，训练过程融入框架

## 可能出现的问题及解决方案
- [Q1] 问题1：在运行run_talkingface.py时，出现如下错误：
```
ModuleNotFoundError: No module named 'talkingface.model.text_to_speech.tts.monotonic_align.core'
```
- [A1]可通过以下步骤解决
```
cd talkingface/model/text_to_speech/tts/monotonic_align
python setup.py --build_ext --inplace
cd ../../../..

先将talkingface/model/text_to_speech/tts/monotonic_align路径下的core.cp38-win_amd64.pyd文件删除
然后将talkingface/model/text_to_speech/tts/monotonic_align/build文件夹下lib开头的文件夹下的pyd文件复制到talkingface/model/text_to_speech/tts/monotonic_align文件夹下
```
## 其他需要说明的地方
- 本次实验使用的数据集为LJSpeech1.1，数据集下载地址为：https://keithito.com/LJ-Speech-Dataset/
- 由于源代码的data_utils.DataLoader中使用了筛选器参数collate_fn，而本项目的数据载入在quick_start.py中,出于项目兼容考虑，不便修改源代码，因此在GradTTSTrainer类中重写了fit方法，将数据集重新载入
- 由于源代码的trainer.Trainer中的valid_epoch方法中出现代码编写错误，因此在GradTTSTrainer类中重写了_valid_epoch方法
- 源代码拥有多人语音和单人语音两种训练模式，但是多人语音模式下的数据集并未提供，因此本次实验只实现了单人语音模式下的训练代码
- 源代码未提供验证方法，因此本次实验只实现了训练代码