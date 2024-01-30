# 小组README文件
## 项目介绍-Emotionally Enhanced Talking Face Generation

"情感增强的说话面部生成"这篇论文主要关注于通过加入广泛的情感范围来创建更加逼真和有说服力的说话面部视频。它解决了以往工作的局限性，这些工作通常无法创建逼真的视频，因为它们很少关注人物的表情和情感。本项目提出的框架旨在生成包含适当表情和情感的唇同步说话面部视频，使其更具说服力。

## 项目功能
说话面部生成：该框架基于基础骨架架构，使用2D-CNN编解码器网络生成单独的帧。这涉及到一个面部编码器、一个音频编码器和一个解码器，强调视觉质量和准确的唇同步生成。

说话面部生成中的情感捕捉：这是关键部分，因为它涉及将情感信息包含在视频中。该方法将语音音频中表示的情感与视频生成的独立情感标签分开，提供了更多控制主题情感的方法。

数据预处理和增强：该框架使用完全遮盖的帧以及参考帧来加入情感，因为情感不仅通过面部的嘴唇区域来表达。

情感编码器：这将分类情感编码进视频生成过程。

## 环境要求（依赖）
依赖库详见 requirements.txt。要求安装ffmpeg, 安装albumentations库

## 实现及演示

### 张卓远

编写实现了模型的数据预处理和数据加载代码，编写实现了模型emo_disc情绪鉴别器模型，编写emogen.yaml。

运行步骤：

进入talkingface/data/dataprocess/文件夹下运行控制台，输入命令

python emogen_process.py --input_folder <folder_of_dataset> --preprocessed_root <output_folder_for_preprocessed_dataset/>

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


### 蒋政
完成训练部分代码编写，实现专家口型同步鉴别器模型

修改requirements.txt

修改utils文件夹中的utils.py

在model文件夹中添加 wav2lip.py，修改__init__.py

在model/audio_driven_talkingface下添加conv.py , emogen_syncnet.py

在trainer文件夹中添加color_syncnet_train.py, emotion_disc_train.py

### 周扬
填充了image_driven所需的模型文件，包括conv.py,emo_disc.py,emo_syncnet.py和wav2lip.py
