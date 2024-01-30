# MakeItTalk README

## 队员 & 分工

* **陈顺章  1120212450**

  负责整体代码框架的编写，进行原仓库到课程仓库接口的迁移重写，`main_end2end.py` 的测试与运行

* **李沅臻  1120210631**

  负责本地模型的训练以及 `train_content.py` 的运行

* **张司睿  1120211750**

  负责 `main_train_image_translation.py` 的运行以及代码框架的阅读，README 文档编写

* **夏诗航  1120212365**

  负责 `main_end2end_cartoon.py` 的运行

* **黎书萌  1120212093**

  负责 `main_gen_new_puppet` 的运行

## 项目简介

MakeItTalk 是一个由马萨诸塞大学阿默斯特分校、Adobe 研究院等机构提出的新方法。这种方法不仅能让真人头像说话，还可以让卡通、油画、素描、日漫中的人像说话。与之前的方法不同，MakeItTalk **将输入音频信号中的内容和说话人身份信息分离开来。音频内容用于稳健地控制嘴唇及周围区域的运动，而说话人信息则决定了面部表情的细节和人物的头部动态。**
其具体工作流程包括：给定一段音频和一张面部图像，MakeItTalk 可以生成说话人的头部状态动画，且声画同步。在训练阶段，研究者使用现成可用的人脸特征点检测器对输入图像进行预处理，提取面部特征点。然后使用输入音频和提取到的特征点直接训练使语音内容动态化的基线模型。为了达到高保真动态效果，研究者尝试将输入音频信号的语音内容和说话人嵌入分离开来，进而实现面部特征点的预测，其特征点-图像合成算法分为两种：对于非真人图像，如油画或矢量图，该研究使用基于德劳内三角剖分的简单换脸方法。对于真人图像，则使用图像转换网络将真人面部图像和底层特征点预测动态化。



**项目网址链接**：https://people.umass.edu/~yangzhou/MakeItTalk/

**原论文链接**：https://people.umass.edu/~yangzhou/MakeItTalk/MakeItTalk_SIGGRAPH_Asia_Final_round-5.pdf

**论文仓库链接**：[GitHub - yzhou359/MakeItTalk](https://github.com/yzhou359/MakeItTalk)

## 模型介绍

共分为四个主要模型：

**FaceAlignment**：该模型属于face_alignment库

**AutoVC_mel_Convertor**：该模型用于将音频转换为 AutoVC 模型的输入

**Audio2landmark_model**：用于从音频生成人脸关键点

**Image_translation_block**：用于进行图像+人脸关键点到处理后的图像的转换

原论文仓库只开放了**Audio2landmark_model**下的Audio2landmark_content的训练内容，属于audio-driven (音频驱动)模型。故本次实验的训练环节仅能复现该模型的训练过程

## 运行环境

环境要求集成在Makeittalk_requirement.txt中，通过下列操作安装所需环境：

```
pip install -r Makeittalk_requirement.txt
```



安装 `ffmpeg` 

```bash
sudo apt-get install ffmpeg
```

## 运行前准备

* 安装运行环境

* 从 https://drive.google.com/drive/folders/1EwuAy3j1b9Zc1MsidUfxG_pJGc_cV60O?usp=sharing 下载 .pickle 文件到 `\checkpoints\dump` 文件夹

* 下载剩余三个需要的模型到 `\checkpoints\ckpt` 文件夹中

  https://drive.google.com/file/d/1i2LJXKp-yWKIEEgJ7C6cE3_2NirfY_0a/view?usp=sharing

  https://drive.google.com/file/d/1rV0jkyDqPW-aDJcj7xSO6Zt1zSXqn1mu/view?usp=sharing

  https://drive.google.com/file/d/1ZiwPp_h62LtjU0DwpelLUoodKPR85K7x/view?usp=sharing

  

## 项目功能

在我们的复现中，实现了**一键训练、生成视频与评估**：

执行指令：

```shell
python run_talkingface.py --model=audio2landmark_content --dataset=audio2landmark_content
```

训练后的**视频**将自动存储在  `dataset/examples`  里面

训练结束后的**模型**将存储在 `saved/` 里面

## 实验结果截图

该部分是提供了在我们的服务器上跑通整个实验部分的截图：

* 输入命令后开始读取 yaml 中的内容：

![a462dca11cf10eba37be31df87faab6](.\images\a462dca11cf10eba37be31df87faab6.png)

* 输出模型信息

  ![image-20240130164339985](.\images\image-20240130164339985.png)

* 完成前置步骤并开始训练（为了节省测试时间，训练的 epoch 数设为 1 ）：

​	![image-20240130164713473](.\images\image-20240130164713473.png)

* 训练完成，自动评估是否过拟合：

![1369e02642f7ca17e3662d85caadf36](.\images\1369e02642f7ca17e3662d85caadf36.png)

* 训练结束后自动生成关键点与用于评估的视频

![a5d2a969333e351073356f97dab5729](.\images\a5d2a969333e351073356f97dab5729.png)

![40b96798b990d054c72e831f351b3fe](.\images\40b96798b990d054c72e831f351b3fe.png)

* 使用框架中的 LSE 方法进行比对与评估，评估结果如下：

![ee9f548f6a737676b375e53608e9e23](.\images\ee9f548f6a737676b375e53608e9e23.png)

* 评估结果：

```
30 Jan 01:47 INFO {'lse’: {'LSE-C: 0.9352130889892578', 'LSE-D: 14.274614334106445'},'ssim': 0.14530561963417538 }
```

* 完整运行流程

  ![progress](.\images\progress.gif)

## 备注

* 由于这个是一个视频生成框架，通过关键点将图片转换成视频，不存在真实视频，我们的对比视频是作者训练 1001 轮的模型之后处理出来的视频
* 作者的代码仓库中只提供了 `train_content` 部分的训练，其余的训练作者并没有提供训练方法以及数据集，因此只将模型 `ckpt_content_branch` 替换成了我们自己的，剩下的均用了作者预训练好的模型

## 报错记录

![image-20240130183232166](.\images\image-20240130183232166.png)
