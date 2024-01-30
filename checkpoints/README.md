这个文件夹中保存的是，模型训练或验证过程中用到的一些额外的预训练权重如：
- LiveSpeechPortraits中用到的数据放在live_speech_portraits目录下。
- LiveSpeechPortraits需要的数据请从此处[data - Google 云端硬盘](https://drive.google.com/drive/folders/1sHc2xEEGwnb0h2rkUhG9sPmOxvRvPVpJ)下载，并且放在live_speech_portraits中。
- wav2lip中用到的syncnet权重
- 计算合成视频lip-audio同步LSE用到的syncnet-v2权重
- .......
- 

目录结构为：
```
checkpoints/

├── LSE
| ├── syncnet_v2.model  ()

├── Wav2Lip
| ├── lipsync_expert.pth ()

```
