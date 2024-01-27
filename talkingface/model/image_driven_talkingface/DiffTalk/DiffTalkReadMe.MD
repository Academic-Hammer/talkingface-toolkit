# DiffTalk #
The pytorch implementation for "DiffTalk: Crafting Diffusion Models for Generalized Audio-Driven Portraits Animation".
Incorporated in this framework by Huanran Chen.

## Extra requirements for training and inference
training:
- pytorch-lightning 1.2.5 (Also required for loading models)

testing:
- configargparse
- Omegaconf
- taming-transformers. https://blog.csdn.net/qq_45560230/article/details/134936234. Need to replace one file after installation.
For more details, please refer to the `requirements.txt`. 

Requires 8 NVIDIA 3090Ti GPUs.

Put the first stage [model](https://cloud.tsinghua.edu.cn/f/7eb11fc208144ed0ad20/?dl=1) to `./models`.

** Remember to run the model at the root of taking-face.

## Model
Obviously its impossible to push the checkpoint to github.

Please download the checkpoint from here: https://cloud.tsinghua.edu.cn/f/7eb11fc208144ed0ad20/?dl=1

Please do not modify the checkpoint name. Keep it as "model.ckpt".

Put it into "./checkpoints/DiffTalk"


## Dataset
Please download the HDTF dataset for training and test, and process the dataset as following.

**Data Preprocessing:**


1. Set all videos to 25 fps.
2. Extract the audio signals and facial landmarks.
3. Put the processed data in `./data/HDTF`, and construct the data directory as following.
4. Constract the `data_train.txt` and `data_test.txt` as following.

./data/HDTF:

    |——data/HDTF
       |——images
          |——0_0.jpg
          |——0_1.jpg
          |——...
          |——N_M.bin
       |——landmarks
          |——0_0.lmd
          |——0_1.lmd
          |——...
          |——N_M.lms
       |——audio_smooth
          |——0_0.npy
          |——0_1.npy
          |——...
          |——N_M.npy

./data/data_train(test).txt:

    0_0
    0_1
    0_2
    ...
    N_M


N is the total number of classes, and M is the class size.


## Training
It requires 8 GPUs to train. We are impossible to train this, but we implement the trainer, and you can use our code to train it.
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./trainer/DiffTalkTrainer.py --base ./model/image_driven_talkingface/DiffTalk/configs/latent-diffusion/talking.yaml -t --gpus 0,1,2,3,4,5,6,7,
```

## Test
I have already incorporated the model into this framework. You can directly initialize the model from the top API.

**I also write a small demo, see "demo.py" in this directory.** You need to copy it into the root and provide it data, then you can directly run it.

## Weakness
1. The DiffTalk models talking head generation as an iterative denoising process, which needs more time to synthesize a frame compared with most GAN-based approaches. This is also a common problem of LDM-based works.
2. The model is trained on the HDTF dataset, and it sometimes fails on some identities from other datasets.
3. When driving a portrait with more challenging cross-identity audio, the audio-lip synchronization of the synthesized video is slightly inferior to the ones under self-driven setting.
4. During inference, the network is also sensitive to the mask shape in z_T , where the mask needs to cover the mouth region completely and its shape cannot leak any
lip shape information.

## Huanran & Zhangshuo's result
For training, it requires 8 GPUs to train. We are impossible to train this, but we implement the trainer, and you can use our code to train it.
For testing, its very easy to initialize the model:
write some thing at the root of this project, just:
```python
from talkingface.model.image_driven_talkingface.DiffTalk import get_difftalk_inference

model = get_difftalk_inference()

print(model)
```
Or just Call the API. You can see the loaded model:
![img.png](img.png)


Here is the results:
<br>①"This administration and this congress have governed with cruelty and have broken every promise they made to the American people.
During the campaign we heard the president promised that he would keep jobs here at home buy and hire Americanand hold countries like China accountable."              
![This administration and this congress have governed with cruelty and have broken every promise they made to the American people. During the campaign we heard the president promised that he would keep jobs here at home buy and hire Americanand hold countries like China accountable.](https://github.com/huanranchen/talkingface-toolkit/blob/main/talkingface/model/image_driven_talkingface/DiffTalk/our_demos/2.gif)
<br>②"Good for the economy I mean what we see is most likely of the ranking member Maxine Waters would become the chairman should the democrats take over. She has been leading the impeachment charge in the house. She is called for te harassment of the presidents' administration and members."
![Good for the economy I mean what we see is most likely of the ranking member Maxine Waters would become the chairman should the democrats take over. She has been leading the impeachment charge in the house. She is called for te harassment of the presidents' administration and members.](https://github.com/huanranchen/talkingface-toolkit/blob/main/talkingface/model/image_driven_talkingface/DiffTalk/our_demos/2.gif)
<br>③"End to the other I talked to everybody that I can find folks waking their dog filling up their gas tank. I talked to them about what matters to them, what worries them and what congress should be doing to help."
![End to the other I talked to everybody that I can find folks waking their dog filling up their gas tank. I talked to them about what matters to them, what worries them and what congress should be doing to help.](https://github.com/huanranchen/talkingface-toolkit/blob/main/talkingface/model/image_driven_talkingface/DiffTalk/our_demos/3.mp4)
<br>④"Nations annual metting. First of all, thank you for your hardworking and advocacy on behalf of one of our state's greatest treasures. And that's our public lands."
![Nations annual metting. First of all, thank you for your hardworking and advocacy on behalf of one of our state's greatest treasures. And that's our public lands.](https://github.com/huanranchen/talkingface-toolkit/blob/main/talkingface/model/image_driven_talkingface/DiffTalk/our_demos/4.gif)
<br>⑤"Ninety days to get assad out of that country so that we can stabilise it and then we need to do the work of bringing in special forces. If we would have kept that base in Iraq that we should have kept when the president  couldn't get the status of forces agreement signed. I think that that would have been a big help. Now that there's new government in Iraq, I think we should revisit that and see if we can get."
![Ninety days to get assad out of that country so that we can stabilise it and then we need to do the work of bringing in special forces. If we would have kept that base in Iraq that we should have kept when the president  couldn't get the status of forces agreement signed. I think that that would have been a big help. Now that there's new government in Iraq, I think we should revisit that and see if we can get.]https://github.com/huanranchen/talkingface-toolkit/blob/main/talkingface/model/image_driven_talkingface/DiffTalk/our_demos/5.gif


## Paper
The paper "DiffTalk: Crafting Diffusion Models for Generalized Audio-Driven Portraits Animation" introduces a novel method for synthesizing high-fidelity, audio-driven talking head videos. This method, called DiffTalk, leverages the capabilities of Latent Diffusion Models. It focuses on generating temporally coherent talking head videos by using audio signals, reference face images, and landmarks as conditions. This approach allows for the creation of personalized, synchronized talking videos without the need for additional fine-tuning, even for different identities. The paper also demonstrates the method's ability to be adapted for higher-resolution synthesis without significant computational cost. Extensive experiments validate the effectiveness of DiffTalk in synthesizing high-quality talking head videos for a wide range of identities.

An overview of the DiffTalk is shown in the follwing figure.
<img src="method.png" width="932" height="391">

<br>
The visualization results are shown in the follwing figure.
<img src="result.png" width="587" height="492">


## Acknowledgement
This code is built on https://github.com/sstzal/DiffTalk

