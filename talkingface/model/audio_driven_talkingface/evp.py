from talkingface.model.abstract_talkingface import AbstractTalkingFace
from logging import getLogger

import torch
import torch.nn as nn
import numpy as np
from talkingface.utils import set_color
from talkingface.model.audio_driven_talkingface.evp_arch import EmotionNet

class evp(AbstractTalkingFace):
    """Abstract class for talking face model."""

    def __init__(self,config):
        self.logger = getLogger()
        super(evp, self).__init__()
        self.model=EmotionNet()
        self.config=config
        self.opt_m = torch.optim.Adam(self.model.parameters(),
            lr=0.001, betas=(0.99, 0.99))
        self.CroEn_loss =  nn.CrossEntropyLoss()
        self.tripletloss = nn.TripletMarginLoss(margin=1)
        # self.train_loader = DataLoader(train_set, batch_size=config.batch_size,
        #                           num_workers=config.num_thread,
        #                           shuffle=True, drop_last=True)
        # self.val_loader = DataLoader(val_set, batch_size=config.batch_size,
        #                         num_workers=config.num_thread,
        #                         shuffle=True, drop_last=True)
    def calculate_loss(self, interaction, valid=False):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            dict: {"loss": loss, "xxx": xxx}
            返回是一个字典,loss 这个键必须有,它代表了加权之后的总loss。
            因为有时总loss可能由多个部分组成。xxx代表其它各部分loss
        """
        if valid:
            with torch.no_grad():
                fake = self.model(interaction["input"].float().to(self.config["device"]))
                # print(fake.shape,interaction["label"].shape)

                # loss_func = nn.CrossEntropyLoss()
                # pre = torch.tensor([0.8, 0.5, 0.2, 0.5], dtype=torch.float)
                # tgt = torch.tensor([1, 0, 0, 0], dtype=torch.float)
                # print(loss_func(pre, tgt))
                loss = self.CroEn_loss(fake.to(self.config["device"]),
                                       interaction["label"].squeeze(1).long().to(self.config["device"]))
        else:
            fake = self.model(interaction["input"].float().to(self.config["device"]))
            # print(fake.shape,interaction["label"].shape)

            # loss_func = nn.CrossEntropyLoss()
            # pre = torch.tensor([0.8, 0.5, 0.2, 0.5], dtype=torch.float)
            # tgt = torch.tensor([1, 0, 0, 0], dtype=torch.float)
            # print(loss_func(pre, tgt))
            loss=self.CroEn_loss(fake.to(self.config["device"]),
                                 interaction["label"].squeeze(1).long().to(self.config["device"]))
        return {"loss": loss}


    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            video/image numpy/tensor
        """
        raise NotImplementedError

    def generate_batch(self):

        """
        根据划分的test_filelist 批量生成数据。

        Returns: dict: {"generated_video": [generated_video], "real_video": [real_video] }
                    必须是一个字典数据, 且字典的键一个时generated_video, 一个是real_video,值都是列表，
                    分别对应生成的视频和真实的视频。且两个列表的长度应该相同。
                    即每个生成视频都有对应的真实视频（或近似对应的视频）。
        """
        x1=[]
        x2=[]
        for i in range(10):
            x1.append(torch.randn(1,3,32,32))
            x2.append(torch.randn(1, 3, 32, 32))
        result={"generated_video": [x1], "real_video": [x2]}
        return result

    def other_parameter(self):
        if hasattr(self, "other_parameter_name"):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return (
                super().__str__()
                + set_color("\nTrainable parameters", "blue")
                + f": {params}"
        )


