from logging import getLogger

import torch
import torch.nn as nn
import numpy as np
#from talkingface.utils import set_color

class AbstractSpeech(nn.Module):
    """Abstract class for talking face model."""

    def __init__(self):
        self.logger = getLogger()
        super(AbstractSpeech, self).__init__()

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            dict: {"loss": loss, "xxx": xxx}
            返回是一个字典,loss 这个键必须有,它代表了加权之后的总loss。
            因为有时总loss可能由多个部分组成。xxx代表其它各部分loss
        """
        raise NotImplementedError
    
    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            video/image numpy/tensor
        """
        raise NotImplementedError
    
    def generate_batch():

        """
        根据划分的test_filelist 批量生成数据。

        Returns: dict: {"generated_audio": [generated_audio], "real_audio": [real_audio] }
                    必须是一个字典数据, 且字典的键一个时generated_audio, 一个是real_audio,值都是列表，
                    分别对应生成的音频和真实的音频。且两个列表的长度应该相同。
                    即每个生成音频都有对应的真实音频（或近似对应的音频）。
        """

        raise NotImplementedError

    def other_parameter(self):
        if hasattr(self, "other_parameter_name"):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)


