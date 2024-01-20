import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from talkingface.utils.fastspeech2_transformerblock import Encoder, Decoder, PostNet, VarianceAdaptor
from talkingface.utils.fastspeech2_transformerblock import get_mask_from_lengths
from talkingface.model.abstract_talkingface import AbstractTalkingFace

class FastSpeech2(AbstractTalkingFace):
    """ FastSpeech2 """
    def __init__(self, config):
        super(FastSpeech2, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.variance_adaptor = VarianceAdaptor(
            config)
        self.decoder = Decoder(config)
        self.mel_linear = nn.Linear(
            config["transformer"]["decoder_hidden"],
            config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if config["multi_speaker"]:
            with open(
                os.path.join(
                    config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                config["transformer"]["encoder_hidden"],
            )

0
