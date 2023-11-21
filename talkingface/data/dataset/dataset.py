from talkingface.data.dataset import wav2lip_dataset

def factory(model, config, split):
    if model == 'wav2lip':
        return wav2lip_dataset(config, split)