import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, datasplit):

        """
        args: datasplit: str, 'train', 'val' or 'test'(这个参数必须要有, 提前将数据集划分为train, val和test三个部分,
                具体参数形式可以自己定,只要在你的dataset子类中可以获取到数据就可以, 
                对应的配置文件的参数为:train_filelist, val_filelist和test_filelist)
                
        """

        self.config = config
        self.split = datasplit

    def __getitem__(self):

        """
        Returns:
            data: dict, 必须是一个字典格式, 具体数据解析在model文件里解析

        """


        raise NotImplementedError