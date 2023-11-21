import torch
from talkingface.utils import EvaluatorType

class AbstractMetric(object):
    """:class:`AbstractMetric` is the base object of all metrics. If you want to
        implement a metric, you should inherit this class.

    Args:
        config (Config): the config of evaluator.
    """

    smaller = False

    def __init__(self, config):
        self.decimal_place = config["metric_decimal_place"]

    def calculate_metric(self, dataobject):
        """Get the dictionary of a metric.

        Args:
            dataobject: (dict): it contains all the information needed to calculate metrics.

        Returns:
            dict: such as ``{LSE-C': 0.0000}``
        """
        raise NotImplementedError("Method [calculate_metric] should be implemented.")
    

class SyncMetric(AbstractMetric):
    """Base class for all Sync metrics. If you want to implement a sync metric, you can inherit this class.
    """
    
    metric_type = EvaluatorType.SYNC
    metric_need = ["generated_video"]
    def __init__(self, config):
        super(SyncMetric, self).__init__(config)
    
    def get_videolist(self, dataobject):
        """Get the list of videos.

        Args:
            dataobject(DataStruct): (dict): it contains all the information needed to calculate metrics.

        Returns:
            list: a list of videos.
        """
        return dataobject["generated_video"]

    def metric_info(self, dataobject):
        """Calculate the value of the metric.

        Args:
            dataobject(DataStruct): it contains all the information needed to calculate metrics.

        Returns:
            dict: {"LSE-C": LSE_C, "LSE-D": LSE_D}
        """
        raise NotImplementedError("Method [metric_info] should be implemented.")
    
class VideoQMetric(AbstractMetric):
    """Base class for all Video Quality metrics. If you want to implement a Video Quality metric, you can inherit this class.
    """

    metric_type = EvaluatorType.VIDEOQ
    def __init__(self, config):
        super(VideoQMetric, self).__init__(config)

    def get_videopair(self, dataobject):
        return list(zip(dataobject["generated_video"], dataobject["real_video"]))
    
    def metric_info(self, dataobject):
        """Calculate the value of the metric.

        Args:
            dataobject(DataStruct): (dict): it contains all the information needed to calculate metrics.

        Returns:
            float: the value of the metric.
        """
        raise NotImplementedError("Method [metric_info] should be implemented.")
    
# class AudioQMetric(AbstractMetric):
#     """Base class for all Audio Quality metrics. If you want to implement a Audio Quality metric, you can inherit this class.
#     """
#     def __init__(self, config):
#         super(SyncMetric, self).__init__(config)

#     def metric_info(self, dataobject):
#         """Calculate the value of the metric.

#         Args:
#             dataobject(DataStruct): it contains all the information needed to calculate metrics.

#         Returns:
#             float: the value of the metric.
#         """
#         raise NotImplementedError("Method [metric_info] should be implemented.")

