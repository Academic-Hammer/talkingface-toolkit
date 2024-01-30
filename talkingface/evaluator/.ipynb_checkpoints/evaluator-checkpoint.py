from talkingface.evaluator.register import metrics_dict

class Evaluator(object):
    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config["metrics"]]
        self.metric_class = {}

        for metric in self.metrics:
            if metric not in metrics_dict:
                raise ValueError(f"Metric '{metric}' is not defined.")
            self.metric_class[metric] = metrics_dict[metric](self.config)
    
    def evaluate(self, datadict):

        result_dict = {}

        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(datadict)
            result_dict[metric] = metric_val
        
        return result_dict
