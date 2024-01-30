from .trainer import AbstractTrainer
from talkingface.evaluator import Evaluator
from logging import getLogger
class video_retalkingTrainer(AbstractTrainer):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.evaluator = Evaluator(config)
        self.logger = getLogger()
    def evaluate(self, model_file):
        """
        Evaluate the model based on the test data.

        args: load_best_model: bool, whether to load the best model in the training process.
                model_file: str, the model file you want to evaluate.

        """

        self.model.eval()

        datadict = self.model.generate_batch()
        # datadict = "../../results/1_1.mp4"
        eval_result = self.evaluator.evaluate(datadict)
        self.logger.info(eval_result)
