from dataclasses import dataclass
from typing import Optional, List, Union
from typing_extensions import Literal
from typeguard import typechecked
import os
from utils import TimmModelOptions

OptimizerOptions = Literal["SGD", "Adam"]
LossOptions = Literal["CrossEntropyLoss", "MSELoss"]

@typechecked
@dataclass
class Config():
    # Model
    model_name:TimmModelOptions
    model_num_classes:int
    model_pretrained:bool

    # Optimizer
    optim_name:OptimizerOptions
    optim_learning_rate:float
    optim_weight_decay:float
    optim_momentum:Optional[float]

    # Loss
    loss_name:LossOptions
    loss_class_weights:List[float]

    # Loop
    loop_epochs:int

    #Data
    data_root:Union[str, os.PathLike]

    def __init__(self, model_name:TimmModelOptions, model_num_classes:int, model_pretrained:bool, 
                 optim_name:OptimizerOptions, optim_learning_rate:float, optim_weight_decay:float, optim_momentum:Optional[float], 
                 loss_name:LossOptions, loss_class_weights:List[float],
                 loop_epochs:int,
                 data_root:Union[str, os.PathLike]):
        if not(model_num_classes == len(loss_class_weights)):
            raise ValueError("Number of classes in parameter 'model_num_classes' should be the same as the length of parameter 'loss_class_weight'. \
                              I.e. 'model_num_classes == len(loss_class_weights)' should be true.")
        # Model
        self.model_name = model_name
        self.model_num_classes = model_num_classes
        self.model_pretrained = model_pretrained

        # Optimizer
        self.optim_name = optim_name
        self.optim_learning_rate = optim_learning_rate
        self.optim_weight_decay = optim_weight_decay
        self.optim_momentum = optim_momentum

        # Loss
        self.loss_name = loss_name
        self.loss_class_weights = loss_class_weights

        # Loop
        self.loop_epochs = loop_epochs

        #Data
        self.data_root = data_root