from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
from typing_extensions import Literal
from typeguard import typechecked
import os
from utils import TimmModelOptions

OptimizerOptions = Literal["SGD", "Adam"]
LossOptions = Literal["CrossEntropyLoss", "BCELoss", "MSELoss"]

@typechecked
@dataclass
class Config():
    # Model
    model_name:TimmModelOptions
    model_num_classes:int
    model_pretrained:bool
    model_freeze:bool
    model_save_path:Union[str, os.PathLike]

    # Optimizer
    optim_name:OptimizerOptions
    optim_learning_rate:float
    optim_weight_decay:float
    optim_momentum:Optional[float]

    # Loss
    loss_name:LossOptions
    loss_compute_class_weights:bool

    # Loop
    loop_epochs:int

    # Data
    data_root:Union[str, os.PathLike]
    data_batch_size:int

    # Augmentation
    augmentation_prob:float

    #DEFAULT PARAMETERS
    # Loss
    loss_class_weights:Optional[List[float]] = None

    # Data
    data_input_size:Optional[Union[int, Tuple[int, int], Tuple[int, int, int]]] = None
    data_equal_instances:bool = False

    # Augmentation
    augmentation_viz:bool = False