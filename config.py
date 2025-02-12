from dataclasses import dataclass, asdict
from typing import Optional, List, Union, Tuple
from typing_extensions import Literal
from typeguard import typechecked
import os
from utils import TimmModelOptions, CWD
import datetime
from itertools import product

OptimizerOptions = Literal["SGD", "Adam", "AdamW"]
LossOptions = Literal["CrossEntropyLoss", "BCELoss", "MSELoss"]

@typechecked
@dataclass
class Config():
    data_root:Union[str, os.PathLike]

    # Model
    model_name:TimmModelOptions = "resnet101"
    model_num_classes:int = 2
    model_pretrained:bool = True
    model_freeze:bool = True
    model_save_path:Union[str, os.PathLike] = os.path.join(CWD, "models", str(datetime.datetime.now().date()))

    # Optimizer
    optim_name:OptimizerOptions = "Adam"
    optim_learning_rate:float = 0.005
    optim_weight_decay:float = 0.001
    optim_momentum:Optional[float] = 0.9
    optim_betas:Optional[Tuple[float, float]] = (0.9, 0.999)

    # Loss
    loss_name:LossOptions = "CrossEntropyLoss"
    loss_compute_class_weights:bool = True
    loss_class_weights:Optional[List[float]] = None

    # Loop
    loop_epochs:int = 20

    # Data
    data_batch_size:int = 5
    data_input_size:Optional[Union[int, Tuple[int, int], Tuple[int, int, int]]] = None
    data_equal_instances:bool = True

    # Augmentation
    augmentation_prob:float = 0.0
    augmentation_viz:bool = False
    
def create_config_list(data_root: Union[str, os.PathLike], **kwargs) -> List[Config]:
    # Get default config as dict and remove data_root to avoid conflicts
    default_config = asdict(Config(data_root))
    default_config.pop("data_root", None)
    
    # Separate single values and lists
    list_params = {k: v for k, v in kwargs.items() if isinstance(v, list)}
    single_params = {k: v for k, v in kwargs.items() if k not in list_params}
    
    if not list_params:
        return [Config(data_root=data_root, **single_params)]
    
    # Generate combinations of list values
    keys, values = zip(*list_params.items()) if list_params else ([], [])
    param_combinations = [dict(zip(keys, v)) for v in product(*values)] if list_params else [{}]
    
    # Create list of Config objects
    config_list = [
        Config(data_root=data_root, **{**default_config, **single_params, **param_comb})
        for param_comb in param_combinations
    ]
    
    return config_list
