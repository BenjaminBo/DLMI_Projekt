from config import Config
from finetune import TimmFinetuner
import os 
from utils import DataPrepUtils, CWD
from typing import Union
import argparse

def prep_dataset(root_folder:Union[str, os.PathLike]):
    DataPrepUtils.prep_capsule_dataset(root_folder=root_folder)

def finetune(model_name:str, dataset_name:Union[str, os.PathLike]):
    config = Config(model_name=model_name, 
                    model_num_classes=2, 
                    model_pretrained=True,
                    optim_name="SGD", 
                    optim_learning_rate=0.01, 
                    optim_weight_decay=0.001, 
                    optim_momentum=0.9,
                    loss_name="CrossEntropyLoss", 
                    loss_class_weights=[1.0, 1.0],
                    loop_epochs=5,
                    data_root=os.path.join(CWD, dataset_name) if not(os.sep in dataset_name) else dataset_name)
    tf = TimmFinetuner(config=config)
    tf.training_loop()

''' Arguments '''
parser = argparse.ArgumentParser(description = 'Main')
subparsers = parser.add_subparsers(help='Choose an action', dest="action")

# data prep
prep_parser = subparsers.add_parser(name="prep_data")
prep_parser.add_argument("-r", "--root_folder", default=os.path.join(CWD, "project_capsule_dataset"))

# finetuning
finetune_parser = subparsers.add_parser(name="finetune")
finetune_parser.add_argument("-m", "--model_name", default="mobilenetv3_small_050")
finetune_parser.add_argument("-d", "--dataset", default="capsule_dataset")

args = parser.parse_args()

if args.action == "prep_data":
    prep_dataset(root_folder=args.root_folder)
elif args.action == "finetune":
    finetune(model_name=args.model_name,
             dataset_name=args.dataset)