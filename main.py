from config import Config
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from finetune import TimmFinetuner
from utils import CWD, DataPrepUtils
from typing import Union
import argparse
import datetime
import logging

# Create logs folder if not given
this_directory = os.path.dirname(os.path.realpath(__file__))
logs_path = os.path.join(this_directory, "logs", str(datetime.datetime.now().date()))
if not(os.path.exists(logs_path)):
    os.makedirs(logs_path)

# Create logger using process id
PID = os.getpid()
LOGGER = logging.getLogger(f"{__name__}_{PID}")
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)#remove loggers that are hierarchically above this one.
logging.basicConfig(filename=os.path.join(logs_path, f"main_{PID}.log"), filemode='a', format='%(process)d,%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)#formatting messages in logfile
logging.getLogger().addHandler(logging.StreamHandler())#print logs in terminal
logging.captureWarnings(capture=True)#log warnings

def prep_dataset(root_folder:Union[str, os.PathLike]):
    DataPrepUtils.prep_capsule_dataset(root_folder=root_folder)

def finetune(model_name:str, dataset_name:Union[str, os.PathLike]):
    config = Config(model_name=model_name, 
                    model_num_classes=2, 
                    model_pretrained=True,
                    model_freeze=True,
                    model_save_path = os.path.join(CWD, "models", str(datetime.datetime.now().date())),
                    optim_name="SGD", 
                    optim_learning_rate=0.005, 
                    optim_weight_decay=0.001, 
                    optim_momentum=0.9,
                    loss_name="CrossEntropyLoss", 
                    loss_compute_class_weights=True,
                    loop_epochs=200,
                    data_root=os.path.join(CWD, dataset_name) if not(os.sep in dataset_name) else dataset_name,
                    augmentation_prob=0.8,
                    augmentation_viz=False)
    LOGGER.info("Instantiated confing:\n {0}".format(config.__dict__))
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
finetune_parser.add_argument("-d", "--dataset", default="")

args = parser.parse_args()

if args.action == "prep_data":
    prep_dataset(root_folder=args.root_folder)
if args.action == "finetune":
    finetune(model_name=args.model_name,
             dataset_name=args.dataset)