from config import Config, create_config_list, TimmModelOptions
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from finetune import TimmFinetuner
from utils import CWD, DataPrepUtils
from typing import Union, List
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
    DataPrepUtils.dataset_with_classfolders(root_folder=root_folder)

def equalize_class_instances_dataset(dataset_path:Union[os.PathLike, str]):
    DataPrepUtils.equalize_classes(dataset_root=dataset_path)

def finetune(model_name:str, dataset_name:Union[str, os.PathLike]):
    model_names:List[TimmModelOptions] = ["resnet101"]
    configs:List[Config] = create_config_list(data_root=os.path.join(CWD, dataset_name) if not(os.sep in dataset_name) else dataset_name,
                                              augmentation_prob = [0.8, 0.0],
                                              data_equal_instances = [False, True],
                                              model_name=model_names)
    for config in configs:
        # try:
        tf = TimmFinetuner(config=config)
        tf.training_loop()
        # except Exception as e:
        #     LOGGER.info("Config {0} throws the following exception:\n{1}".format(config.__dict__, e))

''' Arguments '''
parser = argparse.ArgumentParser(description = 'Main')
subparsers = parser.add_subparsers(help='Choose an action', dest="action")

# data prep
prep_parser = subparsers.add_parser(name="prep_data")
prep_parser.add_argument("-r", "--root_folder")

# data equalize
prep_parser = subparsers.add_parser(name="equalize_classes")
prep_parser.add_argument("-r", "--root_folder")

# finetuning
finetune_parser = subparsers.add_parser(name="finetune")
finetune_parser.add_argument("-m", "--model_name", default="resnet101")
finetune_parser.add_argument("-d", "--dataset", default="")

args = parser.parse_args()

if args.action == "prep_data":
    prep_dataset(root_folder=args.root_folder)
if args.action == "equalize_classes":
    equalize_class_instances_dataset(dataset_path=args.root_folder)
if args.action == "finetune":
    finetune(model_name=args.model_name,
             dataset_name=args.dataset)