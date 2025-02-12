import torch 
import timm
from config import Config, LossOptions
from typeguard import typechecked
import os
from dataloader import Dataset, get_dataloader_from_dataset, Loader
from utils import PID, VisualizationUtils, ClassificationMetricsContainer, DatasetUtils, setup_logger
import datetime
from typing import Union, Tuple, Dict
import csv
from pandas import read_csv
import  logging

@typechecked
class TimmFinetuner():
    '''
    Class that runs through a finetuning process on a model imported from the `timm`-module 
    and determined through the `config.model_name` parameter.
    '''
    model:torch.nn.Module
    optimizer:torch.optim.Optimizer
    loss_name:LossOptions
    criterion:torch.nn.Module
    current_best_loss:float=float('inf')
    epochs:int
    train_set:Dataset#Loader
    val_set:Dataset#Loader
    checkpoint_name:str
    model_dir:Union[str, os.PathLike]
    complete_path:Union[str, os.PathLike]
    training_log_path:Union[str, os.PathLike]
    class_to_index:Dict[str,int]
    __LOGGER__:logging.Logger
    
    def __init__(self, config:Config):
        ################################# Paths #################################
        # name the model will be saved as
        self.model_dir = os.path.join(config.model_save_path, f"{str(datetime.datetime.now().time().strftime('%H_%M_%S'))}.{PID}")
        self.checkpoint_name = "model_params"
        self.complete_path = os.path.join(self.model_dir, self.checkpoint_name)
        self.training_log_path = os.path.join(self.model_dir,'training_log.csv')
        self.document_prediction_path = os.path.join(self.model_dir,'predictions_vs_gt.csv')
        os.makedirs(self.model_dir, exist_ok=True)

        ################################# Finetune Logger #################################
        self.__LOGGER__ = setup_logger(name= f"{__name__}_{config.model_name}_{PID}", 
                                       log_file= os.path.join(self.model_dir, f"{config.model_name}_training.log"), 
                                       level= logging.DEBUG)
        self.__LOGGER__.info("Instantiating Timm Finetuner with config:\n{0}".format(config.__dict__))

        # csv keeping track of training- and validation-loss and -metrics.
        with open(self.training_log_path, 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(["epoch", "train_loss", "val_loss", \
                         "train_accuracy", "train_precision", "train_recall", "train_f1", \
                         "val_accuracy", "val_precision", "val_recall", "val_f1"])
            f.close()

        with open(self.document_prediction_path, 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(["instance", "ground_truth", "predictions"])
            f.close()

        ################################# Model #################################
        self.__LOGGER__.info("Getting {0} model from Timm module.".format(config.model_name))
        if config.model_pretrained:
            self.__LOGGER__.info(" Downloading pretrained weights...") 

        try:
            self.model = timm.create_model(model_name=config.model_name,
                                           pretrained=config.model_pretrained) #instantiate model
        except RuntimeError as e:
            self.__LOGGER__.info(e)
            self.__LOGGER__.info(" There are no pretrained weights for the {0}-model architecture. Randomly initializing weights...".format(config.model_name))
            self.__LOGGER__.debug(" Calling timm.create_model() with `pretrained_weights=False`.")
            self.model = timm.create_model(model_name=config.model_name,
                                           pretrained=False) #instantiate model

        if config.model_freeze:
            self.__LOGGER__.info("Freezing all prameters.")
            # freeze all parameters in model
            for param in self.model.parameters():
                param.requires_grad = False 
        else: 
            self.__LOGGER__.info("Parameters will NOT be frozen")

        # create new model head that is trainable
        # TODO more flexible head
        #   -> maybe as parameter in config?
        if hasattr(self.model, "fc"):
            self.__LOGGER__.info("replacing {0}...".format(self.model.fc))
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, config.model_num_classes)    
        elif hasattr(self.model, "classifier"):
            self.__LOGGER__.info("replacing {0}...".format(self.model.classifier))
            self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, config.model_num_classes)
        else:
            raise AttributeError("Expected the model to have one of the following attributes as final model layer:\n {0}.\n \
                                 Since it doesn't, the model head can not be eschanged and not e trained.".format(["fc", "classifier"]))
        
        if torch.cuda.is_available(): # move to gpu if available
            self.model = self.model.cuda()

        self.__LOGGER__.info(self.model)

        ################################# Optimizer #################################
        self.__LOGGER__.info("Setting up {0}-optimizer".format(config.optim_name))
        if config.optim_name == "SGD":
            self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                             momentum=config.optim_momentum,
                                             weight_decay=config.optim_weight_decay,
                                             lr=config.optim_learning_rate)
        elif config.optim_name == "Adam":
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              weight_decay=config.optim_weight_decay,
                                              lr=config.optim_learning_rate)
        elif config.optim_name == "AdamW":
            self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                               weight_decay=config.optim_weight_decay,
                                               lr=config.optim_learning_rate,
                                               betas=config.optim_betas)
            
        ################################# Data #################################
        self.__LOGGER__.info("Setting up data:")

        if config.data_equal_instances:
            train_root = os.path.join(config.data_root, "train_equal_instances")
        else:
            train_root = os.path.join(config.data_root, "train")

        #Mapping indexes to class labels
        self.class_to_index  = DatasetUtils.get_labels(train_root)
        self.__LOGGER__.info(" Data class labels dictionary: {0}".format(self.class_to_index))
        self.class_labels = list(self.class_to_index.keys())
        self.__LOGGER__.info(" Data class labels: {0}".format(self.class_labels))
        

        #Training data
        # self.train_set = get_dataloader_from_dataset(root=train_root,
        #                                              augmentation_prob=config.augmentation_prob,
        #                                              class_map=self.class_to_index,
        #                                              batch_size=config.data_batch_size,
        #                                              input_size=config.data_input_size)
        self.train_set = Dataset(root=train_root,
                                 augmentation_prob=config.augmentation_prob,
                                 augmentation_viz=False,
                                 in_loader=False,
                                 class_map=self.class_to_index)
        self.__LOGGER__.info(" Training data: {0}".format(len(self.train_set)))

        #Validation data
        # self.val_set = get_dataloader_from_dataset(root=os.path.join(config.data_root, "val"),
        #                                            class_map=self.class_to_index,
        #                                            batch_size=config.data_batch_size)
        self.val_set = Dataset(root=os.path.join(config.data_root, "val"),
                               augmentation_prob=0.0,
                               augmentation_viz=False,
                               in_loader=False,
                               class_map=self.class_to_index)
        self.__LOGGER__.info(" Validation data: {0}".format(len(self.val_set)))

        ################################# Loss #################################
        self.loss_name = config.loss_name
        self.__LOGGER__.info("Setting up {0}:".format(self.loss_name))

        if config.loss_compute_class_weights:
            #Automatically compute class weights for loss computation using the training class balance.
            weights = torch.Tensor(DatasetUtils.compute_weights(dataset_root=train_root,
                                                                labels_folders=self.class_labels))
        else:   
            weights = torch.Tensor(config.loss_class_weights)

        # Inform user of class weights/balances.
        self.__LOGGER__.info("class weights AND class balance in training data:\n {0}".format(weights))
        self.__LOGGER__.info("class balance in validation data (not used for training):\n {0}".format(DatasetUtils.compute_weights(dataset_root=os.path.join(config.data_root, "val"),
                                                                                                                                   labels_folders=self.class_labels)))
        if os.path.exists(os.path.join(config.data_root, "test")):
            self.__LOGGER__.info("class balance in test data (not used for training):\n {0}".format(DatasetUtils.compute_weights(dataset_root=os.path.join(config.data_root, "test"),
                                                                                                                                 labels_folders=self.class_labels)))

        if self.loss_name == "CrossEntropyLoss":
            self.criterion = torch.nn.CrossEntropyLoss(weight=weights,
                                                       reduction='mean')
        if self.loss_name == "BCELoss":
            self.criterion = torch.nn.BCELoss(reduction='mean', 
                                              weight=weights)
        elif self.loss_name == "MSELoss":
            self.criterion = torch.nn.MSELoss()

        ################################# Loop parameters/utilities #################################
        self.epochs = config.loop_epochs
        self.metrics = ClassificationMetricsContainer()

    def _single_epoch_loop(self, dataloader:Dataset, backpropagate:bool=False, document_predictions:bool=False) -> Tuple[float, dict]:
        '''
        Implements the generic loop for a model, during training, cross validation or testing.
        ## Computes in every case:
        1. forward pass,
        2. loss from model output, and
        3. metrics using ground truth and prediction
        ## If `backpropagate` is `True`, also computes:
        4. `model.zero_grad()`,
        5. `batch_loss.backward()` and
        6. `self.optimizer.step()`.
        for parameter optimization (should not be `True` when validating or testing).

        :param dataloader:
        :param backpropagate:
        :return: 
        '''
        f=None
        if document_predictions:
            f = open(self.document_prediction_path, 'a', encoding='utf-8', newline='')

        epoch_loss = 0.0
        for model_input, gt_class, filename in dataloader: # loop over batch 
            # device = self.model.device
            model_output = self.model(model_input) # forwards pass; compute output class prediction from model

            # determine prediction
            # class_probabilities = torch.nn.functional.softmax(model_output, dim=1) # formulate model output to class probabilities using softmax as final activation.
            # predicted_class = torch.argmax(model_output, dim=1).int().cpu()#.unsqueeze(0) # extract class prediction from probabilities.
            
            # class_probabilities = class_probabilities.cpu()
            gt_class = gt_class.detach().cpu()

            if self.loss_name == "CrossEntropyLoss": 
                predicted_class = torch.argmax(model_output, dim=1).int()
                batch_loss = self.criterion(model_output.detach().cpu(), gt_class) # pass batch class probabbilities and batch ground truth to criterion which computes batch loss.
            # elif self.loss_name == "BCELoss":
            #     #TODO
            #     batch_loss = self.criterion(class_probabilities, gt_class)
            #     predicted_class = torch.argmax(class_probabilities, dim=1).int()
            # elif self.loss_name == "MSELoss":
            #     # TODO
            #     predicted_class = predicted_class.float()
            #     batch_loss = self.criterion(predicted_class, gt_class)

            epoch_loss += batch_loss.item() # add up loss per batch to get overall epoch loss.

            predicted_class = predicted_class.detach().cpu().numpy().tolist()
            gt_class = gt_class.numpy().tolist()
            self.metrics.append_pred_and_gt(predicted_class, gt_class) # pass predicted class and gt class to metric computation

            if backpropagate:
                # backpropatgation. 
                # Should only be true for training section of an epoch.
                self.optimizer.zero_grad(set_to_none=True)
                batch_loss.requires_grad = True
                batch_loss.backward()
                self.optimizer.step()
            
            if document_predictions:
                for instance, gt, pred in zip(filename, gt_class, predicted_class):
                    wr = csv.writer(f)
                    wr.writerow([instance, gt, pred])

        epoch_loss = epoch_loss/len(dataloader) # normalize loss over batch (len(dataloader) respects the batchsize)
        metrics = self.metrics.compute_metrics() # compute metrics from predictions and ground truths that have been passed.
        self.metrics.reset() # reset metric computation for further use.

        if document_predictions:
            f.close()

        return epoch_loss, metrics

    def _train(self) -> Tuple[float, dict]:
        '''
        Training section in the training-loop of a model. 
        Model will be set to "train()"-mode, After which "_single_epoch_loop()"  will be called, in which 
        foward-, loss-, backpropagation-, optimization- and metric-computation takes place.
        Metrics and loss will be returned and logged.
        
        :return: float representing loss and dictionary containing metrics
        '''      
        self.__LOGGER__.info("Training Loop...")
        self.model.train() # set model to training mode
        train_epoch_loss, metrics = self._single_epoch_loop(dataloader=self.train_set, 
                                                            backpropagate=True, 
                                                            document_predictions=False) # call function with training-data and set flag to true that allows it to compute backpropagation.
        self.__LOGGER__.info("    Training {0}: {1}".format(self.loss_name, train_epoch_loss))# log loss
        
        return train_epoch_loss, metrics

    def _val(self) -> Tuple[float, dict]:
        '''
        Set model to "eval()"-mode and only do forward-, loss-, and metric- computation.
        Metrics and loss will be returned and logged.

        :return: float representing loss and dictionary containing metrics
        '''
        self.__LOGGER__.info("Validation loop...")
        self.model.eval() # set model to evalutation mode
        val_epoch_loss, metrics = self._single_epoch_loop(dataloader=self.val_set,
                                                          backpropagate=False,
                                                          document_predictions=True) #call function with validation-dataloader and without allowing it to compute backpropagation.
        self.__LOGGER__.info("    Validation {0}: {1}\n".format(self.loss_name, val_epoch_loss)) # log loss

        return val_epoch_loss, metrics
    
    def _log_and_save(self, training_loss:float, validation_loss:float, epoch:int, train_metrics:dict, val_metrics:dict) -> None:
        '''

        '''
        with open(self.training_log_path, 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([epoch+1, training_loss, validation_loss, \
                        train_metrics["accuracy"], train_metrics["precision"], train_metrics["recall"], train_metrics["f1"], \
                        val_metrics["accuracy"], val_metrics["precision"], val_metrics["recall"], val_metrics["f1"]])
            f.close()
        
        if validation_loss < self.current_best_loss:
            self.current_best_loss = validation_loss
            torch.save(self.model.state_dict(), self.complete_path)

    def _visualize_training(self, val_metrics:dict, train_metrics:dict):
        training_log = read_csv(self.training_log_path)
        VisualizationUtils.plot_vs(metric_name="loss", 
                                   graphs_values=[training_log["train_loss"], training_log["val_loss"]], 
                                   graphs_names=["training", "validation"],
                                   savefig_path=os.path.join(self.model_dir, "Loss training vs validation.png"))
        
        VisualizationUtils.viz_confusion_matrix(cm=train_metrics["confusion_matrix"],
                                                labels=self.class_labels,
                                                savefig_path=os.path.join(self.model_dir, "confusion_matrix_train.png"))
        
        VisualizationUtils.viz_confusion_matrix(cm=val_metrics["confusion_matrix"],
                                                labels=self.class_labels,
                                                savefig_path=os.path.join(self.model_dir, "confusion_matrix_val.png"))
        
        VisualizationUtils.viz_roc(fpr=val_metrics["fpr"],
                                   tpr=val_metrics["tpr"],
                                   auc=val_metrics["auc"], 
                                   savefig_path=os.path.join(self.model_dir, "roc_val.png"))

    def training_loop(self):
        for e in range(self.epochs):
            self.__LOGGER__.info("Epoch [{0}/{1}]".format(e+1, self.epochs))
            training_loss, train_metrics = self._train()
            validation_loss, val_metrics = self._val()
            self._log_and_save(training_loss=training_loss,
                               validation_loss=validation_loss,
                               epoch=e,
                               train_metrics=train_metrics,
                               val_metrics=val_metrics)
            self._visualize_training(val_metrics=val_metrics,
                                     train_metrics=train_metrics)