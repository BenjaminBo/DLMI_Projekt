import torch 
import timm
from config import Config
from typeguard import typechecked
from torchvision.transforms.functional import pil_to_tensor
import os
from dataloader import Dataloader

CWD = dir_path = os.path.dirname(os.path.realpath(__file__))

@typechecked
class TimmFinetuner():
    '''
    Class that runs through a finetuning process on a model imported from the `timm`-module 
    and determined through the `config.model_name` parameter.
    TODO Test/evaluation functionality
    TODO training log
    '''
    model:torch.nn.Module
    optimizer:torch.optim.Optimizer
    loss_name:str
    criterion:torch.nn.Module
    epochs:int
    train_set:Dataloader
    val_set:Dataloader
    test_set:Dataloader

    def __init__(self, config:Config):
        # Model
        self.model = timm.create_model(model_name=config.model_name, 
                                       pretrained=config.model_pretrained) #instantiate model
        
        if config.model_freeze:
            for param in self.model.parameters():
                param.requires_grad = False # freeze all parameters in model

        # create new model head that is trainable
        # TODO more flexible head
        #   -> maybe as parameter in config?
        if hasattr(self.model, "fc"):
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, config.model_num_classes)    
        elif hasattr(self.model, "classifier"):
            self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, config.model_num_classes)
        
        print(self.model)

        # Optimizer
        if config.optim_name == "SGD":
            self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                             momentum=config.optim_momentum,
                                             weight_decay=config.optim_weight_decay,
                                             lr=config.optim_learning_rate)
        elif config.optim_name == "Adam":
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              weight_decay=config.optim_weight_decay,
                                              lr=config.optim_learning_rate)

        # Loss
        self.loss_name = config.loss_name
        if self.loss_name == "CrossEntropyLoss":
            self.criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(config.loss_class_weights))
        elif self.loss_name == "MSELoss":
            self.criterion = torch.nn.MSELoss()

        # Loop
        self.epochs = config.loop_epochs

        #Data 
        self.train_set = Dataloader(root=os.path.join(config.data_root, "train"))
        self.val_set = Dataloader(root=os.path.join(config.data_root, "val"))
        self.test_set = Dataloader(root=os.path.join(config.data_root, "test"))

    def _train(self, epoch:int):      
        self.model.train()
        train_epoch_loss = 0.0
        train_batch_loss = 0.0
        for pil_image, gt_class in self.train_set:
            input_data = pil_to_tensor(pil_image) # convert pil Image to tensor
            input_data = input_data.float() # make sure tensor type is float
            input_data = input_data[None, :]    # HACK to add leading dim
                                                # TODO add batch implementation

            gt_class = torch.tensor([gt_class])
            output_data = self.model(input_data)
            
            train_batch_loss = self.criterion(output_data, gt_class)
            train_epoch_loss += train_batch_loss

        train_epoch_loss/len(self.train_set)
        print("Epoch [{0}/{1}]\n    Training Loss: {2}\n".format(epoch+1, self.epochs, train_epoch_loss))
        train_epoch_loss.backward()
        self.optimizer.step()

    def _val(self, epoch:int):
        self.model.eval()
        val_epoch_loss = 0.0
        val_batch_loss = 0.0
        for pil_image, gt_class in self.val_set:
            input_data = pil_to_tensor(pil_image) # convert pil Image to tensor
            input_data = input_data.float() # make sure tensor type is float
            input_data = input_data[None, :] # HACK add leading dim

            gt_class = torch.tensor([gt_class])
            output_data = self.model(input_data)
            
            val_batch_loss = self.criterion(output_data, gt_class)
            val_epoch_loss += val_batch_loss

        val_epoch_loss/len(self.train_set)
        print("    Validation Loss: {2}\n".format(epoch, self.epochs, val_epoch_loss))

    def training_loop(self):
        for e in range(self.epochs):
            self.optimizer.zero_grad()
            self._train(e)
            self._val(e)
                    