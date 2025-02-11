import timm.data
from typing import Tuple, Union, Dict, Optional
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.utils import save_image
import random
from typeguard import typechecked
from utils import CWD, os
import numpy as np
from PIL import Image
from config import Config
import collections.abc
import torch

Loader = Union[timm.data.loader.PrefetchLoader]

random.seed(10)

@typechecked
class Dataset(timm.data.dataset.ImageDataset):

    def __init__(self, root, augmentation_prob:float, augmentation_viz:bool, in_loader:bool,
                 reader=None, split='train', class_map=None, load_bytes=False, input_img_mode='RGB', transform=None, target_transform=None, **kwargs):
        super().__init__(root, reader, split, class_map, load_bytes, input_img_mode, transform, target_transform, **kwargs)

        self.augmentation_prob = augmentation_prob
        self.visualize = augmentation_viz
        self.in_loader = in_loader

        self.viz_counter = 0

    def _live_augemtation(self, model_input):# -> Image:
        '''
        8 augmentations
        '''
        # NOTE: 
        # torchvision.transforms draw random numbers per usage on input.
        Transform = []

        # FUNCTIONAL AUGMENTATION (no random) 
        ## rotation [90, 180, 270]
        p = random.random()
        if p < self.augmentation_prob:
            RotationDegrees = [90, 180, 270]
            RotationDegree = random.randint(0,2)
            RotationDegree = RotationDegrees[RotationDegree]
            model_input = F.rotate(model_input, RotationDegree)

        ## rotation [-10, 10]
        p = random.random()
        if p < self.augmentation_prob:
            RotationRange = random.randint(-10,10) 
            model_input = F.rotate(model_input, RotationRange)

        ## horizontal/vertical flip
        p = random.random()
        if p < self.augmentation_prob:
            model_input = F.hflip(model_input)
        p = random.random()
        if p < self.augmentation_prob:
            model_input = F.vflip(model_input)

        # # ColorJitter
        # p = random.random()
        # if p < self.augmentation_prob:
        #     Transform.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.0))
        
        ## GaussianBlur
        p = random.random()
        if p < self.augmentation_prob:
            Transform.append(T.GaussianBlur(11, (0.1, 5)))

        ## RandomPosterize
        # p = random.random()
        # if p < self.augmentation_prob:
        #     bits = random.randint(1,7) 
        #     Transform.append(T.RandomPosterize(bits, 1))

        ## RandomEqualize
        p = random.random()
        if p < self.augmentation_prob:
            Transform.append(T.RandomEqualize(1))    
        
        #RandomInvert
        #RandomAdjustSharpness
        #RandomAutocontrast

        # apply transformations 
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        model_input = Transform(model_input)

        if self.visualize:
            path = os.path.join(CWD, "augmentation_visualization")
            os.makedirs(path, exist_ok=True)
            save_image(model_input, os.path.join(path, f"{str(self.viz_counter)}.png"))
            self.viz_counter += 1

        # model_input = np.array(model_input)

        return model_input

    def __getitem__(self, index):# -> Tuple[Image.Image, int]:
        model_input, gt_class = super().__getitem__(index)
        filename = os.path.basename(self.reader.samples[index][0])

        if not(self.in_loader):
            model_input = self._live_augemtation(model_input=model_input)

            model_input = model_input.float()
            model_input = model_input.unsqueeze(0) if len(model_input.shape) < 4 else model_input

            gt_class = [gt_class] if not(isinstance(gt_class, collections.abc.Sequence)) else gt_class
            gt_class = torch.Tensor(gt_class).type(torch.LongTensor)

            filename = [filename] #if not(isinstance(filename, collections.abc.Sequence)) else filename

            if torch.cuda.is_available(): # move to gpu if available
                model_input = model_input.cuda()
                gt_class = gt_class.cuda()

        return model_input, gt_class, filename
        
@typechecked
def get_dataloader_from_dataset(root:Union[str, os.PathLike], class_map:Dict[str, int], batch_size:int,
                                augmentation_prob:float=0.0, visualize:bool=False, input_size:Optional[Union[int, Tuple[int, int], Tuple[int, int, int]]]=None) -> Loader:
    dataset = Dataset(root=root, 
                      augmentation_prob=augmentation_prob, 
                      visualize=visualize, 
                      class_map=class_map,
                      in_loader=True)
    if input_size is None:
        input_size = (3, dataset[0][0].size[0], dataset[0][0].size[1])
    dataloader = timm.data.create_loader(dataset=dataset,
                                         input_size=input_size,
                                         batch_size=batch_size)
    return dataloader