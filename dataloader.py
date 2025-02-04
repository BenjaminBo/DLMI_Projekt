import timm.data
import collections.abc
from torch import Tensor, LongTensor, cuda
from typing import Tuple
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.utils import save_image
import random
from typeguard import typechecked
from PIL.Image import Image
from utils import CWD, os

random.seed(10)

@typechecked
class Dataloader(timm.data.dataset.ImageDataset):

    def __init__(self, root, reader=None, split='train', class_map=None, load_bytes=False, input_img_mode='RGB', transform=None, target_transform=None, augmentation_prob:float=1.0, visualize:bool=False, **kwargs):
        super().__init__(root, reader, split, class_map, load_bytes, input_img_mode, transform, target_transform, **kwargs)
        self.augmentation_prob = augmentation_prob
        self.visualize = visualize
        self.viz_counter = 0

    def _live_augemtation(self, model_input:Image) -> Tensor:
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

        ## ColorJitter
        p = random.random()
        if p < self.augmentation_prob:
            Transform.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3))
        
        ## GaussianBlur
        p = random.random()
        if p < self.augmentation_prob:
            Transform.append(T.GaussianBlur(11, (0.1, 5)))

        ## RandomPosterize
        p = random.random()
        if p < self.augmentation_prob:
            bits = random.randint(1,7) 
            Transform.append(T.RandomPosterize(bits, 1))

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
        model_input:Tensor = Transform(model_input)

        if self.visualize:
            path = os.path.join(CWD, "augmentation_visualization")
            os.makedirs(path, exist_ok=True)
            save_image(model_input, os.path.join(path, f"{str(self.viz_counter)}.png"))
            self.viz_counter += 1

        return model_input

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        model_input, target = super().__getitem__(index)

        model_input = self._live_augemtation(model_input=model_input)

        # Format image to model input in tensors
        model_input = model_input.float() # make sure tensor type is float
        model_input = model_input.unsqueeze(0) if len(model_input.shape) < 4 else model_input

        # Format classes to tensors
        gt_class = [target] if not(isinstance(target, collections.abc.Sequence)) else target
        gt_class = Tensor(gt_class).type(LongTensor)

        return model_input, gt_class