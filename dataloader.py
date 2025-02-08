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

Loader = Union[timm.data.loader.PrefetchLoader]

random.seed(10)

@typechecked
class Dataset(timm.data.dataset.ImageDataset):

    def __init__(self, root, reader=None, split='train', class_map=None, load_bytes=False, input_img_mode='RGB', transform=None, target_transform=None, augmentation_prob:float=0.0, visualize:bool=False, **kwargs):
        super().__init__(root, reader, split, class_map, load_bytes, input_img_mode, transform, target_transform, **kwargs)
        self.augmentation_prob = augmentation_prob
        self.visualize = visualize
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

        ## ColorJitter
        # p = random.random()
        # if p < self.augmentation_prob:
        #     Transform.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3))
        
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
        model_input, target = super().__getitem__(index)

        model_input = self._live_augemtation(model_input=model_input)

        # if type(model_input) == np.ndarray:
        #     model_input = Image.fromarray(model_input.astype('uint8'), 'RGB')

        return model_input, target
        
@typechecked
def get_dataloader_from_dataset(root:Union[str, os.PathLike], class_map:Dict[str, int], batch_size:int,
                                augmentation_prob:float=0.0, visualize:bool=False, input_size:Optional[Union[int, Tuple[int, int], Tuple[int, int, int]]]=None) -> Loader:
    dataset = Dataset(root=root, 
                      augmentation_prob=augmentation_prob, 
                      visualize=visualize, 
                      class_map=class_map)
    if input_size is None:
        input_size = (3, dataset[0][0].size[0], dataset[0][0].size[1])
    dataloader = timm.data.create_loader(dataset=dataset,
                                         input_size=input_size,
                                         batch_size=batch_size)
    return dataloader