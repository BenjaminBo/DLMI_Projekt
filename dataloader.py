import timm.data

class Dataloader(timm.data.dataset.ImageDataset):

    def __getitem__(self, index):
        return super().__getitem__(index)
        ## TODO augmentation
