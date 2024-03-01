from mindspore.dataset import transforms, vision
from mindspore.dataset.vision import Inter

dataset_transform = {
    "CIFAR100": {
        "train": transforms.Compose([
            vision.RandomCrop(32, padding=4),
            vision.RandomHorizontalFlip(),
            vision.RandomColorAdjust(brightness=63 / 255),
            vision.ToTensor(),
            # vision.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ]),
        "test": transforms.Compose([
            vision.ToTensor(),
            # vision.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
        ])
    },

    "ImageNet1K": {
        "train": transforms.Compose([
            vision.RandomResizedCrop(224, interpolation=Inter.BICUBIC),
            vision.RandomHorizontalFlip(),
            vision.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
            vision.ToTensor(),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "test": transforms.Compose([
            vision.Resize(256, interpolation=Inter.BICUBIC),
            vision.CenterCrop(224),
            vision.ToTensor(),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    },

    "ImageNet100": {
        "train": transforms.Compose([
            vision.RandomResizedCrop(224, interpolation=Inter.BICUBIC),
            vision.RandomHorizontalFlip(),
            vision.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4),
            vision.ToTensor(),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

        ]),
        "test": transforms.Compose([
            vision.Resize(256, interpolation=Inter.BICUBIC),
            vision.CenterCrop(224),
            vision.ToTensor(),
            vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }
}