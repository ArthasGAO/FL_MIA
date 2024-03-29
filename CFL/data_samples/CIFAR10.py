from torch.utils.data import Dataset, random_split
from torchvision import transforms as T
from torchvision.datasets import CIFAR10


class CIFAR10Dataset(Dataset):
    def __init__(self, normalization="cifar10", loading="torchvision", sub_id=0, number_sub=1, num_workers=4,
                 batch_size=32, iid=True, root_dir="./data"):
        super().__init__()
        self.train_set = None
        self.test_set = None
        self.val_set = None
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.iid = iid
        self.root_dir = root_dir
        self.loading = loading
        self.normalization = normalization
        self.mean = self.set_normalization(normalization)["mean"]
        self.std = self.set_normalization(normalization)["std"]

        train_set = self.get_dataset(
            train=True,
            transform=T.Compose(
                [
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std),
                ]
            ),
        )
        data_train, data_val = random_split(train_set, [round(len(train_set) * (1 - 0.1)),
                                                        round(len(train_set) * 0.1)])

        self.train_set = data_train
        self.val_set = data_val

        self.test_set = self.get_dataset(
            train=False,
            transform=T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(self.mean, self.std),
                ]
            ),
        )

    def set_normalization(self, normalization):
        # Image classification on the CIFAR10 dataset - Albumentations Documentation
        # https://albumentations.ai/docs/autoalbument/examples/cifar10/
        if normalization == "cifar10":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif normalization == "imagenet":
            # ImageNet - torchbench Docs https://paperswithcode.github.io/torchbench/imagenet/
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            raise NotImplementedError
        return {"mean": mean, "std": std}

    def get_dataset(self, train, transform, download=True):
        if self.loading == "torchvision":
            dataset = CIFAR10(
                root=self.root_dir,
                train=train,
                transform=transform,
                download=download,
            )
        elif self.loading == "custom":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return dataset
