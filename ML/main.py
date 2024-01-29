from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from ML.Model.cifar10.CNN import CIFAR10ModelCNN

model = CIFAR10ModelCNN()

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Setup logger
logger = TensorBoardLogger('tb_logs', name='cifar10_cnn')

# Initialize Trainer
trainer = Trainer(max_epochs=100, logger=logger)
trainer.fit(model, train_loader, val_loader)
