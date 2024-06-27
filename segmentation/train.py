import torch
import torch.nn as nn
from segmentation.data_loader import Dataset
from torch.utils.data import DataLoader
from segmentation.U_Net import U_Net

# Provided Trainer class
from trainmodel.trainer import Trainer  # Ensure Trainer is imported from the correct file


# Set the training hyperparameters
datadir = '/mnt/c/Unet/segDataset/'

batch_size = 4
lr = 0.001
epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the dataset and dataloader
train_dataset = Dataset(datadir, split='train', augment=False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_dataset = Dataset(datadir, split='val',
                      augment=False)  # Augment for change image form like crop and mirroring
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Initialize the model output_ch is the num of classe
model = U_Net(3, 2)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()


# Create an instance of Trainer
trainer = Trainer(model=model,
                  train_dataloader=train_dataloader,
                  val_dataloader=val_dataloader,
                  optimizer=optimizer,
                  loss_func=loss_func,
                  device=device)

# Start training

trainer.run(epochs=epochs)
