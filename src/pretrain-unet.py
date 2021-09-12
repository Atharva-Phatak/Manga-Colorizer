import glob
import torch
from dataset import ColorizationDataset

from torchflare.experiments import Experiment, ModelConfig
import torchflare.callbacks as cbs
import segmentation_models_pytorch as smp

path = "./Dataset"
paths = glob.glob(path + "/*.jpg")

# Taking first 2500 images for pretraining
train_paths = paths[:2300]
val_paths = paths[2300:2500]

train_ds = ColorizationDataset(train_paths, img_size=256, split="train")
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16)

val_ds = ColorizationDataset(val_paths, img_size=256, split="valid")
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16)

config = ModelConfig(
    nn_module=smp.Unet,
    module_params={
        "encoder_name": "efficientnet-b1",
        "encoder_weights": "imagenet",
        "in_channels": 1,
        "classes": 2,
    },
    optimizer="Adam",
    optimizer_params={"lr": 3e-4},
    criterion="l1_loss",
)
callbacks = [
    cbs.ModelCheckpoint(
        mode="min", monitor="val_loss", file_name="unet.bin", save_dir="./models"
    )
]
exp = Experiment(num_epochs=5, fp16=True, seed=42, device="cuda")
exp.compile_experiment(config, callbacks)
exp.fit_loader(train_dl, val_dl)
