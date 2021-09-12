import os
import glob
import torch
import numpy as np
from model import PatchDiscriminator
from dataset import ColorizationDataset
from torchflare.experiments import ModelConfig, Experiment
import torchflare.callbacks as cbs
import segmentation_models_pytorch as smp
from skimage.color import lab2rgb
from pix2pix import Pix2PixExperiment
from torchvision.utils import save_image
import warnings

warnings.filterwarnings("ignore")

# Defining paths and dataloaders
path = "./Dataset"
paths = glob.glob(path + "/*.jpg")

rand_idx = np.random.permutation(paths)
sample_paths = rand_idx[9500:]
train_paths = rand_idx[2000:7000]
save_every = 5
save_path = "./saved_images"

train_ds = ColorizationDataset(train_paths, img_size=256, split="train")
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)

sample_ds = ColorizationDataset(sample_paths, img_size=256, split="valid")
sample_dl = torch.utils.data.DataLoader(sample_ds, batch_size=8, shuffle=True)


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.0) * 50.0
    ab = ab * 110.0
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    images = np.stack(rgb_imgs, axis=0)
    return torch.from_numpy(images).permute(0, 3, 1, 2)


# Overwrite load_checkpoint callback
@cbs.on_experiment_start(order=cbs.CallbackOrder.MODEL_INIT)
def unet_load_checkpoint(experiment: "Experiment"):
    ckpt = torch.load("unet.bin", experiment.device)
    experiment.state.model["generator"].load_state_dict(ckpt["model_state_dict"])


@cbs.on_epoch_end(order=cbs.CallbackOrder.EXTERNAL)
def save_generated_images(experiment: "Experiment"):
    if experiment.current_epoch % save_every == 0:
        sample_images = next(iter(sample_dl))
        inputs = sample_images[0].to(experiment.device)

        with torch.no_grad():
            gen_images = experiment.state.model["generator"](inputs)
        gen_images = gen_images.detach().cpu()
        gen_images = lab_to_rgb(sample_images[0], gen_images)

        real_images = lab_to_rgb(sample_images[0], sample_images[1])
        save_image(
            gen_images,
            os.path.join(save_path, f"fake_images_{experiment.current_epoch}.jpg"),
            nrow=2,
        )
        save_image(
            real_images,
            os.path.join(save_path, f"real_images_{experiment.current_epoch}.jpg"),
            nrow=2,
        )


callbacks = [
    cbs.ModelCheckpoint(
        mode="min", monitor="train_G_loss", file_name="model.bin", save_dir="./models"
    ),
    unet_load_checkpoint,
    save_generated_images,
]
config = ModelConfig(
    nn_module={"generator": smp.Unet, "discriminator": PatchDiscriminator},
    module_params={
        "generator": {
            "encoder_name": "efficientnet-b1",
            "encoder_weights": "imagenet",
            "in_channels": 1,
            "classes": 2,
        },
        "discriminator": {"input_channels": 3, "n_down": 3, "num_filters": 64},
    },
    criterion={"BCE": "binary_cross_entropy_with_logits", "L1_LOSS": "l1_loss"},
    optimizer={"generator": "Adam", "discriminator": "Adam"},
    optimizer_params={"generator": {"lr": 2e-4}, "discriminator": {"lr": 2e-4}},
)

trainer = Pix2PixExperiment(lambda_l1=100, num_epochs=10, fp16=True, seed=42, device="cuda")
trainer.compile_experiment(model_config=config, callbacks=callbacks)
trainer.fit_loader(train_dl)
