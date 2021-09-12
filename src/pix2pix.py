import torch
from torchflare.experiments import Experiment


class Pix2PixExperiment(Experiment):
    """Custom class for training Pix2Pix model. TorchFlare makes it easy to write the tre"""

    def __init__(self, lambda_l1, **kwargs):
        self.lambda_l1 = lambda_l1
        super(Pix2PixExperiment, self).__init__(**kwargs)

    def train_step(self):
        """Traininig step for the model."""
        y_fake = self.state.model["generator"](self.batch[self.input_key])

        # Discriminator Training
        self.backend.zero_grad(self.state.optimizer["discriminator"])

        real_image = torch.cat([self.batch[self.input_key], self.batch[self.target_key]], dim=1)
        D_real = self.state.model["discriminator"](real_image)
        D_real_loss = self.state.criterion["BCE"](D_real, torch.ones_like(D_real))

        fake_image = torch.cat([self.batch[self.input_key], y_fake], dim=1)
        D_fake = self.state.model["discriminator"](fake_image.detach())
        D_fake_loss = self.state.criterion["BCE"](D_fake, torch.zeros_like(D_fake))

        D_loss = 0.5 * (D_fake_loss + D_real_loss)
        self.backend.backward_loss(loss=D_loss)
        self.backend.optimizer_step(self.state.optimizer["discriminator"])

        # Generator Training
        self.backend.zero_grad(self.state.optimizer["generator"])

        D_fake_preds = self.state.model["discriminator"](fake_image)
        G_fake_loss = self.state.criterion["BCE"](D_fake_preds, torch.ones_like(D_fake_preds))

        l1 = self.state.criterion["L1_LOSS"](y_fake, self.batch[self.target_key]) * self.lambda_l1

        G_loss = l1 + G_fake_loss
        self.backend.backward_loss(loss=G_loss)
        self.backend.optimizer_step(self.state.optimizer["generator"])

        return {"G_loss": G_loss.item(), "D_loss": D_loss.item()}
