import torch
from torchflare.experiments import Experiment


class Pix2PixExperiment(Experiment):
    def __init__(self, lambda_l1, **kwargs):
        self.lambda_l1 = lambda_l1
        super(Pix2PixExperiment, self).__init__(**kwargs)

    def train_step(self):
        y_fake = self.state.model["generator"](self.batch[self.input_key])

        # Discriminator Training
        self.state.optimizer["discriminator"].zero_grad()

        real_image = torch.cat(
            [self.batch[self.input_key], self.batch[self.target_key]], dim=1
        )
        D_real = self.state.model["discriminator"](real_image)
        D_real_loss = self.state.criterion["BCE"](D_real, torch.ones_like(D_real))

        fake_image = torch.cat([self.batch[self.input_key], y_fake], dim=1)
        D_fake = self.state.model["discriminator"](fake_image.detach())
        D_fake_loss = self.state.criterion["BCE"](D_fake, torch.zeros_like(D_fake))

        D_loss = 0.5 * (D_fake_loss + D_real_loss)
        D_loss.backward()
        self.state.optimizer["discriminator"].step()

        # Generator Training
        self.state.optimizer["generator"].zero_grad()

        D_fake_preds = self.state.model["discriminator"](fake_image)
        G_fake_loss = self.state.criterion["BCE"](
            D_fake_preds, torch.ones_like(D_fake_preds)
        )

        l1 = (
            self.state.criterion["L1_LOSS"](y_fake, self.batch[self.target_key])
            * self.lambda_l1
        )

        G_loss = l1 + G_fake_loss
        G_loss.backward()

        self.state.optimizer["generator"].step()

        self.loss_per_batch = {"G_loss": G_loss.item(), "D_loss": D_loss.item()}
