# src/models/my_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBackbone(nn.Module):
    """
    Same CNN backbone as BaselineCNN (KhaitCNN1D).
    Only extracts embedding h up to the final Dense(128) layer.

    Architecture:
        Conv1D(32, 9, padding='same', relu) x2
        MaxPool1D(4)
        Dropout(0.5)

        Conv1D(64, 9, padding='same', relu) x2
        MaxPool1D(4)
        Dropout(0.5)

        Conv1D(128, 9, padding='same', relu) x2
        MaxPool1D(4)
        Dropout(0.5)

        Flatten
        Dense(emb_dim, relu)
        Dropout(0.5)

    Input:
        x: [B, 1, L]
    Output:
        h: [B, emb_dim]
    """

    def __init__(self, emb_dim: int = 128, input_length: int = 1000):
        super().__init__()

        self.emb_dim = emb_dim

        kernel_size = 9
        padding = kernel_size // 2
        pool_size = 4
        drop_prob = 0.5

        # --- Conv blocks ---
        self.conv1_1 = nn.Conv1d(1, 32, kernel_size, padding=padding)
        self.conv1_2 = nn.Conv1d(32, 32, kernel_size, padding=padding)
        self.pool1 = nn.MaxPool1d(pool_size)
        self.drop1 = nn.Dropout(drop_prob)

        self.conv2_1 = nn.Conv1d(32, 64, kernel_size, padding=padding)
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size, padding=padding)
        self.pool2 = nn.MaxPool1d(pool_size)
        self.drop2 = nn.Dropout(drop_prob)

        self.conv3_1 = nn.Conv1d(64, 128, kernel_size, padding=padding)
        self.conv3_2 = nn.Conv1d(128, 128, kernel_size, padding=padding)
        self.pool3 = nn.MaxPool1d(pool_size)
        self.drop3 = nn.Dropout(drop_prob)

        # --- Flatten + Dense(emb_dim) + Dropout ---
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            feat = self._forward_conv(dummy)
            flat_dim = feat.shape[1] * feat.shape[2]

        self.fc = nn.Linear(flat_dim, emb_dim)
        self.drop_fc = nn.Dropout(drop_prob)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)
        x = self.drop3(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, L] -> h: [B, emb_dim]
        """
        x = self._forward_conv(x)
        x = x.flatten(1)
        h = F.relu(self.fc(x))
        h = self.drop_fc(h)
        return h


class MyModel(nn.Module):
    """
    Khait CNN backbone + VAE + SSL + (optional) DG domain head.

    - Main classifier:
        h -> logit (single logit, binary)
        Loss: BCEWithLogitsLoss

    - VAE:
        h -> (mu, logvar) -> z -> h_rec
        Loss: MSE(h, h_rec) + beta_kl * KL(q(z|h) || N(0, I))

    - SSL:
        For x_aug (input with small noise added),
        enforce similarity between z(x) and z(x_aug) via MSE(z, z_aug)

    - DG:
        h -> domain_logits (domain classifier head)
        Actual DANN backpropagation controlled by alpha in training code (currently only maintains structure)
    """

    def __init__(
        self,
        num_classes: int = 2,    # Current implementation only supports binary tasks
        num_domains: int = 2,
        emb_dim: int = 128,
        latent_dim: int = 32,
        input_length: int = 1000,
    ):
        super().__init__()
        assert num_classes == 2, "MyModel currently assumes binary tasks only."

        self.emb_dim = emb_dim
        self.latent_dim = latent_dim
        self.num_domains = num_domains

        # CNN backbone
        self.backbone = CNNBackbone(emb_dim=emb_dim, input_length=input_length)

        # Main binary classifier head: h -> logit
        self.fc_out = nn.Linear(emb_dim, 1)

        # VAE encoder / decoder
        self.fc_mu = nn.Linear(emb_dim, latent_dim)
        self.fc_logvar = nn.Linear(emb_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, emb_dim)

        # Domain classifier (DG)
        self.fc_domain = nn.Linear(emb_dim, num_domains)

    # --------- VAE utilities ---------
    def encode(self, h: torch.Tensor):
        """h [B, emb_dim] -> (mu, logvar) each [B, latent_dim]."""
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        """Reparameterization trick: z = mu + std * eps."""
        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar_clamped)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        """z [B, latent_dim] -> h_rec [B, emb_dim]."""
        return self.fc_dec(z)

    @staticmethod
    def vae_loss(
        h: torch.Tensor,
        h_rec: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta_kl: float = 1.0,
    ):
        """
        VAE loss on embedding h:

            L = MSE(h_rec, h) + beta_kl * KL(q(z|h) || N(0, I))

        KL(q||p) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        Here we use batch-wise mean.
        """
        mse = F.mse_loss(h_rec, h)

        logvar_clamped = torch.clamp(logvar, min=-10.0, max=10.0)
        kl = -0.5 * torch.mean(
            1 + logvar_clamped - mu.pow(2) - logvar.exp()
        )

        return mse + beta_kl * kl

    # --------- Forward ---------
    def forward(self, x: torch.Tensor, alpha: float = 0.0, return_dict: bool = False):
        """
        x: [B, 1, L]

        return_dict = False:
            -> logits [B]

        return_dict = True:
            -> dict(
                logits:         [B]
                domain_logits:  [B, num_domains]
                h:              [B, emb_dim]
                mu:             [B, latent_dim]
                logvar:         [B, latent_dim]
                z:              [B, latent_dim]
                h_rec:          [B, emb_dim]
            )

        The alpha parameter is reserved for controlling gradient reversal strength in DANN.
        Currently, GRL layer is not actually used; only the structure is maintained.
        """
        # 1) CNN backbone
        h = self.backbone(x)  # [B, emb_dim]

        # 2) VAE
        mu, logvar = self.encode(h)
        z = self.reparameterize(mu, logvar)
        h_rec = self.decode(z)

        # 3) Main binary classifier
        logits = self.fc_out(h).squeeze(-1)  # [B, 1] -> [B]

        # 4) Domain classifier
        domain_logits = self.fc_domain(h)  # [B, num_domains]

        if not return_dict:
            return logits

        return {
            "logits": logits,
            "domain_logits": domain_logits,
            "h": h,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "h_rec": h_rec,
        }
