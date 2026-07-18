"""Inception-v3 wrapper shared by the FID and Inception Score metrics.

The extractor is model-agnostic: it consumes batches of images in ``[0, 1]``
range with shape ``(N, 3, H, W)`` and returns both the 2048-dim pooled
features (used by FID) and the 1000-way softmax class probabilities (used by
the Inception Score). This makes it reusable for any generator, e.g. the VAE
and the (future) DDPM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3

# ImageNet normalization statistics expected by torchvision's Inception-v3.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class InceptionFeatureExtractor(nn.Module):
    """Extract pooled features and class probabilities from Inception-v3."""

    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        weights = Inception_V3_Weights.IMAGENET1K_V1
        self.model = inception_v3(weights=weights, aux_logits=True)
        self.model.fc = nn.Identity()  # expose 2048-d pooled features
        self.model.eval().to(self.device)

        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        # Original classifier weights, kept to turn features into logits.
        self._fc = nn.Linear(2048, 1000)
        self._fc.weight.data = weights.get_state_dict(
            progress=False)["fc.weight"].clone()
        self._fc.bias.data = weights.get_state_dict(
            progress=False)["fc.bias"].clone()
        self._fc.eval().to(self.device)

    def _preprocess(self, images):
        """Resize to 299x299 and apply ImageNet normalization."""
        if images.shape[-2:] != (299, 299):
            images = F.interpolate(
                images, size=(299, 299), mode="bilinear",
                align_corners=False)
        return (images - self.mean) / self.std

    @torch.no_grad()
    def forward(self, images):
        """Return ``(features, probs)`` for a batch of images in ``[0, 1]``."""
        images = images.to(self.device)
        images = self._preprocess(images)
        features = self.model(images)
        probs = F.softmax(self._fc(features), dim=1)
        return features, probs
