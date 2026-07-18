# VAE vs DDPM

Variational Autoencoder (VAE) and (later) DDPM implemented from scratch and
trained on CIFAR-10.

## Setup

```bash
pip install -r requirements.txt
```

CIFAR-10 is downloaded automatically to `./data` on first run.

## Training

Train the VAE with default hyperparameters:

```bash
python -m src.vae.train
```

During training the script logs the total loss, reconstruction term, and KL
term per epoch, periodically saves generated sample grids, and writes the final
model checkpoint to `outputs/vae/vae.pt`.

### Training options

| Flag | Default | Description |
| --- | --- | --- |
| `--data-root` | `./data` | Where CIFAR-10 is stored/downloaded. |
| `--output-dir` | `./outputs/vae` | Checkpoints and sample grids. |
| `--epochs` | `100` | Number of training epochs. |
| `--batch-size` | `128` | Mini-batch size. |
| `--lr` | `1e-3` | Adam learning rate. |
| `--latent-dim` | `128` | Dimensionality of the latent space. |
| `--num-workers` | `4` | DataLoader worker processes. |
| `--sample-every` | `10` | Save a sample grid every N epochs. |
| `--seed` | `0` | Random seed. |

Example with custom settings:

```bash
python -m src.vae.train --epochs 200 --batch-size 256 --latent-dim 256
```

Outputs:
- `outputs/vae/vae.pt` — final model weights.
- `outputs/vae/samples_epoch*.png` — sample grids saved during training.

## Inference / sampling

Generate images from a trained checkpoint:

```bash
python -m src.vae.sample --checkpoint outputs/vae/vae.pt
```

This produces:
- `outputs/vae/sample_grid.png` — an 8x8 preview grid of random samples.
- `outputs/vae/samples/*.png` — individual PNGs (default 10,000) for FID / IS
  evaluation.

### Inference options

| Flag | Default | Description |
| --- | --- | --- |
| `--checkpoint` | `./outputs/vae/vae.pt` | Trained model weights to load. |
| `--output-dir` | `./outputs/vae` | Where the preview grid is written. |
| `--samples-dir` | `./outputs/vae/samples` | Where individual PNGs are written. |
| `--num-samples` | `10000` | Number of images to export. |
| `--batch-size` | `128` | Sampling batch size. |
| `--latent-dim` | `128` | Must match the value used during training. |

> Note: `--latent-dim` at inference must match the value used at training time,
> otherwise the checkpoint will fail to load.

Example generating a smaller set:

```bash
python -m src.vae.sample --checkpoint outputs/vae/vae.pt --num-samples 1000
```
