# VAE vs DDPM

Variational Autoencoder (VAE) and DDPM implemented from scratch and
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
| `--batch-size` | `512` | Mini-batch size. |
| `--lr` | `1e-3` | Adam learning rate. |
| `--latent-dim` | `256` | Dimensionality of the latent space. |
| `--grad-clip` | `1.0` | Max global grad norm (clipping). |
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
| `--latent-dim` | `256` | Must match the value used during training. |

> Note: `--latent-dim` at inference must match the value used at training time,
> otherwise the checkpoint will fail to load.

Example generating a smaller set:

```bash
python -m src.vae.sample --checkpoint outputs/vae/vae.pt --num-samples 1000
```

## Evaluation (FID & Inception Score)

The `src.eval` package is model-agnostic: it scores any directory of generated
images, so it works for both the VAE and the (future) DDPM. It reports **FID**
(against a reference set) and the **Inception Score**, both computed via
`torchmetrics`.

```bash
# FID against the CIFAR-10 train split + Inception Score.
python -m src.eval.evaluate --generated-dir outputs/vae/samples
```

By default the FID reference is the CIFAR-10 training split (the standard,
comparable choice for CIFAR-10). To compare against your own reference images
instead, pass a directory:

```bash
python -m src.eval.evaluate \
    --generated-dir outputs/ddpm/samples \
    --reference-dir data/real_images
```

### Evaluation options

| Flag | Default | Description |
| --- | --- | --- |
| `--generated-dir` | *(required)* | Directory of generated images to score. |
| `--reference-dir` | `None` | Reference images for FID. Falls back to CIFAR-10 train if omitted. |
| `--data-root` | `./data` | Where CIFAR-10 is stored/downloaded. |
| `--batch-size` | `64` | Inception batch size. |
| `--num-workers` | `4` | DataLoader worker processes. |
| `--is-splits` | `10` | Number of splits for the Inception Score std. |

> For results comparable to the literature, generate ~50k samples
> (`--num-samples 50000`) and evaluate against the CIFAR-10 train split.

## DDPM

A Denoising Diffusion Probabilistic Model (DDPM) with a small time-conditioned
U-Net and a linear noise schedule, trained on the same CIFAR-10 pipeline as the
VAE.

### Training

```bash
python -m src.ddpm.train
```

The model predicts the noise added at a random timestep and is trained with a
simple MSE loss. Sample grids are saved periodically and the final checkpoint is
written to `outputs/ddpm/ddpm.pt`.

| Flag | Default | Description |
| --- | --- | --- |
| `--data-root` | `./data` | Where CIFAR-10 is stored/downloaded. |
| `--output-dir` | `./outputs/ddpm` | Checkpoints and sample grids. |
| `--epochs` | `100` | Number of training epochs. |
| `--batch-size` | `128` | Mini-batch size. |
| `--lr` | `2e-4` | Adam learning rate. |
| `--timesteps` | `1000` | Number of diffusion steps. |
| `--num-workers` | `4` | DataLoader worker processes. |
| `--sample-every` | `10` | Save a sample grid every N epochs. |
| `--grad-clip` | `1.0` | Max global grad norm (clipping). |
| `--seed` | `0` | Random seed. |

### Sampling

```bash
python -m src.ddpm.sample --checkpoint outputs/ddpm/ddpm.pt
```

This writes an 8x8 preview grid and, by default, 10,000 individual PNGs to
`outputs/ddpm/samples/` for FID / IS evaluation via `src.eval.evaluate`.

| Flag | Default | Description |
| --- | --- | --- |
| `--checkpoint` | `./outputs/ddpm/ddpm.pt` | Trained model weights to load. |
| `--output-dir` | `./outputs/ddpm` | Where the preview grid is written. |
| `--samples-dir` | `./outputs/ddpm/samples` | Where individual PNGs are written. |
| `--num-samples` | `10000` | Number of images to export. |
| `--batch-size` | `128` | Sampling batch size. |
| `--timesteps` | `1000` | Must match the value used during training. |

### Compute requirements

The U-Net is small (~3.5M parameters), so **memory** is not the bottleneck —
training fits comfortably in a few GB of GPU RAM at batch size 128, well within
a typical GPU. The cost is **time**, driven by two things:

- **Training:** each step runs one U-Net forward/backward. On a single modern
  GPU expect on the order of minutes per epoch; ~100 epochs is a few hours.
  A GPU is strongly recommended — on CPU this is impractically slow.
- **Sampling is the expensive part.** DDPM sampling is *iterative*: generating
  one batch requires `--timesteps` (default **1000**) sequential U-Net passes.
  Exporting 10k images for FID therefore takes far longer than the VAE's
  single forward pass — plan for a meaningful chunk of GPU time, and reduce
  `--num-samples` for quick checks.

> Tip: for faster iteration during development, train/sample with fewer
> timesteps (e.g. `--timesteps 200`) and a smaller `--num-samples`; use the full
> settings only for final, comparable results.

