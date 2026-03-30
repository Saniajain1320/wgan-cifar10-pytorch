# WGAN on CIFAR-10

Implementation of **Wasserstein GAN** (Arjovsky et al., 2017) trained on CIFAR-10,
with a Streamlit frontend for image generation and training visualization.

## Files
```
wgan_cifar10.py    # WGAN model + training loop
app.py             # Streamlit frontend
requirements.txt   # Dependencies
```

## Setup
```bash
pip install -r requirements.txt
```

## Train the Model
```bash
python wgan_cifar10.py
```
- CIFAR-10 is downloaded automatically to `./data/`
- Checkpoints saved to `./checkpoints/latest.pt`
- Sample images saved to `./samples/epoch_XXXX.png` every 10 epochs

## Run the Streamlit App
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

## Key WGAN Hyperparameters (from paper)
| Parameter | Value | Meaning |
|---|---|---|
| `n_critic` | 5 | Critic updates per generator update |
| `clip_value` | 0.01 | Weight clipping range [-c, c] |
| `lr` | 5e-5 | RMSprop learning rate |
| `batch_size` | 64 | Mini-batch size |
| `latent_dim` | 128 | Size of noise vector z |

## Architecture
**Generator** (DCGAN-style)
- FC → reshape to 256×8×8
- Upsample + Conv → 128×16×16
- Upsample + Conv → 64×32×32
- Conv + Tanh → 3×32×32

**Critic** (no sigmoid)
- 4 strided Conv2d blocks with BatchNorm + LeakyReLU
- FC → scalar score

## References
1. Arjovsky, Chintala & Bottou — "Wasserstein GAN" (arXiv:1701.07875)
2. Goodfellow et al. — "Improved Techniques for Training GANs" (NeurIPS 2016)
