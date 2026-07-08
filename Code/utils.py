import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, data
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.patches as patches
from scipy.ndimage import median_filter
from PIL import Image
from pathlib import Path
import os

def make_gaussian_psf_freq(sigma, H, W):
    """
    Build a 2D Gaussian PSF in the frequency domain.
    Args:
        sigma: Gaussian standard deviation (in pixels)
        H, W:  spatial dimensions of the target image
    Returns:
        psf_fft: complex numpy array [H, W] — PSF in frequency domain
    """
    ax = np.arange(H) - H // 2
    ay = np.arange(W) - W // 2
    xx, yy = np.meshgrid(ay, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()
    psf_shifted = np.fft.ifftshift(psf)
    return np.fft.fft2(psf_shifted)


def apply_psf_freq(image_np, psf_fft):
    """
    Convolve a 2D image with a pre-computed frequency-domain PSF.
    Args:
        image_np: 2D numpy array [H, W], values in [0, 1]
        psf_fft:  complex numpy array [H, W] from make_gaussian_psf_freq
    Returns:
        blurred: 2D numpy array [H, W], clipped to [0, 1]
    """
    img_fft = np.fft.fft2(image_np)
    return np.fft.ifft2(img_fft * psf_fft).real.clip(0, 1)


class PositionalEncoding(nn.Module):
    """
    Flexible positional encoding:
    - 'nerf'     : sin/cos with 2^k frequencies (NeRF-style)
    - 'gaussian' : random Fourier features (Gaussian projection matrix B)
    """
    def __init__(self, in_features, num_frequencies=10, include_input=True,
                 encoding_type="nerf", gauss_scale=10.0):
        super().__init__()
        self.in_features   = in_features
        self.include_input = include_input
        self.encoding_type = encoding_type

        if encoding_type == "nerf":
            freq_bands = 2. ** torch.linspace(0., num_frequencies - 1, num_frequencies)
            self.register_buffer("freq_bands_buf", freq_bands)

        elif encoding_type == "gaussian":
            B = torch.randn(num_frequencies, in_features) * gauss_scale
            self.register_buffer("B", B)

        else:
            raise ValueError("encoding_type must be 'nerf' or 'gaussian'")

    def forward(self, x):
        out = [x] if self.include_input else []

        if self.encoding_type == "nerf":
            for freq in self.freq_bands_buf:
                out.append(torch.sin(freq * x))
                out.append(torch.cos(freq * x))

        elif self.encoding_type == "gaussian":
            x_proj = 2. * torch.pi * (x @ self.B.T)   # [N, num_frequencies]
            out.append(torch.sin(x_proj))
            out.append(torch.cos(x_proj))

        return torch.cat(out, dim=-1)


class PosEncMLP(nn.Module):
    """
    ReLU MLP with selectable positional encoding (NeRF or Gaussian Fourier).
    encoding_type : 'nerf' | 'gaussian'
    gauss_scale   : std of random Fourier projection (only used when encoding_type='gaussian')
    """
    def __init__(self, in_features=2, out_features=1, hidden_features=128,
                 hidden_layers=3, num_encoding_freqs=10, include_input=True,
                 encoding_type="nerf", gauss_scale=10.0):
        super().__init__()

        self.pos_enc = PositionalEncoding(
            in_features,
            num_frequencies=num_encoding_freqs,
            include_input=include_input,
            encoding_type=encoding_type,
            gauss_scale=gauss_scale
        )

        # Output dim of positional encoding
        base = in_features if include_input else 0
        if encoding_type == "nerf":
            pe_out_dim = base + in_features * 2 * num_encoding_freqs
        elif encoding_type == "gaussian":
            pe_out_dim = base + 2 * num_encoding_freqs

        layers = [nn.Linear(pe_out_dim, hidden_features), nn.ReLU(inplace=True)]
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_features, out_features))

        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, coords):
        return self.net(self.pos_enc(coords))


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0   = omega_0
        self.is_first  = is_first
        self.in_features = in_features
        self.linear    = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=False, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        for _ in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)

# ============================================================
# PSF IN FREQUENCY DOMAIN (TORCH — keeps gradients alive)
# ============================================================
def make_gaussian_psf_freq_torch(sigma, H, W, device):
    """
    Build a 2D Gaussian PSF in the frequency domain as a torch tensor.
    Computed once and reused every iteration — no gradients needed.
    Args:
        sigma:  Gaussian std dev (pixels)
        H, W:   spatial dims of the image
        device: torch device
    Returns:
        psf_fft: complex torch tensor [H, W]
    """
    ax = torch.arange(H, dtype=torch.float32, device=device) - H // 2
    ay = torch.arange(W, dtype=torch.float32, device=device) - W // 2
    yy, xx = torch.meshgrid(ax, ay, indexing='ij')
    psf = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()
    psf_shifted = torch.fft.ifftshift(psf)
    return torch.fft.fft2(psf_shifted)


def apply_psf_freq_torch(image_tensor, psf_fft):
    """
    Convolve a 2D image tensor with a pre-computed frequency-domain PSF.
    Differentiable — gradients flow through fft2/ifft2.
    Args:
        image_tensor: float torch tensor [H, W]
        psf_fft:      complex torch tensor [H, W] from make_gaussian_psf_freq_torch
    Returns:
        blurred: float torch tensor [H, W], clipped to [0, 1]
    """
    img_fft = torch.fft.fft2(image_tensor)
    return torch.fft.ifft2(img_fft * psf_fft).real.clamp(0, 1)

def tv_loss(img):
    """Isotropic Total Variation loss on a (1,1,H,W) tensor."""
    diff_h = img[:, :, 1:, :] - img[:, :, :-1, :]
    diff_w = img[:, :, :, 1:] - img[:, :, :, :-1]
    return (diff_h.abs().mean() + diff_w.abs().mean()) / 2