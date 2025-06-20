import numpy as np
import torch
import random

# single-channel data is organized as (B, H, W), while 2-channel has(B, C, H, W)
def frame_shift(mels, labels, net_pooling=4):
    bsz, n_bands, frames = mels.shape
    shifted = []
    new_labels = []
    for bindx in range(bsz):
        shift = int(random.gauss(0, 90))
        shifted.append(torch.roll(mels[bindx], shift, dims=-1))
        shift = -abs(shift) // net_pooling if shift < 0 else shift // net_pooling
        new_labels.append(torch.roll(labels[bindx], shift, dims=-1))
    return torch.stack(shifted), torch.stack(new_labels)


def mixup(data, target=None, alpha=0.2, beta=0.2, mixup_label_type="soft"):
    """Mixup data augmentation by permuting the data

    Args:
        data: input tensor, must be a batch so data can be permuted and mixed.
        target: tensor of the target to be mixed, if None, do not return targets.
        alpha: float, the parameter to the np.random.beta distribution
        beta: float, the parameter to the np.random.beta distribution
        mixup_label_type: str, the type of mixup to be used choice between {'soft', 'hard'}.
    Returns:
        torch.Tensor of mixed data and labels if given
    """
    with torch.no_grad():
        batch_size = data.size(0)
        c = np.random.beta(alpha, beta)

        perm = torch.randperm(batch_size)

        mixed_data = c * data + (1 - c) * data[perm, :]
        if target is not None:
            if mixup_label_type == "soft":
                mixed_target = torch.clamp(
                    c * target + (1 - c) * target[perm, :], min=0, max=1
                )
            elif mixup_label_type == "hard":
                mixed_target = torch.clamp(target + target[perm, :], min=0, max=1)
            else:
                raise NotImplementedError(
                    f"mixup_label_type: {mixup_label_type} not implemented. choice in "
                    f"{'soft', 'hard'}"
                )

            return mixed_data, mixed_target
        else:
            return mixed_data


def add_noise(mels, snrs=(10, 30), dims=(1, 2)):
    """Add white noise to mels spectrograms
    Args:
        mels: torch.tensor, mels spectrograms to apply the white noise to.
        snrs: int or tuple, the range of snrs to choose from if tuple (uniform)
        dims: tuple, the dimensions for which to compute the standard deviation (default to (1,2) because assume
            an input of a batch of mel spectrograms.
    Returns:
        torch.Tensor of mels with noise applied
    """
    if isinstance(snrs, (list, tuple)):
        snr = (snrs[0] - snrs[1]) * torch.rand(
            (mels.shape[0],), device=mels.device
        ).reshape(-1, 1, 1,1) + snrs[1]
    else:
        snr = snrs

    snr = 10 ** (snr / 20)  # linear domain
    sigma = torch.std(mels, dim=dims, keepdim=True) / snr
    mels = mels + torch.randn(mels.shape, device=mels.device) * sigma

    return mels

# Apply filteraugment to balance data
def filt_aug(features,
             db_range=[-0.5, 0.5],
             n_band=[3, 6],
             min_bw=6,
             filter_type="step",
             log=True):
    # this is updated FilterAugment algorithm used for ICASSP 2022
    batch_size, n_channels, n_freq_bin, _ = features.shape
    n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1, )).item()  # [low, high)
    if n_freq_band > 1:
        while n_freq_bin - n_freq_band * min_bw + 1 < 0:
            min_bw -= 1
        band_bndry_freqs = torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1,
                                                    (n_freq_band - 1,)))[0] + \
                           torch.arange(1, n_freq_band) * min_bw
        band_bndry_freqs = torch.cat(
            (torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))

        if filter_type == "step":
            band_factors = torch.rand(
                (batch_size, n_freq_band)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            band_factors = 10**(band_factors / 20)

            freq_filt = torch.ones((batch_size, 1, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                freq_filt[:,:, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] =\
                    band_factors[:,i].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        else:
            raise Exception("Unkonwn filter augment type")

        if log:
            ret = features + torch.log10(freq_filt + 0.00001)
        else:
            ret = features * freq_filt
        return ret

    else:
        return features
