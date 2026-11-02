"""Directly trained spiking policy for Sample Factory (ViZDoom doom_basic).

Same layer sizes as the ANN baseline (convnet_simple): 3 convs + a 512 layer, but every
hidden layer is an leaky integrate-and-fire (LIF) layer unrolled over T time steps and trained
with surrogate gradients. State is reset for every frame (no recurrence across frames).
"""
import numpy as np
import torch
from torch import nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder


# ----------------------------------------------------------------------------------------
# Neuron: Heaviside spike in the forward pass, triangular surrogate in the backward pass
# ----------------------------------------------------------------------------------------
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):  # x = membrane - threshold
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * torch.clamp(1.0 - x.abs(), min=0.0)  # triangular, width 1


def lif_step(v, current, threshold=1.0, v_init=0.5, detach_reset=True, beta=0.9):
    """One LIF step: decay voltage, add current, spike, subtract threshold.

    beta=1 recovers the IF update; current is not multiplied by (1-beta)."""
    if v is None:
        v = torch.full_like(current, v_init * threshold)
    v = beta * v + current
    s = SpikeFn.apply(v - threshold)
    v = v - (s.detach() if detach_reset else s) * threshold
    return v, s


# ----------------------------------------------------------------------------------------
# The spiking trunk (no Sample Factory dependency in the maths, so it can be evaluated alone)
# ----------------------------------------------------------------------------------------
class SpikingTrunk(nn.Module):
    def __init__(self, in_channels=3, obs_hw=(72, 128), T=8, v_init=0.5,
                 gains=(1.0, 1.0, 1.0, 1.0), feat_scale=4.0, beta=0.9):
        super().__init__()
        if not 0.0 <= float(beta) <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        self.register_buffer("beta", torch.tensor(float(beta)))
        self.T = int(T)
        self.v_init = float(v_init)
        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2)
        # local (unregistered) stack, only used to compute the flattened size; keeps state_dict keys unique
        self.flat_size = calc_num_elements(nn.Sequential(self.conv1, self.conv2, self.conv3), (in_channels, *obs_hw))
        self.fc = nn.Linear(self.flat_size, 512)
        # fixed (non-trainable) per-layer current gains and output scale: buffers, so they are saved
        # in the checkpoint and are not touched by Sample Factory's weight initialisation
        self.register_buffer("gains", torch.tensor(list(gains), dtype=torch.float32))
        self.register_buffer("feat_scale", torch.tensor(float(feat_scale)))
        self.out_size = 512

    def forward(self, x, T=None, return_spikes=False):
        T = self.T if T is None else int(T)
        g = self.gains
        c1 = self.conv1(x) * g[0]  # direct encoding: identical every step, computed once
        v1 = v2 = v3 = v4 = None
        acc = 0
        spikes = [0.0, 0.0, 0.0, 0.0]
        for _ in range(T):
            v1, s1 = lif_step(v1, c1, v_init=self.v_init, beta=self.beta)
            v2, s2 = lif_step(v2, self.conv2(s1) * g[1], v_init=self.v_init, beta=self.beta)
            v3, s3 = lif_step(v3, self.conv3(s2) * g[2], v_init=self.v_init, beta=self.beta)
            v4, s4 = lif_step(v4, self.fc(s3.flatten(1)) * g[3], v_init=self.v_init, beta=self.beta)
            acc = acc + s4
            if return_spikes:
                for k, s in enumerate((s1, s2, s3, s4)):
                    spikes[k] = spikes[k] + s.detach().flatten(1).sum(1)  # per-sample spike count
        feats = acc * (self.feat_scale / T)  # mean firing rate of the 512 layer, scaled
        if return_spikes:
            return feats, torch.stack(spikes, 1)  # (B, 4) spikes per decision per layer
        return feats


# ----------------------------------------------------------------------------------------
# Sample Factory glue
# ----------------------------------------------------------------------------------------
def add_snn_args(parser):
    p = parser
    p.add_argument("--snn_beta", default=0.9, type=float, help="Fixed LIF membrane retention in [0, 1]")
    p.add_argument("--snn_T", default=8, type=int, help="Time steps per frame for the spiking encoder")
    p.add_argument("--snn_v_init", default=0.5, type=float, help="Initial membrane potential / threshold")
    p.add_argument("--snn_gains", default="1,1,1,1", type=str,
                   help="Comma separated fixed current gains for the 4 spiking layers (from calibration)")
    p.add_argument("--snn_feat_scale", default=4.0, type=float, help="Scale on the mean firing rate fed to the heads")


class SpikingEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        obs_shape = obs_space["obs"].shape
        gains = [float(v) for v in str(cfg.snn_gains).split(",")]
        assert len(gains) == 4, "--snn_gains needs 4 comma separated numbers"
        self.trunk = SpikingTrunk(obs_shape[0], tuple(obs_shape[1:]), cfg.snn_T, cfg.snn_v_init,
                                  gains, cfg.snn_feat_scale, beta=cfg.snn_beta)
        self.encoder_out_size = self.trunk.out_size

    def forward(self, obs_dict):
        return self.trunk(obs_dict["obs"])

    def get_out_size(self):
        return self.encoder_out_size


def make_snn_encoder(cfg, obs_space):
    return SpikingEncoder(cfg, obs_space)


# ----------------------------------------------------------------------------------------
# Data-dependent gain calibration (so the network does not start silent)
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def calibrate_gains(frames, T=8, target_rate=0.15, v_init=0.5, iters=18, device="cpu", seed=0, beta=0.9):
    """Pick fixed per-layer current gains so each layer's mean firing rate is ~target_rate at
    initialisation, using real (already standardised) frames. Weights use the same init Sample
    Factory applies (orthogonal, gain 1, zero bias)."""
    torch.manual_seed(seed)
    trunk = SpikingTrunk(frames.shape[1], tuple(frames.shape[2:]), T, v_init, beta=beta).to(device)
    for m in (trunk.conv1, trunk.conv2, trunk.conv3, trunk.fc):
        nn.init.orthogonal_(m.weight, gain=1.0)
        nn.init.zeros_(m.bias)
    x = frames.to(device)
    gains = [1.0, 1.0, 1.0, 1.0]

    def rate_of_layer(k, g):
        gg = list(gains)
        gg[k] = g
        c1 = trunk.conv1(x) * gg[0]
        v = [None] * 4
        total, count = 0.0, 0
        for _ in range(T):
            v[0], s1 = lif_step(v[0], c1, v_init=v_init, beta=beta)
            if k == 0:
                total += s1.sum().item(); count += s1.numel(); continue
            v[1], s2 = lif_step(v[1], trunk.conv2(s1) * gg[1], v_init=v_init, beta=beta)
            if k == 1:
                total += s2.sum().item(); count += s2.numel(); continue
            v[2], s3 = lif_step(v[2], trunk.conv3(s2) * gg[2], v_init=v_init, beta=beta)
            if k == 2:
                total += s3.sum().item(); count += s3.numel(); continue
            v[3], s4 = lif_step(v[3], trunk.fc(s3.flatten(1)) * gg[3], v_init=v_init, beta=beta)
            total += s4.sum().item(); count += s4.numel()
        return total / count

    for k in range(4):
        lo, hi = -4.0, 4.0  # search log10(gain)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if rate_of_layer(k, 10 ** mid) < target_rate:
                lo = mid
            else:
                hi = mid
        gains[k] = float(10 ** (0.5 * (lo + hi)))
    rates = [rate_of_layer(k, gains[k]) for k in range(4)]
    return gains, rates


# ----------------------------------------------------------------------------------------
# Standalone evaluator (no Sample Factory model), mirrors the PlainCNN / converted-SNN protocol
# ----------------------------------------------------------------------------------------
class DirectSNN(nn.Module):
    def __init__(self, T=8, v_init=0.5, gains=(1, 1, 1, 1), feat_scale=4.0, n_actions=4, beta=0.9):
        super().__init__()
        self.trunk = SpikingTrunk(3, (72, 128), T, v_init, gains, feat_scale, beta=beta)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x, T=None, return_spikes=False):
        out = self.trunk(x, T=T, return_spikes=return_spikes)
        if return_spikes:
            feats, spk = out
            return self.head(feats), spk
        return self.head(out)


def load_direct_snn(sf_state_dict, T, v_init=0.5, n_actions=4):
    """Build the standalone SNN from a Sample Factory checkpoint['model'] state dict."""
    gains = sf_state_dict["encoder.trunk.gains"].cpu().tolist()
    feat_scale = float(sf_state_dict["encoder.trunk.feat_scale"])
    beta = float(sf_state_dict["encoder.trunk.beta"].cpu())
    net = DirectSNN(T, v_init, gains, feat_scale, n_actions, beta=beta)
    state = {}
    for k, v in sf_state_dict.items():
        if k.startswith("encoder.trunk."):
            state["trunk." + k[len("encoder.trunk."):]] = v
    state["head.weight"] = sf_state_dict["action_parameterization.distribution_linear.weight"]
    state["head.bias"] = sf_state_dict["action_parameterization.distribution_linear.bias"]
    missing, unexpected = net.load_state_dict(state, strict=False)
    return net.eval(), missing, unexpected
