#!/usr/bin/env python3
"""Export heightmap student model sub-components as TorchScript JIT modules.

Usage:
    python export_heightmap_jit.py --checkpoint /path/to/model_XXXX.pt --output_dir ./policy

Exports (Approach B - separate sub-components):
    1. heightmap_encoder.jit  - HeightmapEncoder (backbone + GRU + output_mlp)
       Input:  heightmap(1,132), proprio(1,49), hidden(1,1,512)
       Output: tuple(latent_yaw(1,34), new_hidden(1,1,512))
    2. history_encoder.jit    - StateHistoryEncoder
       Input:  obs_history(1,10,53)
       Output: latent(1,20)
    3. actor_backbone.jit     - Actor backbone MLP
       Input:  cat(prop,scan_latent,priv_explicit,hist_latent) = (1,114)
       Output: actions(1,12)
"""

import argparse
import os
import sys
import torch
import torch.nn as nn

# Add project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "rsl_rl"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "legged_gym"))

from rsl_rl.modules import HeightmapMLPBackbone, HeightmapEncoder
from rsl_rl.modules.actor_critic import StateHistoryEncoder, Actor


# ============================================================
# Wrapper: HeightmapEncoder with explicit hidden state I/O
# ============================================================
class HeightmapEncoderExport(nn.Module):
    """Wraps HeightmapEncoder to accept/return GRU hidden state explicitly."""

    def __init__(self, encoder: HeightmapEncoder):
        super().__init__()
        self.backbone = encoder.backbone
        self.combination_mlp = encoder.combination_mlp
        self.rnn = encoder.rnn
        self.output_mlp = encoder.output_mlp

    def forward(self, heightmap: torch.Tensor, proprioception: torch.Tensor,
                hidden: torch.Tensor):
        """
        Args:
            heightmap: (batch, n_points)
            proprioception: (batch, n_proprio_student)
            hidden: (1, batch, hidden_size) GRU hidden state
        Returns:
            tuple(output, new_hidden):
                output: (batch, output_dim + 2)  terrain latent + yaw
                new_hidden: (1, batch, hidden_size)
        """
        terrain_feature = self.backbone(heightmap)
        combined = self.combination_mlp(
            torch.cat((terrain_feature, proprioception), dim=-1)
        )
        combined, new_hidden = self.rnn(combined[:, None, :], hidden)
        output = self.output_mlp(combined.squeeze(1))
        return output, new_hidden


# ============================================================
# Wrapper: HistoryEncoder
# ============================================================
class HistoryEncoderExport(nn.Module):
    """Wraps StateHistoryEncoder for clean JIT tracing."""

    def __init__(self, history_encoder: StateHistoryEncoder):
        super().__init__()
        self.encoder = history_encoder

    def forward(self, obs_history: torch.Tensor):
        """
        Args:
            obs_history: (batch, T, n_proprio) = (1, 10, 53)
        Returns:
            latent: (batch, output_dim) = (1, 20)
        """
        return self.encoder(obs_history)


def main():
    parser = argparse.ArgumentParser(description="Export heightmap student JIT models")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--output_dir", default="./policy", help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device for tracing")

    # Model architecture args (defaults match mybot_v3 config)
    parser.add_argument("--n_points", type=int, default=132)
    parser.add_argument("--n_proprio_student", type=int, default=49)
    parser.add_argument("--n_proprio", type=int, default=53)
    parser.add_argument("--n_scan", type=int, default=132)
    parser.add_argument("--n_priv_explicit", type=int, default=9)
    parser.add_argument("--n_priv_latent", type=int, default=29)
    parser.add_argument("--num_actions", type=int, default=12)
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--backbone_hidden_dims", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--backbone_output_dim", type=int, default=32)
    parser.add_argument("--gru_hidden_size", type=int, default=512)
    parser.add_argument("--encoder_output_dim", type=int, default=32)
    parser.add_argument("--scan_encoder_dims", nargs="+", type=int, default=[128, 64, 32])
    parser.add_argument("--priv_encoder_dims", nargs="+", type=int, default=[64, 20])
    parser.add_argument("--actor_hidden_dims", nargs="+", type=int, default=[512, 256, 128])

    args = parser.parse_args()
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # ============================================================
    # 1. Export HeightmapEncoder
    # ============================================================
    print("\n--- Exporting heightmap_encoder ---")
    backbone = HeightmapMLPBackbone(
        n_points=args.n_points,
        output_dim=args.backbone_output_dim,
        hidden_dims=args.backbone_hidden_dims,
    )
    encoder = HeightmapEncoder(
        backbone=backbone,
        n_proprio_student=args.n_proprio_student,
        hidden_size=args.gru_hidden_size,
        output_dim=args.encoder_output_dim,
    )

    if "heightmap_encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["heightmap_encoder_state_dict"])
        print("  Loaded heightmap_encoder_state_dict")
    else:
        print("  WARNING: heightmap_encoder_state_dict not found!")

    encoder.eval().to(device)
    export_encoder = HeightmapEncoderExport(encoder).eval().to(device)

    # Trace
    test_hm = torch.zeros(1, args.n_points, device=device)
    test_prop = torch.zeros(1, args.n_proprio_student, device=device)
    test_hidden = torch.zeros(1, 1, args.gru_hidden_size, device=device)

    traced_encoder = torch.jit.trace(export_encoder, (test_hm, test_prop, test_hidden))
    encoder_path = os.path.join(args.output_dir, "heightmap_encoder.jit")
    traced_encoder.save(encoder_path)
    print(f"  Saved: {encoder_path}")

    # Verify
    out, h = traced_encoder(test_hm, test_prop, test_hidden)
    print(f"  Output shape: {out.shape} (expected [1, {args.encoder_output_dim + 2}])")
    print(f"  Hidden shape: {h.shape} (expected [1, 1, {args.gru_hidden_size}])")

    # ============================================================
    # 2. Export HistoryEncoder
    # ============================================================
    print("\n--- Exporting history_encoder ---")
    activation = nn.ELU()
    history_enc = StateHistoryEncoder(
        activation_fn=activation,
        input_size=args.n_proprio,
        tsteps=args.history_len,
        output_size=args.priv_encoder_dims[-1],
    )

    # Load from heightmap_actor state dict (which is a copy of Actor)
    if "heightmap_actor_state_dict" in checkpoint:
        actor_sd = checkpoint["heightmap_actor_state_dict"]
        hist_sd = {}
        for k, v in actor_sd.items():
            if k.startswith("history_encoder."):
                hist_sd[k.replace("history_encoder.", "")] = v
        if hist_sd:
            history_enc.load_state_dict(hist_sd)
            print("  Loaded history_encoder from heightmap_actor_state_dict")
        else:
            print("  WARNING: No history_encoder keys found in actor state dict")
    else:
        print("  WARNING: heightmap_actor_state_dict not found!")

    history_enc.eval().to(device)
    export_hist = HistoryEncoderExport(history_enc).eval().to(device)

    test_hist = torch.zeros(1, args.history_len, args.n_proprio, device=device)
    traced_hist = torch.jit.trace(export_hist, (test_hist,))
    hist_path = os.path.join(args.output_dir, "history_encoder.jit")
    traced_hist.save(hist_path)
    print(f"  Saved: {hist_path}")

    # Verify
    h_out = traced_hist(test_hist)
    print(f"  Output shape: {h_out.shape} (expected [1, {args.priv_encoder_dims[-1]}])")

    # ============================================================
    # 3. Export Actor Backbone
    # ============================================================
    print("\n--- Exporting actor_backbone ---")
    # Reconstruct the Actor to extract actor_backbone
    actor = Actor(
        num_prop=args.n_proprio,
        num_scan=args.n_scan,
        num_actions=args.num_actions,
        scan_encoder_dims=args.scan_encoder_dims,
        actor_hidden_dims=args.actor_hidden_dims,
        priv_encoder_dims=args.priv_encoder_dims,
        num_priv_latent=args.n_priv_latent,
        num_priv_explicit=args.n_priv_explicit,
        num_hist=args.history_len,
        activation=activation,
        tanh_encoder_output=False,
    )

    if "heightmap_actor_state_dict" in checkpoint:
        actor.load_state_dict(checkpoint["heightmap_actor_state_dict"])
        print("  Loaded actor from heightmap_actor_state_dict")

    actor.eval().to(device)
    actor_backbone = actor.actor_backbone

    # Actor backbone input = n_proprio + scan_encoder_output + n_priv_explicit + priv_encoder_output
    scan_enc_out = args.scan_encoder_dims[-1]  # 32
    priv_enc_out = args.priv_encoder_dims[-1]  # 20
    backbone_input_dim = args.n_proprio + scan_enc_out + args.n_priv_explicit + priv_enc_out
    print(f"  Backbone input dim: {backbone_input_dim}")

    test_bb = torch.zeros(1, backbone_input_dim, device=device)
    traced_bb = torch.jit.trace(actor_backbone, test_bb)
    bb_path = os.path.join(args.output_dir, "actor_backbone.jit")
    traced_bb.save(bb_path)
    print(f"  Saved: {bb_path}")

    # Verify
    bb_out = traced_bb(test_bb)
    print(f"  Output shape: {bb_out.shape} (expected [1, {args.num_actions}])")

    print(f"\n=== All models exported to {args.output_dir} ===")
    print(f"  heightmap_encoder.jit  ({os.path.getsize(encoder_path)} bytes)")
    print(f"  history_encoder.jit    ({os.path.getsize(hist_path)} bytes)")
    print(f"  actor_backbone.jit     ({os.path.getsize(bb_path)} bytes)")


if __name__ == "__main__":
    main()
