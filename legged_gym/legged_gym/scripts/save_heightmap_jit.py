import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.append("../../../rsl_rl")

from rsl_rl.modules.actor_critic import Actor, get_activation
from rsl_rl.modules.heightmap_backbone import HeightmapMLPBackbone, HeightmapEncoder


def get_load_path(root, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        for name in os.listdir(model_parent):
            if os.path.isdir(os.path.join(model_parent, name)) and name[:6] == model_name_cand[:6]:
                root = os.path.join(model_parent, name)
                break
    if checkpoint == -1:
        models = [f for f in os.listdir(root) if model_name_include in f]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = f"model_{checkpoint}.pt"
    return os.path.join(root, model), checkpoint


class HeightmapActorWrapper(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, obs, terrain_latent):
        return self.actor(obs, hist_encoding=True, eval=False, scandots_latent=terrain_latent)


def build_actor(args):
    activation = get_activation(args.activation)
    return Actor(
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
        tanh_encoder_output=args.tanh,
    )


def build_encoder(args):
    backbone = HeightmapMLPBackbone(
        n_points=args.n_heightmap,
        output_dim=args.heightmap_backbone_output_dim,
        hidden_dims=args.heightmap_backbone_hidden_dims,
    )
    return HeightmapEncoder(
        backbone=backbone,
        n_proprio_student=args.n_proprio_student,
        hidden_size=args.heightmap_hidden_size,
        output_dim=args.heightmap_output_dim,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exptid", type=str, required=True)
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--proj_name", type=str, default="rough_mybot_v3")
    parser.add_argument("--log_root", type=str, default="../../logs")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tanh", action="store_true")

    parser.add_argument("--n_proprio", type=int, default=53)
    parser.add_argument("--n_proprio_student", type=int, default=49)
    parser.add_argument("--n_scan", type=int, default=132)
    parser.add_argument("--n_priv_explicit", type=int, default=9)
    parser.add_argument("--n_priv_latent", type=int, default=29)
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--num_actions", type=int, default=12)
    parser.add_argument("--n_heightmap", type=int, default=132)
    parser.add_argument("--activation", type=str, default="elu")

    parser.add_argument("--scan_encoder_dims", type=int, nargs="+", default=[128, 64, 32])
    parser.add_argument("--actor_hidden_dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--priv_encoder_dims", type=int, nargs="+", default=[64, 20])
    parser.add_argument("--heightmap_backbone_hidden_dims", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--heightmap_backbone_output_dim", type=int, default=32)
    parser.add_argument("--heightmap_hidden_size", type=int, default=512)
    parser.add_argument("--heightmap_output_dim", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device)
    load_run = os.path.join(args.log_root, args.proj_name, args.exptid)
    load_path, checkpoint = get_load_path(load_run, checkpoint=args.checkpoint)
    ckpt = torch.load(load_path, map_location=device)

    actor = build_actor(args).to(device)
    encoder = build_encoder(args).to(device)
    actor.load_state_dict(ckpt["heightmap_actor_state_dict"], strict=True)
    encoder.load_state_dict(ckpt["heightmap_encoder_state_dict"], strict=True)
    actor.eval()
    encoder.eval()

    out_dir = os.path.join(os.path.dirname(load_path), "traced")
    os.makedirs(out_dir, exist_ok=True)

    actor_wrapper = HeightmapActorWrapper(actor).to(device).eval()
    obs = torch.zeros(1, args.n_proprio + args.n_scan + args.n_priv_explicit + args.n_priv_latent + args.history_len * args.n_proprio, device=device)
    terrain_latent = torch.zeros(1, args.heightmap_output_dim, device=device)
    traced_actor = torch.jit.trace(actor_wrapper, (obs, terrain_latent))
    actor_path = os.path.join(out_dir, f"{args.exptid}-{checkpoint}-heightmap_actor.pt")
    traced_actor.save(actor_path)

    heightmap = torch.zeros(1, args.n_heightmap, device=device)
    proprio = torch.zeros(1, args.n_proprio_student, device=device)
    traced_encoder = torch.jit.trace(encoder, (heightmap, proprio))
    encoder_path = os.path.join(out_dir, f"{args.exptid}-{checkpoint}-heightmap_encoder.pt")
    traced_encoder.save(encoder_path)

    print("saved", actor_path)
    print("saved", encoder_path)


if __name__ == "__main__":
    main()
