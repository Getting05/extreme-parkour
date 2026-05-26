import torch
import torch.nn as nn


class HeightmapMLPBackbone(nn.Module):
    """MLP backbone for encoding 1D heightmap data from LiDAR.

    Input: (batch, n_points) heightmap elevation values
    Output: (batch, output_dim) compressed terrain feature
    """
    def __init__(self, n_points, output_dim=32, hidden_dims=[128, 64]):
        super().__init__()
        activation = nn.ELU()

        layers = []
        layers.append(nn.Linear(n_points, hidden_dims[0]))
        layers.append(activation)
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(activation)

        self.encoder = nn.Sequential(*layers)

    def forward(self, heightmap):
        """
        Args:
            heightmap: (batch, n_points) terrain elevation samples
        Returns:
            (batch, output_dim) compressed terrain feature
        """
        return self.encoder(heightmap)


class HeightmapEncoder(nn.Module):
    """Recurrent heightmap encoder that replaces the depth encoder.

    Takes LiDAR heightmap + proprioception (without foot contacts) as input,
    produces a terrain latent (same dim as scan_encoder output) + yaw estimate.

    Architecture:
        HeightmapMLPBackbone(n_points -> 32)
        + concat proprio_student(n_proprio_student)
        -> combination_mlp(32 + n_proprio_student -> 32)
        -> GRU(32 -> hidden_size)
        -> output_mlp(hidden_size -> output_dim + 2)

    Output: [scandots_latent (output_dim), yaw (2)]
    """
    def __init__(self, backbone, n_proprio_student, hidden_size=512, output_dim=32):
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()

        self.backbone = backbone
        self.output_dim = output_dim

        # Combine backbone output with proprioception
        self.combination_mlp = nn.Sequential(
            nn.Linear(output_dim + n_proprio_student, 128),
            activation,
            nn.Linear(128, output_dim)
        )

        # Temporal modeling with GRU
        self.rnn = nn.GRU(input_size=output_dim, hidden_size=hidden_size, batch_first=True)
        self.hidden_states = None

        # Output: terrain latent + yaw estimate
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_size, output_dim + 2),
            last_activation
        )

    def forward(self, heightmap, proprioception):
        """
        Args:
            heightmap: (batch, n_points) LiDAR heightmap samples
            proprioception: (batch, n_proprio_student) proprio without foot contacts
        Returns:
            (batch, output_dim + 2) terrain latent concatenated with yaw estimate
        """
        # Encode heightmap
        terrain_feature = self.backbone(heightmap)
        # Combine with proprioception
        combined = self.combination_mlp(torch.cat((terrain_feature, proprioception), dim=-1))
        # Temporal processing
        combined, self.hidden_states = self.rnn(combined[:, None, :], self.hidden_states)
        # Output projection
        output = self.output_mlp(combined.squeeze(1))
        return output

    def detach_hidden_states(self):
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.detach().clone()

    def reset_hidden_states(self, dones=None):
        """Reset hidden states for done environments."""
        if self.hidden_states is not None and dones is not None:
            self.hidden_states[:, dones.bool().squeeze(), :] = 0.0
