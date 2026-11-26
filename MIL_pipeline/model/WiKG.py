import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention

torch.autograd.set_detect_anomaly(True)


class WiKG_(nn.Module):
    def __init__(self, dim_in=384, dim_hidden=512, topk=6, n_classes=2, agg_type='bi-interaction', dropout=0.3,
                 pool='attn'):
        super().__init__()

        self._fc1 = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.LeakyReLU())

        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)

        self.scale = dim_hidden ** -0.5
        self.topk = topk
        self.agg_type = agg_type

        self.gate_U = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_V = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_W = nn.Linear(dim_hidden // 2, dim_hidden)

        if self.agg_type == 'gcn':
            self.linear = nn.Linear(dim_hidden, dim_hidden)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(dim_hidden * 2, dim_hidden)
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim_hidden)
        self.fc = nn.Linear(dim_hidden, n_classes)

        if pool == "mean":
            self.readout = global_mean_pool
        elif pool == "max":
            self.readout = global_max_pool
        elif pool == "attn":
            att_net = nn.Sequential(nn.Linear(dim_hidden, dim_hidden // 2), nn.LeakyReLU(),
                                    nn.Linear(dim_hidden // 2, 1))
            self.readout = GlobalAttention(att_net)

    def forward(self, x, *args):
        """Accept a full bag tuple, feature and mask inputs, or features alone."""
        # Identify the input format.
        if len(args) == 6:  # Complete six-item bag input
            features, _, _, _, mask, _ = args
        elif len(args) == 5:  # Coordinates or labels may be omitted.
            # Infer the format from the second argument.
            if isinstance(args[1], torch.Tensor) and args[1].dim() == 1:  # The second argument may contain labels.
                features, _, _, mask, _ = args
            else:
                features, mask = args[0], args[1]
        elif len(args) == 2:  # Features and mask
            features, mask = args
        elif len(args) == 1:  # Features only
            features = args[0]
            mask = None
        else:
            raise ValueError(f"Unsupported number of input arguments: {len(args)}")

        # Normalize the inputs.
        try:
            if isinstance(features, dict):
                features = features["feature"]
        except:
            pass
        # Apply the validity mask.
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x = self._fc1(x)  # [B,N,C]

        # B, N, C = x.shape
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5

        e_h = self.W_head(x)
        e_t = self.W_tail(x)

        # construct neighbour
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  # 1
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)

        # add an extra dimension to the index tensor, making it available for advanced indexing, aligned with the dimensions of e_t
        topk_index = topk_index.to(torch.long)

        # expand topk_index dimensions to match e_t
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)  # shape: [1, 10000, 4]

        # create a RANGE tensor to help indexing
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)  # shape: [1, 1, 1]

        Nb_h = e_t[batch_indices, topk_index_expanded, :]  # shape: [1, 10000, 4, 512]

        # use SoftMax to obtain probability
        topk_prob = F.softmax(topk_weight, dim=2)
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1),
                                                                       e_h.unsqueeze(2))  # 1 pixel wise   2 matmul

        # gated knowledge attention
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)

        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)

        if self.agg_type == 'gcn':
            embedding = e_h + e_Nh
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'sage':
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding

        h = self.message_dropout(embedding)

        h = self.readout(h.squeeze(0), batch=None)
        h = self.norm(h)
        h = self.fc(h)

        return h


class WiKG(nn.Module):
    """Weakly supervised knowledge graph aggregator with configurable input and hidden widths, neighbors, graph aggregation, dropout, and pooling."""

    def __init__(self, n_classes=2, in_dim=512, hidden_dim=512, topk=6,
                 agg_type='bi-interaction', dropout=0.3, pool='attn', *args, **kwargs):
        super().__init__()
        self.model = WiKG_(
            dim_in=in_dim,
            dim_hidden=hidden_dim,
            topk=topk,
            n_classes=n_classes,
            agg_type=agg_type,
            dropout=dropout,
            pool=pool
        )

    def forward(self, *args, **kwargs):
        """Accept a full bag tuple, features and a mask, or features alone."""
        # Prefer an explicitly supplied mask.
        mask = kwargs.get('mask', None)

        # Dispatch according to the input arity.
        if mask is not None:
            # With an explicit mask, the first positional argument provides the features.
            return self.model(args[0] if args else None, mask)
        else:
            return self.model(*args)


# Example configuration and forward pass
if __name__ == "__main__":
    # Use the same configuration structure as ABMIL.
    config = {
        'model': {
            'class': 'WiKG',  # Model class
            'params': {
                'n_classes': 4,  # Number of classes
                'in_dim': 1536,  # Input feature width
                'hidden_dim': 512,  # Hidden feature width
                'topk': 6,  # Number of neighbors
                'agg_type': 'bi-interaction',  # Aggregation method
                'dropout': 0.3,  # Dropout rate
                'pool': 'attn'  # Pooling method
            }
        },
        'class_names': ["Cholangiocarcinoma", "Ampullary", "Duodenal", "Pancreatic"],
    }

    # Construct the configured model.
    model_params = config['model']['params']
    model = WiKG(**model_params)

    # Synthetic inputs
    data = torch.randn((2, 1000, 1536))  # batch_size=2, seq_len=1000, feature_dim=1536
    mask = torch.ones((2, 1000))  # Mark all features as valid.

    # Run a forward pass.
    output = model(data, mask)
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")