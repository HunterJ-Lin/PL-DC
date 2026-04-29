import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MoELoRALinear(nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 lora_ranks,
                 lora_alpha=1.0,
                 num_experts=1,
                 finetune_last_expert=False):

        super(MoELoRALinear, self).__init__()
        assert len(lora_ranks) == num_experts
        self.lora_ranks = lora_ranks
        self.lora_alpha = lora_alpha
        self.num_experts = num_experts
        self.lora_A_weight = nn.ParameterList([
            nn.init.xavier_normal_(nn.Parameter(
                torch.empty(
                    in_features,
                    lora_ranks[i]
                ).float())
            ) for i in range(num_experts)
        ])
        self.lora_A_bias = nn.ParameterList([
            nn.init.zeros_(nn.Parameter(
                torch.empty(
                    lora_ranks[i]
                ).float())
            ) for i in range(num_experts)
        ])
        self.lora_B_weight = nn.ParameterList([
            nn.init.zeros_(nn.Parameter(
                torch.empty(
                    lora_ranks[i],
                    out_features
                ).float())
            ) for i in range(num_experts)
        ])
        self.lora_B_bias = nn.ParameterList([
            nn.init.zeros_(nn.Parameter(
                torch.empty(
                    out_features
                ).float())
            ) for i in range(num_experts)
        ])

        if isinstance(finetune_last_expert, bool):
            if finetune_last_expert:
                for param in self.lora_A_weight[:-1]:
                    param.requires_grad = False
                for param in self.lora_A_bias[:-1]:
                    param.requires_grad = False
                for param in self.lora_B_weight[:-1]:
                    param.requires_grad = False
                for param in self.lora_B_bias[:-1]:
                    param.requires_grad = False
        elif isinstance(finetune_last_expert, str):
            # must be finetune-last-k-experts
            match = re.match(
                r"finetune-last-(\d+)-experts", finetune_last_expert)
            if match:
                last_k_experts = int(match.group(1))
            else:
                raise ValueError(
                    f'if finetune_last_expert is str, '
                    f'it must be "finetune-last-k-experts", '
                    f'but got {finetune_last_expert}')
            if last_k_experts <= 0:
                raise ValueError(
                    f'last_k_experts must be positive, '
                    f'but got {last_k_experts}')
            if last_k_experts > num_experts:
                raise ValueError(
                    f'last_k_experts must be less than num_experts, '
                    f'but got {last_k_experts} and {num_experts}')
            for param in self.lora_A_weight[:-last_k_experts]:
                param.requires_grad = False
            for param in self.lora_A_bias[:-last_k_experts]:
                param.requires_grad = False
            for param in self.lora_B_weight[:-last_k_experts]:
                param.requires_grad = False
            for param in self.lora_B_bias[:-last_k_experts]:
                param.requires_grad = False
        else:
            raise NotImplementedError(
                f'finetune_last_expert must be bool or str, '
                f'but got {finetune_last_expert}'
            )

        if num_experts > 1:
            self.routing = nn.Parameter(torch.zeros(num_experts).float())
            self.has_updated_routing = True
            self.routing_info = None

    def forward(self, x):
        if self.num_experts == 1:
            return self._forward(x, 0)

        if self.training:
            self.has_updated_routing = True
            weights, indices = self.routing.topk(2, dim=0)
            weights = weights.softmax(dim=0)
            top1, top2 = indices.unbind(0)  # noqa
            w1, w2 = weights.unbind(0)  # noqa
            self.routing_info = (top1.item(), top2.item(), w1, w2)
        else:
            if self.has_updated_routing:
                weights, indices = self.routing.topk(2, dim=0)
                weights = weights.softmax(dim=0)
                top1, top2 = indices.unbind(0)  # noqa
                w1, w2 = weights.unbind(0)  # noqa
                self.routing_info = (
                    top1.item(), top2.item(), w1.item(), w2.item())
                self.has_updated_routing = False
        top1, top2, w1, w2 = self.routing_info
        return w1 * self._forward(x, top1) + w2 * self._forward(x, top2)

    def _forward(self, x, index):
        return torch.einsum(
            '...j, jk -> ...k',
            self.lora_alpha * (torch.einsum(
                '...i, ij -> ...j',
                x,
                self.lora_A_weight[index]
            ) + self.lora_A_bias[index]),
            self.lora_B_weight[index]
        ) + self.lora_B_bias[index]

    
class MoELoRAFFN(nn.Module):

    def __init__(self,
                 d_model=256, 
                 d_ffn=1024,
                 dropout=0.1, 
                 activation="relu",
                 num_experts=1,
                 freeze_linear_params=True,
                 finetune_last_expert=False,
                 lora_ranks=[0],
                 lora_alpha=1.0,
        ):
        super(MoELoRAFFN, self).__init__()
        assert num_experts == len(lora_ranks)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        if freeze_linear_params:
            for param in self.parameters():
                param.requires_grad = False

        self.linear1_lora = MoELoRALinear(
            d_model, d_ffn,
            lora_ranks=lora_ranks,
            lora_alpha=lora_alpha,
            num_experts=num_experts,
            finetune_last_expert=finetune_last_expert)
        self.linear2_lora = MoELoRALinear(
            d_ffn, d_model,
            lora_ranks=lora_ranks,
            lora_alpha=lora_alpha,
            num_experts=num_experts,
            finetune_last_expert=finetune_last_expert)

    def forward_ffn(self, x):
        x = self.linear1(x) + self.linear1_lora(x)
        x = self.dropout3(self.activation(x))
        x = self.linear2(x) + self.linear2_lora(x)
        x = self.dropout4(x)
        x = self.norm3(x)
        return x
