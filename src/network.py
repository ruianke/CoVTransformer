# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from timm.models.vision_transformer import PatchEmbed, Block

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from sklearn.ensemble import RandomForestRegressor
from pos_embed import get_2d_sincos_pos_embed

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class mlp_block(nn.Module):
    def __init__(self, dim1, dim2, activation='gelu', norm=True, dropout=0.0):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.activation = activation
        self.norm = norm
        self.dropout = dropout

        self.activation_dict = {'sigmodi': nn.Sigmoid(), 'tanh': nn.Tanh(), 'softmax': nn.Softmax(dim=-1),
                                'relu': nn.ReLU(),
                                'elu': nn.ELU(), 'swish': nn.SiLU(), 'gelu': nn.GELU()}

        layers = [torch.nn.Linear(dim1, dim2)]

        if self.norm:
            layers.append(nn.BatchNorm1d(dim2))

        if self.activation:
            layers.append(self.activation_dict[self.activation])

        layers.append(nn.Dropout(self.dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class InverNetDataset(Dataset):

    def __init__(self, embedding, label):
        self.data_list = embedding
        self.label_list = label

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.label_list[idx]

        return data, label

    def __len__(self):
        return self.label_list.shape[0]


class InverNetDataset_name(Dataset):

    def __init__(self, embedding, label, name):
        self.data_list = embedding
        self.label_list = label
        self.name_list = name

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.label_list[idx]
        name = self.name_list[idx]

        return data, label, name[0]

    def __len__(self):
        return self.label_list.shape[0]


class InverNetDataset_name2(Dataset):

    def __init__(self, embedding, label, label_smooth, smooth_label, nan_ind, name):
        self.data_list = embedding
        self.label_list = label
        self.label_smooth_list = label_smooth
        self.smooth_label_list = smooth_label
        self.nan_ind = nan_ind
        self.name_list = name

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.label_list[idx]
        label_smooth = self.label_smooth_list[idx]
        smooth_label = self.smooth_label_list[idx]
        nan_ind = self.nan_ind[idx]
        name = self.name_list[idx]

        return data, label, label_smooth, smooth_label, nan_ind, name[0]

    def __len__(self):
        return self.label_list.shape[0]


class InverNetDataset_name3(Dataset):

    def __init__(self, embedding, nan_ind, name):
        self.data_list = embedding
        self.nan_ind = nan_ind
        self.name_list = name

    def __getitem__(self, idx):
        data = self.data_list[idx]
        nan_ind = self.nan_ind[idx]
        name = self.name_list[idx]

        return data, nan_ind, name[0]

    def __len__(self):
        return self.data_list.shape[0]


"""### network"""


class encoder(nn.Module):
    def __init__(self, n, m, dim=[512, 64], activation=None, norm=False, dropout=0.0, **kwargs):
        super().__init__()
        self.n = n
        self.m = m

        self.activation = activation
        self.norm = norm
        self.dropout = dropout

        layers = [mlp_block(n, dim[0], activation, norm, dropout)]

        for i in range(1, len(dim)):
            layers.append(mlp_block(dim[i - 1], dim[i], activation, norm, dropout))

        layers.append(mlp_block(dim[-1], m, None, norm, dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        output = x

        return output


class transformer_encoder(nn.Module):
    # Transformer
    def __init__(self, n, m, dim=[512, 64], activation=None, norm=False, dropout=0.0, day=5, second=False, **kwargs):
        super().__init__()
        self.n = n
        self.m = m
        self.day = day

        self.activation = activation
        self.norm = norm
        self.dropout = dropout

        if not second:
            self.patch_embd = nn.Linear(n // day, dim[0])
        else:
            self.patch_embd = nn.Linear(n // day + 1, dim[0])
        self.pos_embed = nn.Parameter(torch.zeros(1, day, dim[0]), requires_grad=False)

        self.norm1 = nn.LayerNorm(dim[0])

        layers = [Block(dim[0], 2, qkv_bias=True, drop_path=dropout) for _ in range(len(dim))]
        self.block = nn.Sequential(*layers)

        self.norm2 = nn.LayerNorm(dim[0])

        self.layers = mlp_block(day * dim[0], m, '', False, dropout)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.day, 1), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, recent=None):
        x = x.reshape([x.shape[0], self.day, -1])
        if recent is not None:
            # recent = torch.full((x.shape[0], self.day, 1), recent).to(x.device)
            recent = recent.view(-1, 1, 1)
            tem = recent.expand(-1, self.day, 1).to(x.device)
            x = torch.cat((x, tem), dim=-1)
        x = self.patch_embd(x)
        x = x + self.pos_embed

        if self.norm:
            x = self.norm1(x)
        x = self.block(x)
        if self.norm:
            x = self.norm2(x)
        x = x.reshape([x.shape[0], -1])
        x = self.layers(x)
        
        if recent == None:
            output = x
        else:
            output = x + recent.reshape([x.shape[0], -1])

        return output
    

class ensamble_encoder(nn.Module):
    def __init__(self, n, m, p, dim=[512, 64], activation=None, norm=False, dropout=0.0, day=5, model=transformer_encoder, second=False, **kwargs):
        super().__init__()
        
        self.recent_models = nn.ModuleList([model(n, m, dim=dim, activation=activation, norm=norm, dropout=dropout, day=day, second=second) for _ in range(p)])
        
        # Freeze the parameters of the recent_model
        for recent_model in self.recent_models:
            for param in recent_model.parameters():
                param.requires_grad = False

    def forward(self, x, recent=None):
        recent_outputs = [recent_model(x, recent) for recent_model in self.recent_models]
        recent_outputs = [torch.clamp(out[:, 0].unsqueeze(1), max=0) for out in recent_outputs]
        
        recent_outputs = torch.stack(recent_outputs, dim=0)
        output = torch.mean(recent_outputs, dim=0)
        return output


class moe(nn.Module):
    def __init__(self, n, m, dim=[512, 64], activation=None, norm=False, dropout=0.0, day=5, model=transformer_encoder, second=False, p=5,
                 **kwargs):
        super().__init__()

        self.gate_model = model(n, 2, dim=dim, activation=activation, norm=norm, dropout=dropout, day=day, second=second)
        self.low_model = nn.ModuleList([model(n, m, dim=dim, activation=activation, norm=norm, dropout=dropout, day=day, second=second) for _ in range(p)])
        self.high_model = nn.ModuleList([model(n, m, dim=dim, activation=activation, norm=norm, dropout=dropout, day=day, second=second) for _ in range(p)])

    def forward(self, x, recent=None):
        weight = self.gate_model(x, recent)
        low = [low_model(x, recent) for low_model in self.low_model]
        high = [high_model(x, recent) for high_model in self.high_model]

        low = torch.mean(torch.stack(low, dim=0), dim=0)
        high = torch.mean(torch.stack(high, dim=0), dim=0)

        output = low * weight[:,0].unsqueeze(1) + high * weight[:,1].unsqueeze(1)
        return output


class transformer_seq_encoder(nn.Module):
    # Transformer
    def __init__(self, n, m, dim=[512, 64], activation=None, norm=False, dropout=0.0, day=5, second=False, token=-4, **kwargs):
        super().__init__()
        self.n = n
        self.m = m
        self.day = day
        self.token = token

        self.activation = activation
        self.norm = norm
        self.dropout = dropout

        if not second:
            self.patch_embd = nn.Linear(n // day, dim[0])
        else:
            self.patch_embd = nn.Linear(n // day + 1, dim[0])
        self.pos_embed = nn.Parameter(torch.zeros(1, day + self.m*7*5, dim[0]), requires_grad=False)

        self.norm1 = nn.LayerNorm(dim[0])

        layers = [Block(dim[0], 2, qkv_bias=True, drop_path=dropout) for _ in range(len(dim))]
        self.block = nn.Sequential(*layers)

        self.norm2 = nn.LayerNorm(dim[0])

        self.layers = mlp_block(day * dim[0], m*7, '', False, dropout)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.day + self.m*7*5, 1), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_week(self, x):
        x = self.patch_embd(x)
        x = x + self.pos_embed[:, :x.shape[1], :]

        if self.norm:
            x = self.norm1(x)
        x = self.block(x)
        if self.norm:
            x = self.norm2(x)
        x = x[:, -self.day:, :].reshape([x.shape[0], -1])
        x = self.layers(x)
        return x

    def forward(self, x, recent=None):
        x = x.reshape([x.shape[0], self.day, -1])

        output = []
        for i in range(5):
            w = self.forward_week(x).unsqueeze(-1)  # B * 7m * 1
            output.append(w.squeeze(-1))
            p = torch.ones([w.shape[0], w.shape[1], x.shape[-1]-1]).to(x.device)
            p_temp_1 = p[:, :, :-1].detach() * self.token
            p_temp_2 = p[:, :, -1].detach() * x[:, :self.m * 7, -1].detach()
            p = torch.cat((p_temp_1, p_temp_2.unsqueeze(-1)), dim=-1)
            w = torch.cat((w, p), -1)
            x = torch.cat((x, w), dim=1)

        output = torch.cat(output, dim=1)

        return output


def load_pretrained_models(ensamble_model, checkpoint_paths):
    new_state_dict = OrderedDict()
    
    for i, (recent_model, checkpoint_path) in enumerate(zip(ensamble_model.recent_models, checkpoint_paths)):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_state_dict = checkpoint['model']

        for k, v in model_state_dict.items():
            name = f"recent_models.{i}." + k  # add prefix to match ensamble_encoder's recent_models list
            new_state_dict[name] = v

    # Use the new state dict to update the ensamble_model parameters
    msg = ensamble_model.load_state_dict(new_state_dict, strict=False)
    print(msg)


def load_pretrained_models_moe(moe_model, low_model_paths, high_model_paths):
    """
    This function loads the pretrained weights from low_model_path and high_model_path
    into the low_model and high_model of moe_model.

    Args:
    moe_model (moe): The moe model where the pretrained weights will be loaded.
    low_model_path (str): Path to the .pth file of the pretrained transformer_encoder for low_model.
    high_model_path (str): Path to the .pth file of the pretrained transformer_encoder for high_model.
    """
    new_state_dict = OrderedDict()
    for i, checkpoint_path in enumerate(low_model_paths):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_state_dict = checkpoint['model']

        for k, v in model_state_dict.items():
            name = f"{i}." + k  # add prefix to match ensamble_encoder's recent_models list
            new_state_dict[name] = v
    moe_model.low_model.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    for i, checkpoint_path in enumerate(high_model_paths):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_state_dict = checkpoint['model']

        for k, v in model_state_dict.items():
            name = f"{i}." + k  # add prefix to match ensamble_encoder's recent_models list
            new_state_dict[name] = v
    moe_model.high_model.load_state_dict(new_state_dict)


class transformer_encoder_larger(nn.Module):
    # Transformer
    def __init__(self, n, m, dim=[512, 64], activation=None, norm=False, dropout=0.0, day=5, **kwargs):
        super().__init__()
        self.n = n
        self.m = m
        self.day = day

        self.activation = activation
        self.norm = norm
        self.dropout = dropout

        self.patch_embd = nn.Linear(n // day, dim[0])
        self.pos_embed = nn.Parameter(torch.zeros(1, day, dim[0]), requires_grad=False)

        self.norm1 = nn.LayerNorm(dim[0])

        layers = [Block(dim[0], 12, qkv_bias=True, drop_path=dropout) for _ in range(len(dim))]
        self.block = nn.Sequential(*layers)

        self.norm2 = nn.LayerNorm(dim[0])

        self.layers = mlp_block(day * dim[0], m, '', False, dropout)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.day, 1), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x.reshape([x.shape[0], self.day, -1])
        x = self.patch_embd(x)
        x = x + self.pos_embed

        if self.norm:
            x = self.norm1(x)
        x = self.block(x)
        if self.norm:
            x = self.norm2(x)
        x = x.reshape([x.shape[0], -1])
        x = self.layers(x)
        output = x

        return output


class transformer_reg_classify_encoder(nn.Module):
    # Transformer
    def __init__(self, n, m, dim=[512, 64], activation=None, norm=False, dropout=0.0, day=5, **kwargs):
        super().__init__()
        self.n = n
        self.m = m
        self.day = day

        self.activation = activation
        self.norm = norm
        self.dropout = dropout

        self.patch_embd = nn.Linear(n // day, dim[0])
        self.pos_embed = nn.Parameter(torch.zeros(1, day, dim[0]), requires_grad=False)

        self.norm1 = nn.LayerNorm(dim[0])

        layers = [Block(dim[0], 2, qkv_bias=True, drop_path=dropout) for _ in range(len(dim))]
        self.block = nn.Sequential(*layers)

        self.norm2 = nn.LayerNorm(dim[0])

        self.layers = mlp_block(day * dim[0], m*2, '', False, dropout)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.day, 1), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x.reshape([x.shape[0], self.day, -1])
        x = self.patch_embd(x)
        x = x + self.pos_embed

        if self.norm:
            x = self.norm1(x)
        x = self.block(x)
        if self.norm:
            x = self.norm2(x)
        x = x.reshape([x.shape[0], -1])
        x = self.layers(x)
        output = x

        return output


class transformer_encoder2(nn.Module):
    # Transformer
    def __init__(self, n, m, dim=[512, 64], activation=None, norm=False, dropout=0.0, day=5, second=False, **kwargs):
        super().__init__()
        self.n = n
        self.m = m
        self.day = day

        self.activation = activation
        self.norm = norm
        self.dropout = dropout

        if not second:
            self.patch_embd = nn.Linear(n // day, dim[0])
        else:
            self.patch_embd = nn.Linear(n // day + 1, dim[0])
        self.pos_embed = nn.Parameter(torch.zeros(1, day, dim[0]), requires_grad=False)

        self.norm1 = nn.LayerNorm(dim[0])

        layers = [Block(dim[0], 2, qkv_bias=True, drop_path=dropout) for _ in range(len(dim))]
        self.block = nn.Sequential(*layers)

        self.norm2 = nn.LayerNorm(dim[0])

        self.layers = mlp_block(day * dim[0], m, '', False, dropout)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.day, 1), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, recent=None):
        x = x.reshape([x.shape[0], self.day, -1])

        token = -4
        valid = torch.where(x[:, :, 0] != token)
        is_token = torch.where(x[:, :, 0] == token)
        ind = (torch.cat([valid[0], is_token[0]]), torch.cat([valid[1], is_token[1]]))
        sorted_indices = torch.argsort(ind[0])
        ind = (ind[0][sorted_indices], ind[1][sorted_indices])

        if recent is not None:
            # recent = torch.full((x.shape[0], self.day, 1), recent).to(x.device)
            recent = recent.view(-1, 1, 1)
            tem = recent.expand(-1, self.day, 1).to(x.device)
            x = torch.cat((x, tem), dim=-1)
        x = self.patch_embd(x)
        x = x + self.pos_embed

        x[is_token] = token
        x = x[ind].view(x.shape[0], self.day, -1)

        if self.norm:
            x = self.norm1(x)
        x = self.block(x)
        if self.norm:
            x = self.norm2(x)
        x = x.reshape([x.shape[0], -1])
        x = self.layers(x)

        if recent == None:
            output = x
        else:
            output = x + recent.reshape([x.shape[0], -1])

        return output


class transformer_token_encoder(nn.Module):
    # Transformer
    def __init__(self, n, m, dim=[512, 64], activation=None, norm=False, dropout=0.0, day=5, token=-4, **kwargs):
        super().__init__()
        self.n = n
        self.m = m
        self.day = day

        self.token = token

        if 5 >= n//day > 2:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, n // day-2))
        elif n//day > 5:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, n // day-4))
        else:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, n//day))

        self.activation = activation
        self.norm = norm
        self.dropout = dropout

        self.patch_embd = nn.Linear(n//day, dim[0])
        self.pos_embed = nn.Parameter(torch.zeros(1, day, dim[0]), requires_grad=False)  # fixed sin-cos embedding

        self.norm1 = nn.LayerNorm(dim[0])

        layers = [Block(dim[0], 2, qkv_bias=True, drop_path=dropout) for _ in range(len(dim))]
        self.block = nn.Sequential(*layers)

        self.norm2 = nn.LayerNorm(dim[0])

        self.layers = mlp_block(day*dim[0], m, '', False, dropout)

        self.print = False

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.day, 1), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x.reshape([x.shape[0], self.day, -1])

        if self.print:
            print('origin:')
            print(x[0])

        if 5 >= self.n//self.day > 2:
            mask = ((x[:,:,:-2] == self.token).sum(dim=2) >= (self.n//self.day - 2)) & ((x[:,:,:-2] == self.token).any(dim=2))
            x[:,:,:-2][mask] = self.mask_token.expand(x.shape[0], x.shape[1], -1)[mask]
        elif self.n//self.day > 5:
            mask = ((x[:,:,:-4] == self.token).sum(dim=2) >= (self.n//self.day - 4)) & ((x[:,:,:-4] == self.token).any(dim=2))
            x[:,:,:-4][mask] = self.mask_token.expand(x.shape[0], x.shape[1], -1)[mask]
        else:
            mask = ((x == self.token).any(dim=2))
            x[mask] = self.mask_token.expand(x.shape[0], x.shape[1], -1)[mask]

        if self.print:
            print('token:', self.n//self.day)
            print(x[0])
            self.print = False


        x = self.patch_embd(x)
        x = x + self.pos_embed

        if self.norm:
          x = self.norm1(x)

        x = self.block(x)

        if self.norm:
          x = self.norm2(x)

        x = x.reshape([x.shape[0], -1])

        x = self.layers(x)
        output = x

        return output


class cnn_encoder(nn.Module):
    # Transformer
    def __init__(self, n, m, dim=[512, 64], activation=None, norm=False, dropout=0.0, day=5, **kwargs):
        super().__init__()
        self.n = n
        self.m = m
        self.day = day

        self.activation = activation
        self.norm = norm
        self.dropout = dropout

        self.patch_embd = nn.Linear(n // day, dim[0])

        self.norm1 = nn.LayerNorm(dim[0])

        layers = [nn.Conv1d(dim[0], dim[0], 3, stride=1, padding=1) for _ in range(len(dim))]
        self.block = nn.Sequential(*layers)

        self.norm2 = nn.LayerNorm(dim[0])

        self.layers = mlp_block(day * dim[0], m, '', False, dropout)

    def forward(self, x):
        x = x.reshape([x.shape[0], self.day, -1])
        x = self.patch_embd(x)
        if self.norm:
            x = self.norm1(x)
        x = self.block(x.permute(0,2,1)).permute(0,2,1)
        if self.norm:
            x = self.norm2(x)
        x = x.reshape([x.shape[0], -1])
        x = self.layers(x)
        output = x

        return output


class LSTM_encoder(nn.Module):
    def __init__(self, n, m, dim=[512, 64], activation=None, norm=False, dropout=0.0, day=5, **kwargs):
        super().__init__()
        self.n = n
        self.m = m
        self.day = day

        self.activation = activation
        self.norm = norm
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=n//day, hidden_size=dim[0], batch_first=True, num_layers=len(dim), dropout=dropout if len(dim) > 1 else 0)

        self.norm1 = nn.LayerNorm(dim[0])

        self.layers = mlp_block(day * dim[0], m, '', False, dropout)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x.reshape([x.shape[0], self.day, -1])

        x, _ = self.lstm(x)

        if self.norm:
            x = self.norm1(x)

        x = x.reshape([x.shape[0], -1])
        x = self.layers(x)
        output = x

        return output



def ensamble_expReg(x, day, future):
    reg_input = x.reshape([x.shape[0], day, -1])[:, :, 0].detach().numpy()
    reg_input = np.log10(reg_input)

    result = []

    for i in range(reg_input.shape[0]):
        d = reg_input[i]
        x = np.arange(d.shape[0])

        clf = LinearRegression()
        # clf = Ridge(alpha=1.0)

        clf.fit(x.reshape(-1, 1), d.reshape(-1, 1))

        pred = clf.predict(np.array(d.shape[0] + future).reshape(-1, 1))
        pred = np.minimum(pred, 0)

        result.append(torch.tensor(pred))

    result = torch.cat(result)

    return result


"""### Loss"""


def censor_mse(label, pred, c=0.001, c2=0.001, mean=True):
    c = torch.log10(torch.tensor(c))
    c2 = torch.log10(torch.tensor(c2))

    mask = (label >= c).float()
    mask2 = (pred > c).float()

    high = (pred - label) ** 2 * mask
    low = torch.clamp(pred - c2, min=0) ** 2 * (1 - mask) * mask2

    loss = high + low
    if mean:
        loss = torch.mean(loss)

    return loss


def censor_mae(label, pred, c=0.001, c2=0.001, mean=True):
    c = torch.log10(torch.tensor(c))
    c2 = torch.log10(torch.tensor(c2))

    mask = (label >= c).float()
    mask2 = (pred > c).float()

    high = torch.abs((pred - label)) * mask
    low = torch.abs(torch.clamp(pred - c2, min=0)) * (1 - mask) * mask2

    loss = high + low
    if mean:
        loss = torch.mean(loss)

    return loss


def combain_censor_loss(label, pred):
    l1 = censor_mae(label, pred)
    l2 = censor_mse(label, pred)
    l = torch.mean(l1 + l2)

    return l


def weighted_combain_loss_quad(label, pred):
    weight = (1/30)*(label**2)+(17/30)*(label)+2

    l1 = weight * torch.abs(label - pred)
    l2 = weight * (label - pred) ** 2
    l = torch.mean(l1 + l2)

    return l


def weighted_combain_loss_tanh(label, pred):
    weight = torch.tanh(label) * 2 + 2

    l1 = weight * torch.abs(label - pred)
    l2 = weight * (label - pred) ** 2
    l = torch.mean(l1 + l2)

    return l


def weighted_combain_loss_tanh2(label, pred):
    k = (2/(1+torch.tanh(torch.tensor(-1))))
    weight = torch.tanh(label-1)*k+k

    l1 = weight * torch.abs(label - pred)
    l2 = weight * (label - pred) ** 2
    l = torch.mean(l1 + l2)

    return l

def weighted_combain_loss_censored_tanh(label, pred, c=0.00316, c2=0.001): # c=0.0316
    k = (2/(1+torch.tanh(torch.tensor(-1))))
    weight = torch.tanh(label-1)*k+k

    l1 = weight * censor_mae(label, pred, c=c, c2=c2, mean=False)
    l2 = weight * censor_mse(label, pred, c=c, c2=c2, mean=False)
    l = torch.mean(l1 + l2)

    return l

def weighted_combain_loss_piece_linear(label, pred):
    mask = (label<-1).float()
    weight = (0.001*label+0.005) * mask + (1.5233*label+2) * (1-mask)
    
    l1 = weight * torch.abs(label - pred)
    l2 = weight * (label - pred) ** 2
    l = torch.mean(l1 + l2)

    return l


def weighted_combain_loss_mask(label, pred):
    mask = (label < -2).float()
    weight = torch.pow(10, label + 2) * mask + (0.5 * label + 2) * (1 - mask)

    l1 = weight * torch.abs(label - pred)
    l2 = weight * (label - pred) ** 2
    l = torch.mean(l1 + l2)

    return l


def weighted_combain_loss_exp(label, pred):
    weight = torch.exp(label) * 2

    l1 = weight * torch.abs(label - pred)
    l2 = weight * (label - pred) ** 2
    l = torch.mean(l1 + l2)

    return l

def weighted_combain_loss_exp2(label, pred):
    weight = torch.exp(label) * 3

    l1 = weight * torch.abs(label - pred)
    l2 = weight * (label - pred) ** 2
    l = torch.mean(l1 + l2)

    return l

def weighted_combain_loss_exp3(label, pred):
    weight = torch.exp(label) * 2 + 1

    l1 = weight * torch.abs(label - pred)
    l2 = weight * (label - pred) ** 2
    l = torch.mean(l1 + l2)

    return l


def uncertainty_loss(y, pred):
    pred_y, s = pred[:, 0], pred[:, 1]
    return torch.mean(torch.exp(-1.0 * s) * (pred_y - y) ** 2 + s)


def combain_loss(label, pred):
    l1 = torch.abs(label - pred)
    l2 = (label - pred) ** 2
    l = torch.mean(l1 + l2)

    return l


class quantile_loss(nn.Module):
    # Transformer
    def __init__(self, tau=None):
        super().__init__()
        if tau is None:
            self.tau = [0.1, 0.5, 0.9]
        else:
            self.tau = tau

    def forward(self, y_true, y_pred):
        total_loss = 0
        for t in self.tau:
            error = y_true - y_pred
            loss = torch.max(t * error, (t - 1) * error)
            total_loss += torch.mean(loss)
        return total_loss
    
def classify_high_freq_loss(label, pred, c=-1.5):
    binary_label = (label >= c).float()
    l = torch.nn.functional.binary_cross_entropy_with_logits(pred, binary_label)
    
def classify_loss(label, pred, weight):
    l = torch.nn.functional.cross_entropy(pred, label, weight)

    return l

"""### Tree"""


def Tree_classfier(input_feature, x_train, y_train, x_test, y_test, dims=[8], ensemble=False):
    # x_train = StandardScaler().fit_transform(x_train)
    # x_test = StandardScaler().fit_transform(x_test)

    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()
    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test).float()

    y_train = torch.log10(y_train)
    y_test = torch.log10(y_test)

    if ensemble:
        reg_result = ensamble_expReg(x_train)
        x_train = torch.cat([x_train, reg_result], dim=-1).float()

        reg_result = ensamble_expReg(x_test)
        x_test = torch.cat([x_test, reg_result], dim=-1).float()

    if input_feature == 4:
        lows = min([torch.amin(y_train), torch.amin(torch.log10(x_train[:, 0])), torch.amin(y_test),
                    torch.amin(torch.log10(x_test[:, 0]))])
        highs = max([torch.amax(y_train), torch.amax(torch.log10(x_train[:, 0])), torch.amax(y_test),
                     torch.amax(torch.log10(x_test[:, 0]))])
        # print(lows, highs)

        lows = min(lows, 0)

        y_train = (y_train - lows)  # / (highs-lows)
        y_test = (y_test - lows)  # / (highs-lows)

        x_train[:, 0] = (torch.log10(x_train[:, 0]) - lows)  # / (highs-lows)
        x_test[:, 0] = (torch.log10(x_test[:, 0]) - lows)  # / (highs-lows)

    clf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=dims[0], max_features=28)
    # clf = BaggingRegressor(base_estimator=MLPRegressor(hidden_layer_sizes=(32,64,), learning_rate='adaptive', max_iter=500), max_samples=1200, n_estimators=10)
    # clf = AdaBoostRegressor(base_estimator=MLPRegressor(hidden_layer_sizes=(64,128,64,32,), learning_rate='adaptive'), random_state=0, n_estimators=10)

    clf.fit(x_train.numpy(), y_train.numpy())

    pred = clf.predict(x_test.numpy())

    pred = torch.tensor(pred)
    true = torch.tensor(y_test.numpy())

    cmae = censor_mae(true, pred)
    cmse = censor_mse(true, pred)
    mae = torch.mean(torch.abs(pred - true))
    mse = torch.mean((pred - true) ** 2)
    # print(f'MAE: {mae}, MSE: {mse}, CMAE: {cmae}, CMSE: {cmse}')
    # print()

    # c = torch.log10(torch.tensor(0.01))
    # mask = (true >= c).float()
    # high  = torch.abs((pred-true)) * mask
    # low = torch.abs(torch.clamp(pred-c, min=0)) * (1-mask)
    # err = high + low

    # plt.boxplot(err)
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.scatter(true, pred, alpha=0.2)

    # lims = [
    #     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    #     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    # ]
    # ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    # ax.plot([lims[0], lims[1]], [-2, -2], 'g--', alpha=0.75, zorder=0)
    # ax.plot([-2, -2], [lims[0], lims[1]], 'g--', alpha=0.75, zorder=0)
    # ax.set_aspect('equal')
    # ax.set_xlim(lims)
    # ax.set_ylim(lims)
    # plt.show()

    return true, pred, mae, mse, cmae, cmse, clf


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    nf = 7
    input= torch.rand((2,35*nf))

    input[0,:nf] = -4
    input[0, 5:nf] = 99

    print(input.shape)

    model = transformer_seq_encoder(nf*35, 1, [8], activation='gelu', norm=True, dropout=0.0, day=35, token=-4)

    output = model(input)

    print(output.shape)

    params = parameter_count_table(model)
    print(params)

    # pre_model = transformer_encoder(nf*35,1,[8], activation='gelu', norm=True, dropout=0.0, day=35, token=-4)
    #
    # stat = pre_model.state_dict()
    #
    # for key in stat:
    #     print(key)
    #
    # print('--------'*10)
    #
    # for key in model.low_model.state_dict():
    #     print(key)
    #
    # model.low_model.load_state_dict(stat)
    # model.high_model.load_state_dict(stat)
    # print('load success')