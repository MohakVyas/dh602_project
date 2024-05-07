import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
import os
import cv2
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import collections.abc
import torch.nn as nn

# Define constants
num_classes = 6
input_shape = (32, 32, 1)  # Grayscale, so channel is 1
patch_size = (2, 2)  # 2-by-2 sized patches
dropout_rate = 0  # Dropout rate
num_heads = 8  # Attention heads
embed_dim = 64  # Embedding dimension
num_mlp = 128  # MLP layer size
# Convert embedded patches to query, key, and values with a learnable additive
# value
qkv_bias = True
window_size = 2  # Size of attention window
shift_size = 1  # Size of shifting window
image_dimension = 32  # Initial image size

num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[1] // patch_size[1]

learning_rate = 1e-3
batch_size = 16
num_epochs = 2
validation_split = 0.1
weight_decay = 0.0001
label_smoothing = 0.5

# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (input_shape[1], input_shape[0]))  # Resize to input_shape
        if self.transform:
            image = self.transform(image)
        label = self.df.iloc[idx, 1]
        return image, label
    

def window_partition(x, window_size):
    B, H, W, C = x.shape
    patch_num_y = H // window_size
    patch_num_x = W // window_size
    x = x.view(B, patch_num_y, window_size, patch_num_x, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5)
    windows = x.reshape(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = windows.view(-1, patch_num_y, patch_num_x, window_size, window_size, channels)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(-1, height, width, channels)
    return x

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)

        num_window_elements = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_window_elements, num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_matrix = torch.meshgrid(coords_h, coords_w)
        coords = torch.stack(coords_matrix)
        coords_flatten = coords.view(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer('relative_position_index', relative_coords.sum(-1))

    def forward(self, x, mask=None):
        B, N, C = x.shape
        head_dim = C // self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = self.relative_position_index.view(-1)
        relative_position_bias = self.relative_position_bias_table[relative_position_index_flat].view(
            num_window_elements, num_window_elements, -1
        ).permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            attn += mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x_qkv = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_qkv = self.proj(x_qkv)
        return x_qkv
    


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # print(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        # print("H:", H)
        # print("W:", W)
        # print(x.shape)
        # print("hello")
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = x + self.pos_embed
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, num_patch, embed_dim):
        super().__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = nn.Linear(4 * embed_dim, 2 * embed_dim, bias=False)

    def forward(self, x):
        batch_size, numt, C = x.shape
        height,width = self.num_patch
        # batch_size, height, width, C = x.size()
        x = x.view(batch_size, height, width, C)
        # print("initial_patch_merging:", x.size())
        x = x.permute(0, 3, 1, 2)  # Change dimensions to (batch_size, C, height, width)
        # print("after_permute:", x.size())
        x = x.reshape(batch_size, C, height // 2, 2, width // 2, 2)
        # print("after_reshape:", x.size())
        x = x.permute(0, 1, 3, 5, 2, 4)
        # print("after_permute:", x.size())
        x = x.reshape(batch_size, -1, height // 2, width // 2)
        # print("after_reshape:", x.size())
        x = x.permute(0, 2, 3, 1)  # Change dimensions back to (batch_size, height // 2, width // 2, ...)
        # print("after_permute:", x.size())
        x = x.reshape(batch_size, -1, 4 * C)
        # print("after_reshape:", x.size())
        temp = self.linear_trans(x)
        # print("temp_size:", temp.size())
        return temp
    

class LayerNormalization(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SwinTransformer(nn.Module):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0,
    ):
        super(SwinTransformer, self).__init__()

        self.dim = dim
        self.num_patch = num_patch
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_mlp = num_mlp

        self.norm1 = LayerNormalization(dim)
        self.attn = WindowAttention(dim, window_size =(window_size,window_size), num_heads = num_heads, qkv_bias = qkv_bias, dropout_rate = dropout_rate)
        self.drop_path = nn.Dropout(dropout_rate)
        self.norm2 = LayerNormalization(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, num_mlp),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_mlp, dim),
            nn.Dropout(dropout_rate)
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def forward(self, x):
        height, width = self.num_patch
        batch_size, num_patches_before, channels = x.shape
        x_skip = x.clone()
        x = self.norm1(x)
        x = x.view(batch_size, height, width, channels)
        
        if self.shift_size > 0:
            shifted_x = torch.cat((x[:, :, self.shift_size:, :], x[:, :, :self.shift_size, :]), dim=2)
            shifted_x = torch.cat((shifted_x[:, :, :, self.shift_size:], shifted_x[:, :, :, :self.shift_size]), dim=3)
        else:
            shifted_x = x
        
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)
        attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels)
        shifted_x = window_reverse(attn_windows, self.window_size, height, width, channels)
        
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(batch_size, height * width, channels)
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x.clone()
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        # print("swin_layer_x_shape:", x.shape)
        return x
    

# Custom transform for random crop on single-channel images
class RandomCropSingleChannel(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img):
        h, w = img.shape[:2]
        new_h, new_w = self.output_size

        if new_h > h or new_w > w:
            raise ValueError(f"Required crop size {self.output_size} is larger than input image size {(h, w)}")

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        return img[top: top + new_h, left: left + new_w]

class SwinModel(nn.Module):
    def __init__(self, embed_dim, num_patch_x, num_patch_y, num_heads, num_mlp, window_size, shift_size, qkv_bias, num_classes):
        super(SwinModel, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size=(32,32), patch_size=(2,2), embed_dim=64)
        # PatchEmbedding(image_size=(32,32), patch_size=(2,2), embed_dim=64)
        self.swin_transformer1 = SwinTransformer(
            dim=embed_dim,
            num_patch=(num_patch_x, num_patch_y),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            num_mlp=num_mlp,
            qkv_bias=qkv_bias,
            dropout_rate=0.0
        )
        self.swin_transformer2 = SwinTransformer(
            dim=embed_dim,
            num_patch=(num_patch_x, num_patch_y),
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            num_mlp=num_mlp,
            qkv_bias=qkv_bias,
            dropout_rate=0.0
        )
        self.patch_merging = PatchMerging((num_patch_x, num_patch_y), embed_dim)
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print("swin_model_x_shape:", x.shape)
        x = x.permute(0, 3, 1, 2)
        x = self.patch_embedding(x)
        x = self.swin_transformer1(x)
        x = self.swin_transformer2(x)
        x = self.patch_merging(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = self.softmax(x)
        return x
    
