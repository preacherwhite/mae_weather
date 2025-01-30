# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.layers import trunc_normal_
from util.pos_embed import get_2d_sincos_pos_embed
from parallel_patch_embed import ParallelVarPatchEmbed

class ChannelAttention(nn.Module):
    """
    Perform channel self-attention on [B, C, H, W],
    assigning one attention-driven scale factor per channel.
    
    Steps:
      1) Flatten each channel’s (H, W) -> vector of length H*W
      2) Learn an embedding from size (H*W) -> embed_dim
      3) Multi-head self-attention across the C channels
      4) Produce a scale factor per channel, broadcast to (H, W)
      5) Multiply original x by that scale factor
    """
    def __init__(self, in_chans, embed_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Project from (H*W) => embed_dim for each channel
        # (We create this linear AFTER flattening H*W.)
        # This is effectively a "learned aggregator" over the spatial dimension.
        self.aggregator = nn.Linear(
            in_features=0,   # We will set this properly later (see note below)
            out_features=embed_dim,
            bias=True
        )

        # Multi-head attention: expects [seq_len, batch_size, embed_dim]
        # where seq_len = C (the channels), batch_size = B, embed_dim = E
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

        # Positional embedding for each of the C channels: shape [1, C, E]
        self.pos_embed = nn.Parameter(torch.zeros(1, in_chans, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)

        # Optionally, after attention we create a gating factor per channel
        # that scales the original x. We'll generate a shape [B, C, 1, 1].
        self.gate = nn.Linear(embed_dim, 1, bias=True)

        # We'll fill in aggregator.in_features later if you want to set it dynamically.
        # Alternatively, you can just specify a fixed H, W in the constructor.


    def random_masking_channels(self, x, mask_ratio: float):
        """
        Optionally mask out some fraction of channels (MAE-style).
        x: [B, C, H, W]
        mask_ratio: fraction of channels to mask
        """
        B, C, H, W = x.shape
        len_keep = int(C * (1 - mask_ratio))

        noise = torch.rand(B, C, device=x.device)
        # Sort channels by noise for each sample in the batch
        ids_shuffle = torch.argsort(noise, dim=1)       # [B, C]
        ids_restore = torch.argsort(ids_shuffle, dim=1) # [B, C]

        # Keep the first 'len_keep' channels
        ids_keep = ids_shuffle[:, :len_keep]            # [B, len_keep]
        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        )

        # Construct the binary mask: 0=keep, 1=masked
        mask = torch.ones([B, C], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)  # reorder back to original channel order

        return x_masked, mask, ids_restore


    def forward(self, x, channel_mask_ratio=0.0):
        """
        x: [B, C, H, W]
        Return: x_out of same shape, after channel-wise attention scaling.
        """
        B, C, H, W = x.shape

        # If aggregator.in_features was left as 0, we set it dynamically:
        if self.aggregator.in_features == 0:
            self.aggregator = nn.Linear(H*W, self.embed_dim, bias=True).to(x.device)

        # 1) Optional channel masking
        if channel_mask_ratio > 0:
            x_masked, mask, ids_restore = self.random_masking_channels(x, channel_mask_ratio)
        else:
            x_masked = x
            mask = None
            ids_restore = None

        # 2) Flatten spatial dims -> shape [B, C, H*W]
        x_flat = x_masked.view(B, C, -1)  # => [B, C, H*W]

        # 3) aggregator: project from (H*W) => embed_dim
        #    So each channel becomes a length-E vector for each sample.
        #    => shape [B, C, E]
        x_embed = self.aggregator(x_flat)

        # 4) Add channel positional embedding:
        #    pos_embed shape is [1, C, E], broadcast to [B, C, E]
        x_embed = x_embed + self.pos_embed

        # 5) Reorder for multi‐head attention:
        #    MHA wants [seq_len, batch_size, embed_dim].
        #    Here, seq_len = C (the channels), batch_size = B.
        x_embed = x_embed.permute(1, 0, 2)  # => [C, B, E]

        # 6) Self‐attention across channels
        attn_out, _ = self.attn(x_embed, x_embed, x_embed)  # => [C, B, E]

        # 7) Permute back => [B, C, E]
        attn_out = attn_out.permute(1, 0, 2)

        # 8) Compute a single gating scalar per channel
        #    => shape [B, C, 1]
        gate_weight = self.gate(attn_out)  # => [B, C, 1]

        # 9) Broadcast -> [B, C, 1, 1]
        gate_weight = gate_weight.unsqueeze(-1)

        # 10) Scale the original x (masked or unmasked) with the learned attention weights
        x_out = x_masked * gate_weight

        # If you masked channels, you could (optionally) restore them as zeros or do something else
        # For now we just keep x_out as is. The unmasked channels are scaled. The masked channels are gone.

        # Return the updated feature map
        if mask is not None:
            return x_out, mask, ids_restore
        else:
            return x_out

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # Channel attention block
        self.channel_attention = ChannelAttention(in_chans=in_chans, 
                                                embed_dim=embed_dim, 
                                                num_heads=num_heads)
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
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

    def patchify(self, imgs):
        """
        imgs: (N, c, H, W)
        x: (N, L, patch_size**2 *c)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        c = imgs.shape[1]
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *c)
        imgs: (N, c, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        c = x.shape[2] // (p**2)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, channel_mask_ratio=0.2):

        #Apply channel attention and masking
        if channel_mask_ratio > 0:
            x, channel_mask, channel_ids_restore = self.channel_attention(x, channel_mask_ratio)
        else:
            x = self.channel_attention(x)
            channel_mask = None
            channel_ids_restore = None
        
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore#, channel_mask, channel_ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, channel_mask_ratio=0.2):
        latent, mask, ids_restore= self.forward_encoder(imgs, mask_ratio, channel_mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask#, channel_mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
