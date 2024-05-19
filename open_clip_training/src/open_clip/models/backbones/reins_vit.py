import torch

from .reins import Reins
from .utils import set_requires_grad, set_train
from .vit import VisualTransformer


class ReinsVisualTransformer(VisualTransformer):
    def __init__(
            self,
            reins_return_query: bool,
            **kwargs):
        super().__init__(**kwargs)
        self.reins: Reins = Reins(kwargs['layers'], kwargs['width'], self.patch_size[0],
                                  link_token_to_query=reins_return_query)

    def forward(self, x: torch.Tensor, m=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        if m is not None:
            m = self.mask_pool(m.to(torch.float)).reshape(m.shape[0], -1).unsqueeze(-1)
            m = torch.ceil(m)
            # mask_embedding = self.mask_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            if self.mask_embedding.shape[1] == 1:
                mask_embedding = self.mask_embedding.to(x.dtype).repeat(1, x.shape[1], 1)
            else:
                mask_embedding = self.mask_embedding.to(x.dtype)
            # mask_embedding = mask_embedding.repeat(1, x.shape[1], 1)
            x = x * m + mask_embedding[0].unsqueeze(0) * (1 - m)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        for i, blk in enumerate(self.transformer.resblocks):
            d = i + 1
            x = blk(x, attn_mask=None)
            if m is not None and d < self.mask_emb_depth:
                masked_x = x[1:, :, :] * m.permute(1, 0, 2) + \
                           mask_embedding[d].unsqueeze(0).permute(1, 0, 2) * (1 - m.permute(1, 0, 2))
                x = torch.cat([x[:1, :, :], masked_x], dim=0)
            x = self.reins.forward(x, i, batch_first=False, has_cls_token=True)
            # if i in self.out_indices:
            #     xp = x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return self.reins.return_auto(x)

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins"])
        set_train(self, ["reins"])
