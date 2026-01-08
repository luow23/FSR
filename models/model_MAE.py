import random
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from models.utils import get_2d_sincos_pos_embed
from torch.nn.functional import adaptive_avg_pool2d
import torch.nn.functional as F
# from losses.ssim_loss import SSIM_Loss


class Conv_Blcok(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x
class out_order_AE_rec_and_oo(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=960,
                 embed_dim=768, depth=8, num_heads=12, shuffle_ratio=0.25,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super(out_order_AE_rec_and_oo, self).__init__()
        self.len_keep = 0  # 初始化
        self.in_chans = in_chans
        self.shuffle_ratio = shuffle_ratio
        # ------------------------------    --------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.inpainting_pred = nn.Linear(embed_dim, patch_size ** 2 * in_chans, bias=True)

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
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
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs



    def my_patchify(self, imgs, p):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def my_unpatchify(self, x, p):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs



    def out_order(self, x):
        indices = torch.randperm(x.shape[1])
        return x[:, indices]


    def rand_shuffle(self, x, shuffle_ratio):
        N, L, D = x.shape

        len_keep = int(L*(1-shuffle_ratio))
        self.len_keep = len_keep
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        ids_not_keep = ids_shuffle[:, len_keep:]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_not_keep = torch.gather(x, dim=1, index=ids_not_keep.unsqueeze(-1).repeat(1, 1, D))
        # x_1 = torch.cat([x_random_shuffle, x_keep])
        x_not_keep_shuffled = self.out_order(x_not_keep)
        x_ = torch.cat([x_keep, x_not_keep_shuffled], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        return x_

    def forward_encoder(self, x, stage):

        # self.auxiliary_feature = self.auxiliary_feature

        x = self.patch_embed(x)
        x1 = x + self.pos_embed
        for blk in self.blocks:
            x1 = blk(x1)
        x1 = self.norm(x1)
        x1 = self.inpainting_pred(x1)
        if stage == 'train':
            x_out_order = self.rand_shuffle(x, self.shuffle_ratio)
            x2 = x_out_order + self.pos_embed
            for blk in self.blocks:
                x2 = blk(x2)
            x2 = self.norm(x2)
            x2 = self.inpainting_pred(x2)
            return x1, x2
        else:
            return x1

    def global_diff(self, input_feature, rec_feature):
        input_feature = input_feature.reshape(input_feature.shape[0], input_feature.shape[1], -1).permute(0, 2, 1)
        rec_feature = rec_feature.reshape(rec_feature.shape[0], rec_feature.shape[1], -1).permute(0, 2, 1)  # [B, HW, C]
        # target = F.cosine_similarity(input_feature.permute(0, 2, 1), input_feature.permute(0, 2, 1), dim=-1)
        input_feature_norm = input_feature / torch.norm(input_feature, dim=-1, keepdim=True)
        rec_feature_norm = rec_feature / torch.norm(rec_feature, dim=-1, keepdim=True)
        target_dis = torch.softmax(torch.matmul(input_feature_norm, input_feature_norm.permute(0, 2, 1)), dim=-1)
        rec_dis = torch.softmax(torch.matmul(rec_feature_norm, input_feature_norm.permute(0, 2, 1)), dim=-1)
        KL_div = torch.sum(target_dis * (torch.log(target_dis) - torch.log(rec_dis)), dim=-1)
        return KL_div
    def forward_loss(self, imgs, pred, pred2):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = self.patchify(imgs)
        target = imgs
        pred = self.unpatchify(pred)
        pred2 = self.unpatchify(pred2)
        # N, L, _ = target.shape
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6) ** .5

        # dis_loss = (pred - target) ** 2 #
        # dis_loss = dis_loss.mean(dim=-1)  # [N, L], mean loss per patch
        mse_loss = torch.mean((pred-target)**2, dim=1)

        #
        cos_loss_k1 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 1), self.my_patchify(target, 1))
        # cos_loss_k2 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 2), self.my_patchify(target, 2))
        # cos_loss_k4 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 4), self.my_patchify(target, 4))
        # cos_loss_k8 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 8), self.my_patchify(target, 8))
        # cos_loss_k16 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 16), self.my_patchify(target, 16))
        # cos_loss_k32 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 32), self.my_patchify(target, 32))
        cos_loss_k64 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 64), self.my_patchify(target, 64))
        # global_loss = self.global_diff(target, pred)
        # dis_loss_nor = (pred_nor - target) ** 2
        # dis_loss_nor = dis_loss_nor.mean(dim=-1)  # [N, L], mean loss per patch
        # dir_loss_nor = 1 - torch.nn.CosineSimilarity(-1)(pred_nor, target)

        # auxi_loss = torch.mean(self.anomaly_score, dim=1) -  0.1*torch.sum(self.diff_cos, dim=[1, 2]) / (
        #             self.center_num * self.center_num - self.center_num)
        # auxi_loss = torch.mean(self.anomaly_score, dim=1)
        mse_loss_2 = torch.mean((pred2 - target) ** 2, dim=1)

        #
        cos_loss_k1_2 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred2, 1), self.my_patchify(target, 1))
        cos_loss_k64_2 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred2, 64), self.my_patchify(target, 64))
        loss_g = cos_loss_k1.mean() + cos_loss_k64.mean() + mse_loss.mean()
        loss_g_2 = cos_loss_k1_2.mean() + cos_loss_k64_2.mean() + mse_loss_2.mean()
        # loss_g = 5 * dir_loss.mean() + dis_loss.mean()
        # loss_g = auxi_loss.mean()
        # print(f'cos: f{5 * dir_loss.mean()} mse:{dis_loss.mean()}, auxi:{auxi_loss.mean()}')
        # loss_g = 5 * dir_loss.mean() + dis_loss.mean()

        return loss_g + loss_g_2

    def forward(self, imgs, stage):

        # pred = self.forward_decoder(latent1)  # [N, L, p*p*3]
        if stage == "train":
            x1, x2 = self.forward_encoder(imgs, stage)
            loss = self.forward_loss(imgs, x1, x2)
        else:
            x1 = self.forward_encoder(imgs, stage)
            loss = 0.

        return loss, x1


class out_order_AE(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=960,
                 embed_dim=768, depth=8, num_heads=12, shuffle_ratio=0.25,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super(out_order_AE, self).__init__()
        self.len_keep = 0  # 初始化
        self.in_chans = in_chans
        self.shuffle_ratio = shuffle_ratio
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)


        num_patches = self.patch_embed.num_patches


        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.inpainting_pred = nn.Linear(embed_dim, patch_size ** 2 * in_chans, bias=True)

        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
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
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs


    def my_patchify(self, imgs, p):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def my_unpatchify(self, x, p):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs


    def out_order(self, x):
        indices = torch.randperm(x.shape[1])
        return x[:, indices]


    def rand_shuffle(self, x,  shuffle_ratio):
        N, L, D = x.shape

        len_keep = int(L*(1-shuffle_ratio))
        self.len_keep = len_keep
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        ids_not_keep = ids_shuffle[:, len_keep:]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_not_keep = torch.gather(x, dim=1, index=ids_not_keep.unsqueeze(-1).repeat(1, 1, D))
        # x_1 = torch.cat([x_random_shuffle, x_keep])
        x_not_keep_shuffled = self.out_order(x_not_keep)
        x_ = torch.cat([x_keep, x_not_keep_shuffled], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        return x_

    def add_jitter(self, feature_tokens, scale, prob):
        # feature_tokens = self.my_patchify(feature_tokens, p=1)
        if random.uniform(0, 1) <= prob:
            batch_size, num_tokens, dim_channel = feature_tokens.shape
            feature_norms = (
                feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((batch_size, num_tokens, dim_channel), device=feature_tokens.device)
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens
    def forward_encoder(self, x, stage):

        # self.auxiliary_feature = self.auxiliary_feature
        # x1 = self.norm(x1)
        # x1 = self.inpainting_pred(x1)
        # if stage == 'train':
        #     x = self.add_jitter(x, 20, 1)
        x = self.patch_embed(x)
        if stage == 'train':
            x = self.add_jitter(x, 30, 1)
            x = self.rand_shuffle(x, self.shuffle_ratio)
        x1 = x + self.pos_embed
        for blk in self.blocks:
            x1 = blk(x1)
        x1 = self.norm(x1)
        x1 = self.inpainting_pred(x1)
        return x1

    # def global_diff(self, input_feature, rec_feature):
    #     input_feature = input_feature.reshape(input_feature.shape[0], input_feature.shape[1], -1).permute(0, 2, 1)
    #     rec_feature = rec_feature.reshape(rec_feature.shape[0], rec_feature.shape[1], -1).permute(0, 2, 1)  # [B, HW, C]
    #     # target = F.cosine_similarity(input_feature.permute(0, 2, 1), input_feature.permute(0, 2, 1), dim=-1)
    #     input_feature_norm = input_feature / torch.norm(input_feature, dim=-1, keepdim=True)
    #     rec_feature_norm = rec_feature / torch.norm(rec_feature, dim=-1, keepdim=True)
    #     target_dis = torch.softmax(torch.matmul(input_feature_norm, input_feature_norm.permute(0, 2, 1)), dim=-1)
    #     rec_dis = torch.softmax(torch.matmul(rec_feature_norm, input_feature_norm.permute(0, 2, 1)), dim=-1)
    #     KL_div = torch.sum(target_dis * (torch.log(target_dis) - torch.log(rec_dis)), dim=-1)
    #     return KL_div
    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = self.patchify(imgs)
        target = imgs
        pred = self.unpatchify(pred)
        # pred2 = self.unpatchify(pred2)
        # N, L, _ = target.shape
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6) ** .5

        # dis_loss = (pred - target) ** 2 #
        # dis_loss = dis_loss.mean(dim=-1)  # [N, L], mean loss per patch
        mse_loss = torch.mean((pred-target)**2, dim=1)

        #
        cos_loss_k1 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 1), self.my_patchify(target, 1))

        cos_loss_k64 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 64), self.my_patchify(target, 64))
        # global_loss = self.global_diff(target, pred)
        # dis_loss_nor = (pred_nor - target) ** 2
        # dis_loss_nor = dis_loss_nor.mean(dim=-1)  # [N, L], mean loss per patch
        # dir_loss_nor = 1 - torch.nn.CosineSimilarity(-1)(pred_nor, target)

        # auxi_loss = torch.mean(self.anomaly_score, dim=1) -  0.1*torch.sum(self.diff_cos, dim=[1, 2]) / (
        #             self.center_num * self.center_num - self.center_num)
        # auxi_loss = torch.mean(self.anomaly_score, dim=1)
        # mse_loss_2 = torch.mean((pred2 - target) ** 2, dim=1)
        #
        # #
        # cos_loss_k1_2 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred2, 1), self.my_patchify(target, 1))
        # cos_loss_k64_2 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred2, 64), self.my_patchify(target, 64))
        loss_g = cos_loss_k1.mean() + cos_loss_k64.mean() + mse_loss.mean()
        # loss_g_2 = cos_loss_k1_2.mean() + cos_loss_k64_2.mean() + mse_loss_2.mean()
        # loss_g = 5 * dir_loss.mean() + dis_loss.mean()
        # loss_g = auxi_loss.mean()
        # print(f'cos: f{5 * dir_loss.mean()} mse:{dis_loss.mean()}, auxi:{auxi_loss.mean()}')
        # loss_g = 5 * dir_loss.mean() + dis_loss.mean()

        return loss_g

    def forward(self, imgs, stage):
        x1 = self.forward_encoder(imgs, stage)

        # pred = self.forward_decoder(latent1)  # [N, L, p*p*3]
        if stage == "train":
            loss = self.forward_loss(imgs, x1)
        else:
            # x1 = self.forward_encoder(imgs, stage)
            loss = 0.

        return loss, x1


class out_order_AE_CNN(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=960,
                 embed_dim=768, depth=8, num_heads=12, shuffle_ratio=0.25,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super(out_order_AE_CNN, self).__init__()
        self.len_keep = 0  # 初始化
        self.in_chans = in_chans
        self.shuffle_ratio = shuffle_ratio
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.embed_dim = embed_dim

        num_patches = self.patch_embed.num_patches


        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Conv_Blcok(embed_dim)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.inpainting_pred = nn.Linear(embed_dim, patch_size ** 2 * in_chans, bias=True)

        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
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
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs


    def my_patchify(self, imgs, p):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def my_unpatchify(self, x, p):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs


    def embed_patchify(self, imgs, p):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.embed_dim, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.embed_dim))
        return x

    def embed_unpatchify(self, x, p):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.embed_dim))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.embed_dim, h * p, h * p))
        return imgs


    def out_order(self, x):
        indices = torch.randperm(x.shape[1])
        return x[:, indices]


    def rand_shuffle(self, x,  shuffle_ratio):
        N, L, D = x.shape

        len_keep = int(L*(1-shuffle_ratio))
        self.len_keep = len_keep
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        ids_not_keep = ids_shuffle[:, len_keep:]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_not_keep = torch.gather(x, dim=1, index=ids_not_keep.unsqueeze(-1).repeat(1, 1, D))
        # x_1 = torch.cat([x_random_shuffle, x_keep])
        x_not_keep_shuffled = self.out_order(x_not_keep)
        x_ = torch.cat([x_keep, x_not_keep_shuffled], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        return x_

    def add_jitter(self, feature_tokens, scale, prob):
        # feature_tokens = self.my_patchify(feature_tokens, p=1)
        if random.uniform(0, 1) <= prob:
            batch_size, num_tokens, dim_channel = feature_tokens.shape
            feature_norms = (
                feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((batch_size, num_tokens, dim_channel), device=feature_tokens.device)
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens
    def forward_encoder(self, x, stage):

        # self.auxiliary_feature = self.auxiliary_feature
        # x1 = self.norm(x1)
        # x1 = self.inpainting_pred(x1)
        # if stage == 'train':
        #     x = self.add_jitter(x, 20, 1)
        x = self.patch_embed(x)
        if stage == 'train':
            x = self.add_jitter(x, 30, 1)
            x = self.rand_shuffle(x, self.shuffle_ratio)
        # print(x.shape)
        x1 = self.embed_unpatchify(x, p=1)
        # print(x1.shape)
        for blk in self.blocks:
            x1 = blk(x1)
        x1 = self.embed_patchify(x1, p=1)
        x1 = self.norm(x1)
        x1 = self.inpainting_pred(x1)
        return x1

    # def global_diff(self, input_feature, rec_feature):
    #     input_feature = input_feature.reshape(input_feature.shape[0], input_feature.shape[1], -1).permute(0, 2, 1)
    #     rec_feature = rec_feature.reshape(rec_feature.shape[0], rec_feature.shape[1], -1).permute(0, 2, 1)  # [B, HW, C]
    #     # target = F.cosine_similarity(input_feature.permute(0, 2, 1), input_feature.permute(0, 2, 1), dim=-1)
    #     input_feature_norm = input_feature / torch.norm(input_feature, dim=-1, keepdim=True)
    #     rec_feature_norm = rec_feature / torch.norm(rec_feature, dim=-1, keepdim=True)
    #     target_dis = torch.softmax(torch.matmul(input_feature_norm, input_feature_norm.permute(0, 2, 1)), dim=-1)
    #     rec_dis = torch.softmax(torch.matmul(rec_feature_norm, input_feature_norm.permute(0, 2, 1)), dim=-1)
    #     KL_div = torch.sum(target_dis * (torch.log(target_dis) - torch.log(rec_dis)), dim=-1)
    #     return KL_div
    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = self.patchify(imgs)
        target = imgs
        pred = self.unpatchify(pred)
        # pred2 = self.unpatchify(pred2)
        # N, L, _ = target.shape
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6) ** .5

        # dis_loss = (pred - target) ** 2 #
        # dis_loss = dis_loss.mean(dim=-1)  # [N, L], mean loss per patch
        mse_loss = torch.mean((pred-target)**2, dim=1)

        #
        cos_loss_k1 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 1), self.my_patchify(target, 1))

        cos_loss_k64 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 64), self.my_patchify(target, 64))
        # global_loss = self.global_diff(target, pred)
        # dis_loss_nor = (pred_nor - target) ** 2
        # dis_loss_nor = dis_loss_nor.mean(dim=-1)  # [N, L], mean loss per patch
        # dir_loss_nor = 1 - torch.nn.CosineSimilarity(-1)(pred_nor, target)

        # auxi_loss = torch.mean(self.anomaly_score, dim=1) -  0.1*torch.sum(self.diff_cos, dim=[1, 2]) / (
        #             self.center_num * self.center_num - self.center_num)
        # auxi_loss = torch.mean(self.anomaly_score, dim=1)
        # mse_loss_2 = torch.mean((pred2 - target) ** 2, dim=1)
        #
        # #
        # cos_loss_k1_2 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred2, 1), self.my_patchify(target, 1))
        # cos_loss_k64_2 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred2, 64), self.my_patchify(target, 64))
        loss_g = cos_loss_k1.mean() + cos_loss_k64.mean() + mse_loss.mean()
        # loss_g_2 = cos_loss_k1_2.mean() + cos_loss_k64_2.mean() + mse_loss_2.mean()
        # loss_g = 5 * dir_loss.mean() + dis_loss.mean()
        # loss_g = auxi_loss.mean()
        # print(f'cos: f{5 * dir_loss.mean()} mse:{dis_loss.mean()}, auxi:{auxi_loss.mean()}')
        # loss_g = 5 * dir_loss.mean() + dis_loss.mean()

        return loss_g

    def forward(self, imgs, stage):
        x1 = self.forward_encoder(imgs, stage)

        # pred = self.forward_decoder(latent1)  # [N, L, p*p*3]
        if stage == "train":
            loss = self.forward_loss(imgs, x1)
        else:
            # x1 = self.forward_encoder(imgs, stage)
            loss = 0.

        return loss, x1

class out_order_AE_without_pos(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=960,
                 embed_dim=768, depth=8, num_heads=12, shuffle_ratio=0.25,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super(out_order_AE_without_pos, self).__init__()
        self.len_keep = 0  # 初始化
        self.in_chans = in_chans
        self.shuffle_ratio = shuffle_ratio
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)


        num_patches = self.patch_embed.num_patches


        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.inpainting_pred = nn.Linear(embed_dim, patch_size ** 2 * in_chans, bias=True)

        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
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
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs


    def my_patchify(self, imgs, p):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def my_unpatchify(self, x, p):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs


    def out_order(self, x):
        indices = torch.randperm(x.shape[1])
        return x[:, indices]


    def rand_shuffle(self, x,  shuffle_ratio):
        N, L, D = x.shape

        len_keep = int(L*(1-shuffle_ratio))
        self.len_keep = len_keep
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        ids_not_keep = ids_shuffle[:, len_keep:]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_not_keep = torch.gather(x, dim=1, index=ids_not_keep.unsqueeze(-1).repeat(1, 1, D))
        # x_1 = torch.cat([x_random_shuffle, x_keep])
        x_not_keep_shuffled = self.out_order(x_not_keep)
        x_ = torch.cat([x_keep, x_not_keep_shuffled], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        return x_

    def add_jitter(self, feature_tokens, scale, prob):
        # feature_tokens = self.my_patchify(feature_tokens, p=1)
        if random.uniform(0, 1) <= prob:
            batch_size, num_tokens, dim_channel = feature_tokens.shape
            feature_norms = (
                feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((batch_size, num_tokens, dim_channel), device=feature_tokens.device)
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens
    def forward_encoder(self, x, stage):

        # self.auxiliary_feature = self.auxiliary_feature
        # x1 = self.norm(x1)
        # x1 = self.inpainting_pred(x1)
        # if stage == 'train':
        #     x = self.add_jitter(x, 20, 1)
        x = self.patch_embed(x)
        if stage == 'train':
            x = self.add_jitter(x, 30, 1)
            x = self.rand_shuffle(x, self.shuffle_ratio)
        # x1 = x + self.pos_embed
        x1 = x
        for blk in self.blocks:
            x1 = blk(x1)
        x1 = self.norm(x1)
        x1 = self.inpainting_pred(x1)
        return x1

    # def global_diff(self, input_feature, rec_feature):
    #     input_feature = input_feature.reshape(input_feature.shape[0], input_feature.shape[1], -1).permute(0, 2, 1)
    #     rec_feature = rec_feature.reshape(rec_feature.shape[0], rec_feature.shape[1], -1).permute(0, 2, 1)  # [B, HW, C]
    #     # target = F.cosine_similarity(input_feature.permute(0, 2, 1), input_feature.permute(0, 2, 1), dim=-1)
    #     input_feature_norm = input_feature / torch.norm(input_feature, dim=-1, keepdim=True)
    #     rec_feature_norm = rec_feature / torch.norm(rec_feature, dim=-1, keepdim=True)
    #     target_dis = torch.softmax(torch.matmul(input_feature_norm, input_feature_norm.permute(0, 2, 1)), dim=-1)
    #     rec_dis = torch.softmax(torch.matmul(rec_feature_norm, input_feature_norm.permute(0, 2, 1)), dim=-1)
    #     KL_div = torch.sum(target_dis * (torch.log(target_dis) - torch.log(rec_dis)), dim=-1)
    #     return KL_div
    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = self.patchify(imgs)
        target = imgs
        pred = self.unpatchify(pred)
        # pred2 = self.unpatchify(pred2)
        # N, L, _ = target.shape
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6) ** .5

        # dis_loss = (pred - target) ** 2 #
        # dis_loss = dis_loss.mean(dim=-1)  # [N, L], mean loss per patch
        mse_loss = torch.mean((pred-target)**2, dim=1)

        #
        cos_loss_k1 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 1), self.my_patchify(target, 1))

        cos_loss_k64 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 64), self.my_patchify(target, 64))
        # global_loss = self.global_diff(target, pred)
        # dis_loss_nor = (pred_nor - target) ** 2
        # dis_loss_nor = dis_loss_nor.mean(dim=-1)  # [N, L], mean loss per patch
        # dir_loss_nor = 1 - torch.nn.CosineSimilarity(-1)(pred_nor, target)

        # auxi_loss = torch.mean(self.anomaly_score, dim=1) -  0.1*torch.sum(self.diff_cos, dim=[1, 2]) / (
        #             self.center_num * self.center_num - self.center_num)
        # auxi_loss = torch.mean(self.anomaly_score, dim=1)
        # mse_loss_2 = torch.mean((pred2 - target) ** 2, dim=1)
        #
        # #
        # cos_loss_k1_2 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred2, 1), self.my_patchify(target, 1))
        # cos_loss_k64_2 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred2, 64), self.my_patchify(target, 64))
        loss_g = cos_loss_k1.mean() + cos_loss_k64.mean() + mse_loss.mean()
        # loss_g_2 = cos_loss_k1_2.mean() + cos_loss_k64_2.mean() + mse_loss_2.mean()
        # loss_g = 5 * dir_loss.mean() + dis_loss.mean()
        # loss_g = auxi_loss.mean()
        # print(f'cos: f{5 * dir_loss.mean()} mse:{dis_loss.mean()}, auxi:{auxi_loss.mean()}')
        # loss_g = 5 * dir_loss.mean() + dis_loss.mean()

        return loss_g

    def forward(self, imgs, stage):
        x1 = self.forward_encoder(imgs, stage)

        # pred = self.forward_decoder(latent1)  # [N, L, p*p*3]
        if stage == "train":
            loss = self.forward_loss(imgs, x1)
        else:
            # x1 = self.forward_encoder(imgs, stage)
            loss = 0.

        return loss, x1


class out_order_AE_learned_pos(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=960,
                 embed_dim=768, depth=8, num_heads=12, shuffle_ratio=0.25,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super(out_order_AE_learned_pos, self).__init__()
        self.len_keep = 0  # 初始化
        self.in_chans = in_chans
        self.shuffle_ratio = shuffle_ratio
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)


        num_patches = self.patch_embed.num_patches


        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=True)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.inpainting_pred = nn.Linear(embed_dim, patch_size ** 2 * in_chans, bias=True)

        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
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
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs


    def my_patchify(self, imgs, p):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def my_unpatchify(self, x, p):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs


    def out_order(self, x):
        indices = torch.randperm(x.shape[1])
        return x[:, indices]


    def rand_shuffle(self, x,  shuffle_ratio):
        N, L, D = x.shape

        len_keep = int(L*(1-shuffle_ratio))
        self.len_keep = len_keep
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        ids_not_keep = ids_shuffle[:, len_keep:]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_not_keep = torch.gather(x, dim=1, index=ids_not_keep.unsqueeze(-1).repeat(1, 1, D))
        # x_1 = torch.cat([x_random_shuffle, x_keep])
        x_not_keep_shuffled = self.out_order(x_not_keep)
        x_ = torch.cat([x_keep, x_not_keep_shuffled], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        return x_

    def add_jitter(self, feature_tokens, scale, prob):
        # feature_tokens = self.my_patchify(feature_tokens, p=1)
        if random.uniform(0, 1) <= prob:
            batch_size, num_tokens, dim_channel = feature_tokens.shape
            feature_norms = (
                feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((batch_size, num_tokens, dim_channel), device=feature_tokens.device)
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens
    def forward_encoder(self, x, stage):

        # self.auxiliary_feature = self.auxiliary_feature
        # x1 = self.norm(x1)
        # x1 = self.inpainting_pred(x1)
        # if stage == 'train':
        #     x = self.add_jitter(x, 20, 1)
        x = self.patch_embed(x)
        if stage == 'train':
            x = self.add_jitter(x, 30, 1)
            x = self.rand_shuffle(x, self.shuffle_ratio)
        x1 = x + self.pos_embed
        for blk in self.blocks:
            x1 = blk(x1)
        x1 = self.norm(x1)
        x1 = self.inpainting_pred(x1)
        return x1

    # def global_diff(self, input_feature, rec_feature):
    #     input_feature = input_feature.reshape(input_feature.shape[0], input_feature.shape[1], -1).permute(0, 2, 1)
    #     rec_feature = rec_feature.reshape(rec_feature.shape[0], rec_feature.shape[1], -1).permute(0, 2, 1)  # [B, HW, C]
    #     # target = F.cosine_similarity(input_feature.permute(0, 2, 1), input_feature.permute(0, 2, 1), dim=-1)
    #     input_feature_norm = input_feature / torch.norm(input_feature, dim=-1, keepdim=True)
    #     rec_feature_norm = rec_feature / torch.norm(rec_feature, dim=-1, keepdim=True)
    #     target_dis = torch.softmax(torch.matmul(input_feature_norm, input_feature_norm.permute(0, 2, 1)), dim=-1)
    #     rec_dis = torch.softmax(torch.matmul(rec_feature_norm, input_feature_norm.permute(0, 2, 1)), dim=-1)
    #     KL_div = torch.sum(target_dis * (torch.log(target_dis) - torch.log(rec_dis)), dim=-1)
    #     return KL_div
    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = self.patchify(imgs)
        target = imgs
        pred = self.unpatchify(pred)
        # pred2 = self.unpatchify(pred2)
        # N, L, _ = target.shape
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6) ** .5

        # dis_loss = (pred - target) ** 2 #
        # dis_loss = dis_loss.mean(dim=-1)  # [N, L], mean loss per patch
        mse_loss = torch.mean((pred-target)**2, dim=1)

        #
        cos_loss_k1 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 1), self.my_patchify(target, 1))

        cos_loss_k64 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 64), self.my_patchify(target, 64))
        # global_loss = self.global_diff(target, pred)
        # dis_loss_nor = (pred_nor - target) ** 2
        # dis_loss_nor = dis_loss_nor.mean(dim=-1)  # [N, L], mean loss per patch
        # dir_loss_nor = 1 - torch.nn.CosineSimilarity(-1)(pred_nor, target)

        # auxi_loss = torch.mean(self.anomaly_score, dim=1) -  0.1*torch.sum(self.diff_cos, dim=[1, 2]) / (
        #             self.center_num * self.center_num - self.center_num)
        # auxi_loss = torch.mean(self.anomaly_score, dim=1)
        # mse_loss_2 = torch.mean((pred2 - target) ** 2, dim=1)
        #
        # #
        # cos_loss_k1_2 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred2, 1), self.my_patchify(target, 1))
        # cos_loss_k64_2 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred2, 64), self.my_patchify(target, 64))
        loss_g = cos_loss_k1.mean() + cos_loss_k64.mean() + mse_loss.mean()
        # loss_g_2 = cos_loss_k1_2.mean() + cos_loss_k64_2.mean() + mse_loss_2.mean()
        # loss_g = 5 * dir_loss.mean() + dis_loss.mean()
        # loss_g = auxi_loss.mean()
        # print(f'cos: f{5 * dir_loss.mean()} mse:{dis_loss.mean()}, auxi:{auxi_loss.mean()}')
        # loss_g = 5 * dir_loss.mean() + dis_loss.mean()

        return loss_g

    def forward(self, imgs, stage):
        x1 = self.forward_encoder(imgs, stage)

        # pred = self.forward_decoder(latent1)  # [N, L, p*p*3]
        if stage == "train":
            loss = self.forward_loss(imgs, x1)
        else:
            # x1 = self.forward_encoder(imgs, stage)
            loss = 0.

        return loss, x1



class out_order_AE_without_jitter(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=960,
                 embed_dim=768, depth=8, num_heads=12, shuffle_ratio=0.25,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super(out_order_AE_without_jitter, self).__init__()
        self.len_keep = 0  # 初始化
        self.in_chans = in_chans
        self.shuffle_ratio = shuffle_ratio
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)


        num_patches = self.patch_embed.num_patches


        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.inpainting_pred = nn.Linear(embed_dim, patch_size ** 2 * in_chans, bias=True)

        # --------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
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
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs


    def my_patchify(self, imgs, p):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def my_unpatchify(self, x, p):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs


    def out_order(self, x):
        indices = torch.randperm(x.shape[1])
        return x[:, indices]


    def rand_shuffle(self, x,  shuffle_ratio):
        N, L, D = x.shape

        len_keep = int(L*(1-shuffle_ratio))
        self.len_keep = len_keep
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        ids_not_keep = ids_shuffle[:, len_keep:]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_not_keep = torch.gather(x, dim=1, index=ids_not_keep.unsqueeze(-1).repeat(1, 1, D))
        # x_1 = torch.cat([x_random_shuffle, x_keep])
        x_not_keep_shuffled = self.out_order(x_not_keep)
        x_ = torch.cat([x_keep, x_not_keep_shuffled], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        return x_

    def add_jitter(self, feature_tokens, scale, prob):
        # feature_tokens = self.my_patchify(feature_tokens, p=1)
        if random.uniform(0, 1) <= prob:
            batch_size, num_tokens, dim_channel = feature_tokens.shape
            feature_norms = (
                feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((batch_size, num_tokens, dim_channel), device=feature_tokens.device)
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens
    def forward_encoder(self, x, stage):

        # self.auxiliary_feature = self.auxiliary_feature
        # x1 = self.norm(x1)
        # x1 = self.inpainting_pred(x1)
        # if stage == 'train':
        #     x = self.add_jitter(x, 20, 1)
        x = self.patch_embed(x)
        if stage == 'train':
            # x = self.add_jitter(x, 30, 1)
            x = self.rand_shuffle(x, self.shuffle_ratio)
        x1 = x + self.pos_embed
        for blk in self.blocks:
            x1 = blk(x1)
        x1 = self.norm(x1)
        x1 = self.inpainting_pred(x1)
        return x1

    # def global_diff(self, input_feature, rec_feature):
    #     input_feature = input_feature.reshape(input_feature.shape[0], input_feature.shape[1], -1).permute(0, 2, 1)
    #     rec_feature = rec_feature.reshape(rec_feature.shape[0], rec_feature.shape[1], -1).permute(0, 2, 1)  # [B, HW, C]
    #     # target = F.cosine_similarity(input_feature.permute(0, 2, 1), input_feature.permute(0, 2, 1), dim=-1)
    #     input_feature_norm = input_feature / torch.norm(input_feature, dim=-1, keepdim=True)
    #     rec_feature_norm = rec_feature / torch.norm(rec_feature, dim=-1, keepdim=True)
    #     target_dis = torch.softmax(torch.matmul(input_feature_norm, input_feature_norm.permute(0, 2, 1)), dim=-1)
    #     rec_dis = torch.softmax(torch.matmul(rec_feature_norm, input_feature_norm.permute(0, 2, 1)), dim=-1)
    #     KL_div = torch.sum(target_dis * (torch.log(target_dis) - torch.log(rec_dis)), dim=-1)
    #     return KL_div
    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = self.patchify(imgs)
        target = imgs
        pred = self.unpatchify(pred)
        # pred2 = self.unpatchify(pred2)
        # N, L, _ = target.shape
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6) ** .5

        # dis_loss = (pred - target) ** 2 #
        # dis_loss = dis_loss.mean(dim=-1)  # [N, L], mean loss per patch
        mse_loss = torch.mean((pred-target)**2, dim=1)

        #
        cos_loss_k1 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 1), self.my_patchify(target, 1))

        cos_loss_k64 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 64), self.my_patchify(target, 64))
        # global_loss = self.global_diff(target, pred)
        # dis_loss_nor = (pred_nor - target) ** 2
        # dis_loss_nor = dis_loss_nor.mean(dim=-1)  # [N, L], mean loss per patch
        # dir_loss_nor = 1 - torch.nn.CosineSimilarity(-1)(pred_nor, target)

        # auxi_loss = torch.mean(self.anomaly_score, dim=1) -  0.1*torch.sum(self.diff_cos, dim=[1, 2]) / (
        #             self.center_num * self.center_num - self.center_num)
        # auxi_loss = torch.mean(self.anomaly_score, dim=1)
        # mse_loss_2 = torch.mean((pred2 - target) ** 2, dim=1)
        #
        # #
        # cos_loss_k1_2 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred2, 1), self.my_patchify(target, 1))
        # cos_loss_k64_2 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred2, 64), self.my_patchify(target, 64))
        loss_g = cos_loss_k1.mean() + cos_loss_k64.mean() + mse_loss.mean()
        # loss_g_2 = cos_loss_k1_2.mean() + cos_loss_k64_2.mean() + mse_loss_2.mean()
        # loss_g = 5 * dir_loss.mean() + dis_loss.mean()
        # loss_g = auxi_loss.mean()
        # print(f'cos: f{5 * dir_loss.mean()} mse:{dis_loss.mean()}, auxi:{auxi_loss.mean()}')
        # loss_g = 5 * dir_loss.mean() + dis_loss.mean()

        return loss_g

    def forward(self, imgs, stage):
        x1 = self.forward_encoder(imgs, stage)

        # pred = self.forward_decoder(latent1)  # [N, L, p*p*3]
        if stage == "train":
            loss = self.forward_loss(imgs, x1)
        else:
            # x1 = self.forward_encoder(imgs, stage)
            loss = 0.

        return loss, x1




class VIT(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=960,
                 embed_dim=768, depth=8, num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super(VIT, self).__init__()
        self.len_keep = 0  # 初始化
        self.in_chans = in_chans

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)


        num_patches = self.patch_embed.num_patches


        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)


        self.inpainting_pred = nn.Linear(embed_dim, patch_size ** 2 * in_chans, bias=True)

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def generate_rand_noise(self, Batch, embed_dim, x):
        num_patches = self.patch_embed.num_patches
        noise_num = int(random.uniform(0, 0.75) * num_patches)
        self.noise_num = noise_num
        noise_index_list = [random.sample(range(num_patches), noise_num) for _ in range(Batch)]
        self.noise_index_list = noise_index_list
        tensor_defect = torch.zeros((Batch, num_patches, embed_dim))
        x_norm = x.norm(dim=2).unsqueeze(-1) / embed_dim
        tensor_defect[torch.arange(Batch).unsqueeze(-1), torch.tensor(noise_index_list, dtype=torch.long), :] = 1
        tensor_defect = tensor_defect.to(x.device)
        noise = torch.randn(Batch, num_patches, embed_dim).to(x.device) * x_norm * 5
        noise = tensor_defect * noise
        # noise = noise.to(device)
        return noise

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        #
        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
        #                                             int(self.patch_embed.num_patches ** .5), cls_token=False)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)

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
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))

        return imgs
    def my_patchify(self, imgs, p):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def my_unpatchify(self, x, p):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs
    # def out_order(self, x):
    #     indices = torch.randperm(x.shape[1])
    #     return x[:, indices]


    def forward_encoder(self, x, stage):
        x = self.patch_embed(x)

        x1 = x + self.pos_embed
        for blk in self.blocks:
            x1 = blk(x1)
        x1 = self.norm(x1)
        x1 = self.inpainting_pred(x1)
        return x1

    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = self.patchify(imgs)
        target = imgs
        pred = self.unpatchify(pred)
        # pred2 = self.unpatchify(pred2)
        # N, L, _ = target.shape
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6) ** .5

        # dis_loss = (pred - target) ** 2 #
        # dis_loss = dis_loss.mean(dim=-1)  # [N, L], mean loss per patch
        mse_loss = torch.mean((pred - target) ** 2, dim=1)

        #
        cos_loss_k1 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 1), self.my_patchify(target, 1))

        cos_loss_k64 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 64), self.my_patchify(target, 64))
        # global_loss = self.global_diff(target, pred)
        # dis_loss_nor = (pred_nor - target) ** 2
        # dis_loss_nor = dis_loss_nor.mean(dim=-1)  # [N, L], mean loss per patch
        # dir_loss_nor = 1 - torch.nn.CosineSimilarity(-1)(pred_nor, target)

        # auxi_loss = torch.mean(self.anomaly_score, dim=1) -  0.1*torch.sum(self.diff_cos, dim=[1, 2]) / (
        #             self.center_num * self.center_num - self.center_num)
        # auxi_loss = torch.mean(self.anomaly_score, dim=1)
        # mse_loss_2 = torch.mean((pred2 - target) ** 2, dim=1)
        #
        # #
        # cos_loss_k1_2 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred2, 1), self.my_patchify(target, 1))
        # cos_loss_k64_2 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred2, 64), self.my_patchify(target, 64))
        loss_g = cos_loss_k1.mean() + cos_loss_k64.mean() + mse_loss.mean()
        # loss_g = mse_loss.mean()
        # loss_g_2 = cos_loss_k1_2.mean() + cos_loss_k64_2.mean() + mse_loss_2.mean()
        # loss_g = 5 * dir_loss.mean() + dis_loss.mean()
        # loss_g = auxi_loss.mean()
        # print(f'cos: f{5 * dir_loss.mean()} mse:{dis_loss.mean()}, auxi:{auxi_loss.mean()}')
        # loss_g = 5 * dir_loss.mean() + dis_loss.mean()

        return loss_g

    def forward(self, imgs, stage):
        pred_mask = self.forward_encoder(imgs, stage)
        # pred = self.forward_decoder(latent1)  # [N, L, p*p*3]
        if stage == "train":
            loss = self.forward_loss(imgs, pred_mask)
        else:
            loss = 0.

        return loss, pred_mask



class VIT_no_skip(nn.Module):
    def __init__(self, img_size=64, patch_size=4, in_chans=960,
                 embed_dim=768, depth=8, num_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super(VIT_no_skip, self).__init__()
        self.len_keep = 0  # 初始化
        self.in_chans = in_chans

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)


        num_patches = self.patch_embed.num_patches


        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block_no_skip(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)


        self.inpainting_pred = nn.Linear(embed_dim, patch_size ** 2 * in_chans, bias=True)

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def generate_rand_noise(self, Batch, embed_dim, x):
        num_patches = self.patch_embed.num_patches
        noise_num = int(random.uniform(0, 0.75) * num_patches)
        self.noise_num = noise_num
        noise_index_list = [random.sample(range(num_patches), noise_num) for _ in range(Batch)]
        self.noise_index_list = noise_index_list
        tensor_defect = torch.zeros((Batch, num_patches, embed_dim))
        x_norm = x.norm(dim=2).unsqueeze(-1) / embed_dim
        tensor_defect[torch.arange(Batch).unsqueeze(-1), torch.tensor(noise_index_list, dtype=torch.long), :] = 1
        tensor_defect = tensor_defect.to(x.device)
        noise = torch.randn(Batch, num_patches, embed_dim).to(x.device) * x_norm * 5
        noise = tensor_defect * noise
        # noise = noise.to(device)
        return noise

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        #
        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
        #                                             int(self.patch_embed.num_patches ** .5), cls_token=False)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)

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
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))

        return imgs
    def my_patchify(self, imgs, p):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        # p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def my_unpatchify(self, x, p):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs
    # def out_order(self, x):
    #     indices = torch.randperm(x.shape[1])
    #     return x[:, indices]


    def forward_encoder(self, x, stage):
        x = self.patch_embed(x)

        x1 = x + self.pos_embed
        for blk in self.blocks:
            x1 = blk(x1)
        x1 = self.norm(x1)
        x1 = self.inpainting_pred(x1)
        return x1

    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        # target = self.patchify(imgs)
        target = imgs
        pred = self.unpatchify(pred)
        # pred2 = self.unpatchify(pred2)
        # N, L, _ = target.shape
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6) ** .5

        # dis_loss = (pred - target) ** 2 #
        # dis_loss = dis_loss.mean(dim=-1)  # [N, L], mean loss per patch
        mse_loss = torch.mean((pred - target) ** 2, dim=1)

        #
        cos_loss_k1 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 1), self.my_patchify(target, 1))

        cos_loss_k64 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred, 64), self.my_patchify(target, 64))
        # global_loss = self.global_diff(target, pred)
        # dis_loss_nor = (pred_nor - target) ** 2
        # dis_loss_nor = dis_loss_nor.mean(dim=-1)  # [N, L], mean loss per patch
        # dir_loss_nor = 1 - torch.nn.CosineSimilarity(-1)(pred_nor, target)

        # auxi_loss = torch.mean(self.anomaly_score, dim=1) -  0.1*torch.sum(self.diff_cos, dim=[1, 2]) / (
        #             self.center_num * self.center_num - self.center_num)
        # auxi_loss = torch.mean(self.anomaly_score, dim=1)
        # mse_loss_2 = torch.mean((pred2 - target) ** 2, dim=1)
        #
        # #
        # cos_loss_k1_2 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred2, 1), self.my_patchify(target, 1))
        # cos_loss_k64_2 = 1 - torch.nn.CosineSimilarity(-1)(self.my_patchify(pred2, 64), self.my_patchify(target, 64))
        loss_g = cos_loss_k1.mean() + cos_loss_k64.mean() + mse_loss.mean()
        # loss_g_2 = cos_loss_k1_2.mean() + cos_loss_k64_2.mean() + mse_loss_2.mean()
        # loss_g = 5 * dir_loss.mean() + dis_loss.mean()
        # loss_g = auxi_loss.mean()
        # print(f'cos: f{5 * dir_loss.mean()} mse:{dis_loss.mean()}, auxi:{auxi_loss.mean()}')
        # loss_g = 5 * dir_loss.mean() + dis_loss.mean()

        return loss_g

    def forward(self, imgs, stage):
        pred_mask = self.forward_encoder(imgs, stage)
        # pred = self.forward_decoder(latent1)  # [N, L, p*p*3]
        if stage == "train":
            loss = self.forward_loss(imgs, pred_mask)
        else:
            loss = 0.

        return loss, pred_mask
