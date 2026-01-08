from models.model_MAE import *
from models.networks import *

from torch import nn
import random
class OO_AE(nn.Module):
    def __init__(self, opt):
        super(OO_AE, self).__init__()

        if opt.backbone_name == 'D_VGG':
            self.Feature_extractor = D_VGG().eval()
            self.Roncon_model = out_order_AE(in_chans=768, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'VGG':
            self.Feature_extractor = VGG().eval()
            self.Roncon_model = out_order_AE(in_chans=960,  patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'Resnet34':
            self.Feature_extractor = Resnet34().eval()
            self.Roncon_model = out_order_AE(in_chans=512, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'Resnet50':
            self.Feature_extractor = Resnet50().eval()
            self.Roncon_model = out_order_AE(in_chans=1536, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'WideResnet50':
            self.Feature_extractor = WideResNet50().eval()
            self.Roncon_model = out_order_AE(in_chans=1792, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'Resnet101':
            self.Feature_extractor = Resnet101().eval()
            self.Roncon_model = out_order_AE(in_chans=1856, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'WideResnet101':
            self.Feature_extractor = WideResnet101().eval()
            self.Roncon_model = out_order_AE(in_chans=1856, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'MobileNet':
            self.Feature_extractor = MobileNet().eval()
            self.Roncon_model = out_order_AE(in_chans=104, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'EfficientNet':
            self.Feature_extractor = EfficientNet().eval()
            self.Roncon_model = out_order_AE(in_chans=272, embed_dim=256, num_heads=8, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)
    def forward(self, imgs, stages):
        deep_feature = self.Feature_extractor(imgs)
        # arti_deep_feature = self.Feature_extractor(arti_imgs)
        loss, pre_feature = self.Roncon_model(deep_feature, stages)
        pre_feature_recon = self.Roncon_model.unpatchify(pre_feature)
        return deep_feature, deep_feature, pre_feature_recon, loss

    def a_map(self, deep_feature, recon_feature):
        # recon_feature = self.Roncon_model.unpatchify(pre_feature)
        batch_size = recon_feature.shape[0]
        mse_map = torch.mean((deep_feature - recon_feature) ** 2, dim=1, keepdim=True)
        mse_map = nn.functional.interpolate(mse_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        mse_map = mse_map.clone().squeeze(0).cpu().detach().numpy()

        cos_mapk1 = 1 - torch.nn.CosineSimilarity()(deep_feature, recon_feature)
        cos_mapk1 = cos_mapk1.reshape(batch_size, 1, 64, 64)
        cos_mapk1 = nn.functional.interpolate(cos_mapk1, size=(256, 256), mode="bilinear", align_corners=True).squeeze(
            1)
        cos_mapk1 = cos_mapk1.clone().squeeze(0).cpu().detach().numpy()

        # cos_mapk2 = 1 - torch.nn.CosineSimilarity(dim=-1)(self.Roncon_model.my_patchify(deep_feature, 2), self.Roncon_model.my_patchify(recon_feature, 2))
        # cos_mapk2 = cos_mapk2.reshape(batch_size, 1, int(64//2), int(64//2))
        # cos_mapk2 = nn.functional.interpolate(cos_mapk2, size=(256, 256), mode="bilinear", align_corners=False).squeeze(
        #     1)
        # cos_mapk2 = cos_mapk2.clone().squeeze(0).cpu().detach().numpy()

        cos_mapk64 = 1 - torch.nn.CosineSimilarity(dim=-1)(self.Roncon_model.my_patchify(deep_feature, 64),
                                                           self.Roncon_model.my_patchify(recon_feature, 64))
        cos_mapk64 = cos_mapk64.reshape((batch_size))
        # cos_mapk64 = cos_mapk64.reshape(batch_size, 1, int(64 // 8), int(64 // 8))
        # cos_mapk64 = nn.functional.interpolate(cos_mapk64, size=(256, 256), mode="bilinear", align_corners=False).squeeze(
        #     1)
        cos_mapk64 = cos_mapk64.clone().cpu().detach().numpy()

        # print(deep_feature.permute(1, 0, 2, 3).shape)
        # ssim_map = torch.mean(ssim(deep_feature.permute(1, 0, 2, 3), recon_feature.permute(1, 0, 2, 3)), dim=0, keepdim=True)
        # # print(ssim_map.shape)
        # ssim_map = nn.functional.interpolate(ssim_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        # ssim_map = ssim_map.clone().squeeze(0).cpu().detach().numpy()
        return cos_mapk1, cos_mapk64, mse_map


class OO_AE_CNN(nn.Module):
    def __init__(self, opt):
        super(OO_AE_CNN, self).__init__()

        if opt.backbone_name == 'D_VGG':
            self.Feature_extractor = D_VGG().eval()
            self.Roncon_model = out_order_AE(in_chans=768, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'VGG':
            self.Feature_extractor = VGG().eval()
            self.Roncon_model = out_order_AE(in_chans=960,  patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'Resnet34':
            self.Feature_extractor = Resnet34().eval()
            self.Roncon_model = out_order_AE(in_chans=512, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'Resnet50':
            self.Feature_extractor = Resnet50().eval()
            self.Roncon_model = out_order_AE(in_chans=1536, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'WideResnet50':
            self.Feature_extractor = WideResNet50().eval()
            self.Roncon_model = out_order_AE_CNN(in_chans=1792, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'Resnet101':
            self.Feature_extractor = Resnet101().eval()
            self.Roncon_model = out_order_AE(in_chans=1856, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'WideResnet101':
            self.Feature_extractor = WideResnet101().eval()
            self.Roncon_model = out_order_AE(in_chans=1856, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'MobileNet':
            self.Feature_extractor = MobileNet().eval()
            self.Roncon_model = out_order_AE(in_chans=104, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'EfficientNet':
            self.Feature_extractor = EfficientNet().eval()
            self.Roncon_model = out_order_AE(in_chans=272, embed_dim=256, num_heads=8, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)
    def forward(self, imgs, stages):
        deep_feature = self.Feature_extractor(imgs)
        # arti_deep_feature = self.Feature_extractor(arti_imgs)
        loss, pre_feature = self.Roncon_model(deep_feature, stages)
        pre_feature_recon = self.Roncon_model.unpatchify(pre_feature)
        return deep_feature, deep_feature, pre_feature_recon, loss

    def a_map(self, deep_feature, recon_feature):
        # recon_feature = self.Roncon_model.unpatchify(pre_feature)
        batch_size = recon_feature.shape[0]
        mse_map = torch.mean((deep_feature - recon_feature) ** 2, dim=1, keepdim=True)
        mse_map = nn.functional.interpolate(mse_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        mse_map = mse_map.clone().squeeze(0).cpu().detach().numpy()

        cos_mapk1 = 1 - torch.nn.CosineSimilarity()(deep_feature, recon_feature)
        cos_mapk1 = cos_mapk1.reshape(batch_size, 1, 64, 64)
        cos_mapk1 = nn.functional.interpolate(cos_mapk1, size=(256, 256), mode="bilinear", align_corners=True).squeeze(
            1)
        cos_mapk1 = cos_mapk1.clone().squeeze(0).cpu().detach().numpy()

        # cos_mapk2 = 1 - torch.nn.CosineSimilarity(dim=-1)(self.Roncon_model.my_patchify(deep_feature, 2), self.Roncon_model.my_patchify(recon_feature, 2))
        # cos_mapk2 = cos_mapk2.reshape(batch_size, 1, int(64//2), int(64//2))
        # cos_mapk2 = nn.functional.interpolate(cos_mapk2, size=(256, 256), mode="bilinear", align_corners=False).squeeze(
        #     1)
        # cos_mapk2 = cos_mapk2.clone().squeeze(0).cpu().detach().numpy()

        cos_mapk64 = 1 - torch.nn.CosineSimilarity(dim=-1)(self.Roncon_model.my_patchify(deep_feature, 64),
                                                           self.Roncon_model.my_patchify(recon_feature, 64))
        cos_mapk64 = cos_mapk64.reshape((batch_size))
        # cos_mapk64 = cos_mapk64.reshape(batch_size, 1, int(64 // 8), int(64 // 8))
        # cos_mapk64 = nn.functional.interpolate(cos_mapk64, size=(256, 256), mode="bilinear", align_corners=False).squeeze(
        #     1)
        cos_mapk64 = cos_mapk64.clone().cpu().detach().numpy()

        # print(deep_feature.permute(1, 0, 2, 3).shape)
        # ssim_map = torch.mean(ssim(deep_feature.permute(1, 0, 2, 3), recon_feature.permute(1, 0, 2, 3)), dim=0, keepdim=True)
        # # print(ssim_map.shape)
        # ssim_map = nn.functional.interpolate(ssim_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        # ssim_map = ssim_map.clone().squeeze(0).cpu().detach().numpy()
        return cos_mapk1, cos_mapk64, mse_map


class OO_AE_without_pos(nn.Module):
    def __init__(self, opt):
        super(OO_AE_without_pos, self).__init__()

        if opt.backbone_name == 'D_VGG':
            self.Feature_extractor = D_VGG().eval()
            self.Roncon_model = out_order_AE(in_chans=768, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'VGG':
            self.Feature_extractor = VGG().eval()
            self.Roncon_model = out_order_AE(in_chans=960,  patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'Resnet34':
            self.Feature_extractor = Resnet34().eval()
            self.Roncon_model = out_order_AE(in_chans=512, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'Resnet50':
            self.Feature_extractor = Resnet50().eval()
            self.Roncon_model = out_order_AE(in_chans=1536, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'WideResnet50':
            self.Feature_extractor = WideResNet50().eval()
            self.Roncon_model = out_order_AE_without_pos(in_chans=1792, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'Resnet101':
            self.Feature_extractor = Resnet101().eval()
            self.Roncon_model = out_order_AE(in_chans=1856, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'WideResnet101':
            self.Feature_extractor = WideResnet101().eval()
            self.Roncon_model = out_order_AE(in_chans=1856, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'MobileNet':
            self.Feature_extractor = MobileNet().eval()
            self.Roncon_model = out_order_AE(in_chans=104, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'EfficientNet':
            self.Feature_extractor = EfficientNet().eval()
            self.Roncon_model = out_order_AE(in_chans=272, embed_dim=256, num_heads=8, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)
    def forward(self, imgs, stages):
        deep_feature = self.Feature_extractor(imgs)
        # arti_deep_feature = self.Feature_extractor(arti_imgs)
        loss, pre_feature = self.Roncon_model(deep_feature, stages)
        pre_feature_recon = self.Roncon_model.unpatchify(pre_feature)
        return deep_feature, deep_feature, pre_feature_recon, loss

    def a_map(self, deep_feature, recon_feature):
        # recon_feature = self.Roncon_model.unpatchify(pre_feature)
        batch_size = recon_feature.shape[0]
        mse_map = torch.mean((deep_feature - recon_feature) ** 2, dim=1, keepdim=True)
        mse_map = nn.functional.interpolate(mse_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        mse_map = mse_map.clone().squeeze(0).cpu().detach().numpy()

        cos_mapk1 = 1 - torch.nn.CosineSimilarity()(deep_feature, recon_feature)
        cos_mapk1 = cos_mapk1.reshape(batch_size, 1, 64, 64)
        cos_mapk1 = nn.functional.interpolate(cos_mapk1, size=(256, 256), mode="bilinear", align_corners=True).squeeze(
            1)
        cos_mapk1 = cos_mapk1.clone().squeeze(0).cpu().detach().numpy()

        # cos_mapk2 = 1 - torch.nn.CosineSimilarity(dim=-1)(self.Roncon_model.my_patchify(deep_feature, 2), self.Roncon_model.my_patchify(recon_feature, 2))
        # cos_mapk2 = cos_mapk2.reshape(batch_size, 1, int(64//2), int(64//2))
        # cos_mapk2 = nn.functional.interpolate(cos_mapk2, size=(256, 256), mode="bilinear", align_corners=False).squeeze(
        #     1)
        # cos_mapk2 = cos_mapk2.clone().squeeze(0).cpu().detach().numpy()

        cos_mapk64 = 1 - torch.nn.CosineSimilarity(dim=-1)(self.Roncon_model.my_patchify(deep_feature, 64),
                                                           self.Roncon_model.my_patchify(recon_feature, 64))
        cos_mapk64 = cos_mapk64.reshape((batch_size))
        # cos_mapk64 = cos_mapk64.reshape(batch_size, 1, int(64 // 8), int(64 // 8))
        # cos_mapk64 = nn.functional.interpolate(cos_mapk64, size=(256, 256), mode="bilinear", align_corners=False).squeeze(
        #     1)
        cos_mapk64 = cos_mapk64.clone().cpu().detach().numpy()

        # print(deep_feature.permute(1, 0, 2, 3).shape)
        # ssim_map = torch.mean(ssim(deep_feature.permute(1, 0, 2, 3), recon_feature.permute(1, 0, 2, 3)), dim=0, keepdim=True)
        # # print(ssim_map.shape)
        # ssim_map = nn.functional.interpolate(ssim_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        # ssim_map = ssim_map.clone().squeeze(0).cpu().detach().numpy()
        return cos_mapk1, cos_mapk64, mse_map


class OO_AE_learned_pos(nn.Module):
    def __init__(self, opt):
        super(OO_AE_learned_pos, self).__init__()

        if opt.backbone_name == 'D_VGG':
            self.Feature_extractor = D_VGG().eval()
            self.Roncon_model = out_order_AE(in_chans=768, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'VGG':
            self.Feature_extractor = VGG().eval()
            self.Roncon_model = out_order_AE(in_chans=960,  patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'Resnet34':
            self.Feature_extractor = Resnet34().eval()
            self.Roncon_model = out_order_AE(in_chans=512, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'Resnet50':
            self.Feature_extractor = Resnet50().eval()
            self.Roncon_model = out_order_AE(in_chans=1536, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'WideResnet50':
            self.Feature_extractor = WideResNet50().eval()
            self.Roncon_model = out_order_AE_learned_pos(in_chans=1792, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'Resnet101':
            self.Feature_extractor = Resnet101().eval()
            self.Roncon_model = out_order_AE(in_chans=1856, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'WideResnet101':
            self.Feature_extractor = WideResnet101().eval()
            self.Roncon_model = out_order_AE(in_chans=1856, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'MobileNet':
            self.Feature_extractor = MobileNet().eval()
            self.Roncon_model = out_order_AE(in_chans=104, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'EfficientNet':
            self.Feature_extractor = EfficientNet().eval()
            self.Roncon_model = out_order_AE(in_chans=272, embed_dim=256, num_heads=8, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)
    def forward(self, imgs, stages):
        deep_feature = self.Feature_extractor(imgs)
        # arti_deep_feature = self.Feature_extractor(arti_imgs)
        loss, pre_feature = self.Roncon_model(deep_feature, stages)
        pre_feature_recon = self.Roncon_model.unpatchify(pre_feature)
        return deep_feature, deep_feature, pre_feature_recon, loss

    def a_map(self, deep_feature, recon_feature):
        # recon_feature = self.Roncon_model.unpatchify(pre_feature)
        batch_size = recon_feature.shape[0]
        mse_map = torch.mean((deep_feature - recon_feature) ** 2, dim=1, keepdim=True)
        mse_map = nn.functional.interpolate(mse_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        mse_map = mse_map.clone().squeeze(0).cpu().detach().numpy()

        cos_mapk1 = 1 - torch.nn.CosineSimilarity()(deep_feature, recon_feature)
        cos_mapk1 = cos_mapk1.reshape(batch_size, 1, 64, 64)
        cos_mapk1 = nn.functional.interpolate(cos_mapk1, size=(256, 256), mode="bilinear", align_corners=True).squeeze(
            1)
        cos_mapk1 = cos_mapk1.clone().squeeze(0).cpu().detach().numpy()

        # cos_mapk2 = 1 - torch.nn.CosineSimilarity(dim=-1)(self.Roncon_model.my_patchify(deep_feature, 2), self.Roncon_model.my_patchify(recon_feature, 2))
        # cos_mapk2 = cos_mapk2.reshape(batch_size, 1, int(64//2), int(64//2))
        # cos_mapk2 = nn.functional.interpolate(cos_mapk2, size=(256, 256), mode="bilinear", align_corners=False).squeeze(
        #     1)
        # cos_mapk2 = cos_mapk2.clone().squeeze(0).cpu().detach().numpy()

        cos_mapk64 = 1 - torch.nn.CosineSimilarity(dim=-1)(self.Roncon_model.my_patchify(deep_feature, 64),
                                                           self.Roncon_model.my_patchify(recon_feature, 64))
        cos_mapk64 = cos_mapk64.reshape((batch_size))
        # cos_mapk64 = cos_mapk64.reshape(batch_size, 1, int(64 // 8), int(64 // 8))
        # cos_mapk64 = nn.functional.interpolate(cos_mapk64, size=(256, 256), mode="bilinear", align_corners=False).squeeze(
        #     1)
        cos_mapk64 = cos_mapk64.clone().cpu().detach().numpy()

        # print(deep_feature.permute(1, 0, 2, 3).shape)
        # ssim_map = torch.mean(ssim(deep_feature.permute(1, 0, 2, 3), recon_feature.permute(1, 0, 2, 3)), dim=0, keepdim=True)
        # # print(ssim_map.shape)
        # ssim_map = nn.functional.interpolate(ssim_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        # ssim_map = ssim_map.clone().squeeze(0).cpu().detach().numpy()
        return cos_mapk1, cos_mapk64, mse_map


class OO_AE_without_jitter(nn.Module):
    def __init__(self, opt):
        super(OO_AE_without_jitter, self).__init__()

        if opt.backbone_name == 'D_VGG':
            self.Feature_extractor = D_VGG().eval()
            self.Roncon_model = out_order_AE(in_chans=768, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'VGG':
            self.Feature_extractor = VGG().eval()
            self.Roncon_model = out_order_AE(in_chans=960,  patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'Resnet34':
            self.Feature_extractor = Resnet34().eval()
            self.Roncon_model = out_order_AE(in_chans=512, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'Resnet50':
            self.Feature_extractor = Resnet50().eval()
            self.Roncon_model = out_order_AE(in_chans=1536, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'WideResnet50':
            self.Feature_extractor = WideResNet50().eval()
            self.Roncon_model = out_order_AE_without_jitter(in_chans=1792, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'Resnet101':
            self.Feature_extractor = Resnet101().eval()
            self.Roncon_model = out_order_AE(in_chans=1856, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'WideResnet101':
            self.Feature_extractor = WideResnet101().eval()
            self.Roncon_model = out_order_AE(in_chans=1856, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'MobileNet':
            self.Feature_extractor = MobileNet().eval()
            self.Roncon_model = out_order_AE(in_chans=104, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)

        if opt.backbone_name == 'EfficientNet':
            self.Feature_extractor = EfficientNet().eval()
            self.Roncon_model = out_order_AE(in_chans=272, embed_dim=256, num_heads=8, patch_size=opt.k, depth=opt.in_lay_num, shuffle_ratio=opt.shuffle_ratio)
    def forward(self, imgs, stages):
        deep_feature = self.Feature_extractor(imgs)
        # arti_deep_feature = self.Feature_extractor(arti_imgs)
        loss, pre_feature = self.Roncon_model(deep_feature, stages)
        pre_feature_recon = self.Roncon_model.unpatchify(pre_feature)
        return deep_feature, deep_feature, pre_feature_recon, loss

    def a_map(self, deep_feature, recon_feature):
        # recon_feature = self.Roncon_model.unpatchify(pre_feature)
        batch_size = recon_feature.shape[0]
        mse_map = torch.mean((deep_feature - recon_feature) ** 2, dim=1, keepdim=True)
        mse_map = nn.functional.interpolate(mse_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        mse_map = mse_map.clone().squeeze(0).cpu().detach().numpy()

        cos_mapk1 = 1 - torch.nn.CosineSimilarity()(deep_feature, recon_feature)
        cos_mapk1 = cos_mapk1.reshape(batch_size, 1, 64, 64)
        cos_mapk1 = nn.functional.interpolate(cos_mapk1, size=(256, 256), mode="bilinear", align_corners=True).squeeze(
            1)
        cos_mapk1 = cos_mapk1.clone().squeeze(0).cpu().detach().numpy()

        # cos_mapk2 = 1 - torch.nn.CosineSimilarity(dim=-1)(self.Roncon_model.my_patchify(deep_feature, 2), self.Roncon_model.my_patchify(recon_feature, 2))
        # cos_mapk2 = cos_mapk2.reshape(batch_size, 1, int(64//2), int(64//2))
        # cos_mapk2 = nn.functional.interpolate(cos_mapk2, size=(256, 256), mode="bilinear", align_corners=False).squeeze(
        #     1)
        # cos_mapk2 = cos_mapk2.clone().squeeze(0).cpu().detach().numpy()

        cos_mapk64 = 1 - torch.nn.CosineSimilarity(dim=-1)(self.Roncon_model.my_patchify(deep_feature, 64),
                                                           self.Roncon_model.my_patchify(recon_feature, 64))
        cos_mapk64 = cos_mapk64.reshape((batch_size))
        # cos_mapk64 = cos_mapk64.reshape(batch_size, 1, int(64 // 8), int(64 // 8))
        # cos_mapk64 = nn.functional.interpolate(cos_mapk64, size=(256, 256), mode="bilinear", align_corners=False).squeeze(
        #     1)
        cos_mapk64 = cos_mapk64.clone().cpu().detach().numpy()

        # print(deep_feature.permute(1, 0, 2, 3).shape)
        # ssim_map = torch.mean(ssim(deep_feature.permute(1, 0, 2, 3), recon_feature.permute(1, 0, 2, 3)), dim=0, keepdim=True)
        # # print(ssim_map.shape)
        # ssim_map = nn.functional.interpolate(ssim_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        # ssim_map = ssim_map.clone().squeeze(0).cpu().detach().numpy()
        return cos_mapk1, cos_mapk64, mse_map


class VIT_dir(nn.Module):
    def __init__(self, opt):
        super(VIT_dir, self).__init__()

        if opt.backbone_name == 'D_VGG':
            self.Feature_extractor = D_VGG().eval()
            self.Roncon_model = VIT(in_chans=768, patch_size=opt.k, depth=opt.in_lay_num)

        if opt.backbone_name == 'VGG':
            self.Feature_extractor = VGG().eval()
            self.Roncon_model = VIT(in_chans=960,  patch_size=opt.k, depth=opt.in_lay_num)

        if opt.backbone_name == 'Resnet34':
            self.Feature_extractor = Resnet34().eval()
            self.Roncon_model = VIT(in_chans=512, patch_size=opt.k, depth=opt.in_lay_num)

        if opt.backbone_name == 'Resnet50':
            self.Feature_extractor = Resnet50().eval()
            self.Roncon_model = VIT(in_chans=1536, patch_size=opt.k, depth=opt.in_lay_num)

        if opt.backbone_name == 'WideResnet50':
            self.Feature_extractor = WideResNet50().eval()
            self.Roncon_model = VIT(in_chans=1792, patch_size=opt.k, depth=opt.in_lay_num)

        if opt.backbone_name == 'Resnet101':
            self.Feature_extractor = Resnet101().eval()
            self.Roncon_model = VIT(in_chans=1856, patch_size=opt.k, depth=opt.in_lay_num)

        if opt.backbone_name == 'WideResnet101':
            self.Feature_extractor = WideResnet101().eval()
            self.Roncon_model = VIT(in_chans=1856, patch_size=opt.k, depth=opt.in_lay_num)

        if opt.backbone_name == 'MobileNet':
            self.Feature_extractor = MobileNet().eval()
            self.Roncon_model = VIT(in_chans=104, patch_size=opt.k, depth=opt.in_lay_num)

        if opt.backbone_name == 'EfficientNet':
            self.Feature_extractor = EfficientNet().eval()
            self.Roncon_model = VIT(in_chans=272, embed_dim=256, num_heads=8, patch_size=opt.k, depth=opt.in_lay_num)

    def forward(self, imgs, stages):
        deep_feature = self.Feature_extractor(imgs)
        # arti_deep_feature = self.Feature_extractor(arti_imgs)
        loss, pre_feature = self.Roncon_model(deep_feature, stages)
        pre_feature_recon = self.Roncon_model.unpatchify(pre_feature)
        return deep_feature, deep_feature, pre_feature_recon, loss

    def a_map(self, deep_feature, recon_feature):
        # recon_feature = self.Roncon_model.unpatchify(pre_feature)
        batch_size = recon_feature.shape[0]
        mse_map = torch.mean((deep_feature - recon_feature) ** 2, dim=1, keepdim=True)
        mse_map = nn.functional.interpolate(mse_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        mse_map = mse_map.clone().squeeze(0).cpu().detach().numpy()

        cos_mapk1 = 1 - torch.nn.CosineSimilarity()(deep_feature, recon_feature)
        cos_mapk1 = cos_mapk1.reshape(batch_size, 1, 64, 64)
        cos_mapk1 = nn.functional.interpolate(cos_mapk1, size=(256, 256), mode="bilinear", align_corners=True).squeeze(
            1)
        cos_mapk1 = cos_mapk1.clone().squeeze(0).cpu().detach().numpy()



        cos_mapk64 = 1 - torch.nn.CosineSimilarity(dim=-1)(self.Roncon_model.my_patchify(deep_feature, 64),
                                                           self.Roncon_model.my_patchify(recon_feature, 64))
        cos_mapk64 = cos_mapk64.reshape((batch_size))

        cos_mapk64 = cos_mapk64.clone().cpu().detach().numpy()


        return cos_mapk1, cos_mapk64, mse_map


class VIT_dir_noskip(nn.Module):
    def __init__(self, opt):
        super(VIT_dir_noskip, self).__init__()

        if opt.backbone_name == 'D_VGG':
            self.Feature_extractor = D_VGG().eval()
            self.Roncon_model = VIT_no_skip(in_chans=768, patch_size=opt.k, depth=opt.in_lay_num)

        if opt.backbone_name == 'VGG':
            self.Feature_extractor = VGG().eval()
            self.Roncon_model = VIT_no_skip(in_chans=960,  patch_size=opt.k, depth=opt.in_lay_num)

        if opt.backbone_name == 'Resnet34':
            self.Feature_extractor = Resnet34().eval()
            self.Roncon_model = VIT_no_skip(in_chans=512, patch_size=opt.k, depth=opt.in_lay_num)

        if opt.backbone_name == 'Resnet50':
            self.Feature_extractor = Resnet50().eval()
            self.Roncon_model = VIT_no_skip(in_chans=1536, patch_size=opt.k, depth=opt.in_lay_num)

        if opt.backbone_name == 'WideResnet50':
            self.Feature_extractor = WideResNet50().eval()
            self.Roncon_model = VIT_no_skip(in_chans=1536, patch_size=opt.k, depth=opt.in_lay_num)

        if opt.backbone_name == 'Resnet101':
            self.Feature_extractor = Resnet101().eval()
            self.Roncon_model = VIT_no_skip(in_chans=1856, patch_size=opt.k, depth=opt.in_lay_num)

        if opt.backbone_name == 'WideResnet101':
            self.Feature_extractor = WideResnet101().eval()
            self.Roncon_model = VIT_no_skip(in_chans=1856, patch_size=opt.k, depth=opt.in_lay_num)

        if opt.backbone_name == 'MobileNet':
            self.Feature_extractor = MobileNet().eval()
            self.Roncon_model = VIT_no_skip(in_chans=104, patch_size=opt.k, depth=opt.in_lay_num)

    def forward(self, imgs, stages):
        deep_feature = self.Feature_extractor(imgs)
        # arti_deep_feature = self.Feature_extractor(arti_imgs)
        loss, pre_feature = self.Roncon_model(deep_feature, stages)
        pre_feature_recon = self.Roncon_model.unpatchify(pre_feature)
        return deep_feature, deep_feature, pre_feature_recon, loss

    def a_map(self, deep_feature, recon_feature):
        # recon_feature = self.Roncon_model.unpatchify(pre_feature)
        batch_size = recon_feature.shape[0]
        mse_map = torch.mean((deep_feature - recon_feature) ** 2, dim=1, keepdim=True)
        mse_map = nn.functional.interpolate(mse_map, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        mse_map = mse_map.clone().squeeze(0).cpu().detach().numpy()

        cos_mapk1 = 1 - torch.nn.CosineSimilarity()(deep_feature, recon_feature)
        cos_mapk1 = cos_mapk1.reshape(batch_size, 1, 64, 64)
        cos_mapk1 = nn.functional.interpolate(cos_mapk1, size=(256, 256), mode="bilinear", align_corners=True).squeeze(
            1)
        cos_mapk1 = cos_mapk1.clone().squeeze(0).cpu().detach().numpy()



        cos_mapk64 = 1 - torch.nn.CosineSimilarity(dim=-1)(self.Roncon_model.my_patchify(deep_feature, 64),
                                                           self.Roncon_model.my_patchify(recon_feature, 64))
        cos_mapk64 = cos_mapk64.reshape((batch_size))

        cos_mapk64 = cos_mapk64.clone().cpu().detach().numpy()


        return cos_mapk1, cos_mapk64, mse_map
