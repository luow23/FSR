from visualizer import get_local
get_local.activate()

# from models.RB_VIT import RB_VIT
import torch
from config import DefaultConfig
import os
from torch import optim

import torch.nn.functional as F
from tqdm import tqdm

from matplotlib import pyplot as plt
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from utils.metric import cal_pro_metric_new
import cv2
from models.misc import NativeScalerWithGradNormCount as NativeScaler

from scipy.ndimage import gaussian_filter
from datasets.dataset import denormalize
# from ref_find import get_pos_sample
from models.RB_VIT import *
from matplotlib.patches import ConnectionPatch
import math
class Model(object):
    def __init__(self, opt, test_no_mask, test_add):
        super(Model, self).__init__()
        # if test_no_mask == True:
        #     model_name = 'RB_VIT_dir_res_ref'
        # else:
        #     model_name = opt.model_name
        self.opt = opt
        self.model = eval(opt.model_name)(opt)
        self.device = opt.device
        self.test_add = test_add
        self.class_name = opt.class_name
        self.trainloader = opt.trainloader
        self.testloader = opt.testloader
        self.loss_scaler = NativeScaler()

        if self.opt.resume != "":
            print('\nload pre-trained networks')
            self.opt.iter = \
                torch.load(os.path.join(self.opt.resume, self.opt.class_name, f'{opt.model_name}.pth'))[
                    'epoch']
            print(self.opt.iter)
            self.model.load_state_dict(torch.load(
                os.path.join(self.opt.resume, self.opt.class_name, f'{opt.model_name}.pth'))[
                                           'state_dict'], strict=False)
            print('\ndone.\n')

        if self.opt.isTrain:
            self.model.Roncon_model.train()
            self.optimizer_g = optim.AdamW(self.model.Roncon_model.parameters(), lr=opt.lr, betas=(0.9, 0.95))
        # if test_no_mask == True:
        #     self.save_root = f"./attention_map_test/{opt.model_name}_{opt.backbone_name}"
        # else:
        if self.opt.model_name == "VIT_dir":
            self.save_root = f"./attention_map_test/{opt.model_name}_{opt.backbone_name}_k={opt.k}_inpaint_num={opt.in_lay_num}"
        else:
            self.save_root = f"./attention_map_test/{opt.model_name}_{opt.backbone_name}_k={opt.k}_inpaint_num={opt.in_lay_num}_shuffle_ratio={opt.shuffle_ratio}"

        # if test_add == True:
        #     self.save_root += 'add'
        # os.makedirs(os.path.join(self.save_root, "weight"), exist_ok=True)
        self.ckpt_root = os.path.join(self.save_root, "weight/{}".format(self.class_name))
        self.vis_root = os.path.join(self.save_root, "img/{}".format(self.class_name))



    def get_max(self, tensor):
        a_1, _ = torch.max(tensor, dim=1, keepdim=True)
        a_2, _ = torch.max(a_1, dim=2, keepdim=True)
        a_3, _ = torch.max(a_2, dim=3, keepdim=True)
        return a_3

    def cal_auc(self, score_list, score_map_list, test_y_list, test_mask_list):
        flatten_y_list = np.array(test_y_list).ravel()
        flatten_score_list = np.array(score_list).ravel()
        image_level_ROCAUC = roc_auc_score(flatten_y_list, flatten_score_list)
        image_level_AP = average_precision_score(flatten_y_list, flatten_score_list)

        flatten_mask_list = np.concatenate(test_mask_list).ravel()
        flatten_score_map_list = np.concatenate(score_map_list).ravel()
        pixel_level_ROCAUC = roc_auc_score(flatten_mask_list, flatten_score_map_list)
        pixel_level_AP = average_precision_score(flatten_mask_list, flatten_score_map_list)
        # pro_auc_score = 0
        # pro_auc_score = cal_pro_metric_new(test_mask_list, score_map_list, fpr_thresh=0.3)
        return round(image_level_ROCAUC,3), round(pixel_level_ROCAUC,3), round(image_level_AP,3), round(pixel_level_AP,3)
        # return  image_level_ROCAUC, pixel_level_ROCAUC

    def F1_score(self, score_map_list, test_mask_list):
        flatten_mask_list = np.concatenate(test_mask_list).ravel()
        flatten_score_map_list = np.concatenate(score_map_list).ravel()
        F1_score = f1_score(flatten_mask_list, flatten_score_map_list)
        return F1_score

    def filter(self, pred_mask):
        pred_mask_my = np.squeeze(np.squeeze(pred_mask, 0), 0)
        pred_mask_my = cv2.medianBlur(np.uint8(pred_mask_my * 255), 7)
        mean = np.mean(pred_mask_my)
        std = np.std(pred_mask_my)
        _ , binary_pred_mask = cv2.threshold(pred_mask_my, mean+2.75*std, 255, type=cv2.THRESH_BINARY)
        binary_pred_mask = np.uint8(binary_pred_mask/255)
        pred_mask_my = np.expand_dims(np.expand_dims(pred_mask_my, 0), 0)
        binary_pred_mask = np.expand_dims(np.expand_dims(binary_pred_mask, 0), 0)
        return pred_mask_my, binary_pred_mask


    # def thresholding(self, pred_mask_my):
    #     np_img

        # return
    def feature_map_vis(self, feature_map_list):
        # feature_map_list = [torch.mean(i.clone(), dim=1).squeeze(0).cpu().detach().numpy() for i in feature_map_list]
        feature_map_list = [(i.squeeze(0))[25, :, :].cpu().detach().numpy() for i in feature_map_list]
        return feature_map_list

    def test(self):
        test_y_list = []
        test_mask_list = []
        score_list = []
        score_map_list = []

        for idx, (x, y, mask, name) in enumerate(tqdm(self.testloader, ncols=80)):
            get_local.clear()
            test_y_list.extend(y.detach().cpu().numpy())
            test_mask_list.extend(mask.detach().cpu().numpy())
            self.model.eval()
            self.model.to(self.device)
            # self.model.Roncon_model.inpainting_pred
            x = x.to(self.device)
            mask = mask.to(self.device)
            mask_cpu = mask.cpu().detach().numpy()[0, :, :, :].transpose((1, 2, 0))
            # ref_x = get_pos_sample(self.opt.referenc_img_file, self.device, 1)
            deep_feature, ref_feature, recon_feature, _ = self.model(x, 'test')
            feature_map_vis_list = self.feature_map_vis([deep_feature, ref_feature, recon_feature])
            # dis_amap, dir_amap = self.model.a_map(deep_feature, recon_feature)
            # dis_amap = gaussian_filter(dis_amap, sigma=4)
            # dir_amap = gaussian_filter(dir_amap, sigma=4)
            cache = get_local.cache
            attention_maps = cache['Attention.forward']
            print(len(attention_maps))
            # print(type(attention_maps))
            # print(attention_maps[0].shape)
            # torch_attention = torch.tensor(attention_maps, device=self.opt.device)
            # # print(torch_attention.shape)
            # torch_attention = torch_attention.transpose(1, 2).reshape(8, 256, -1)
            # # print(torch_attention.shape)
            # vis_value = self.model.Roncon_model.inpainting_pred(self.model.Roncon_model.norm(torch_attention))
            # vis_value = self.model.Roncon_model.unpatchify(vis_value)
            # # print(vis_value.shape)
            # vis_value = vis_value.detach().cpu().numpy()
            # vis_value = vis_value[-1]
            # vis_value = np.mean(vis_value, axis=0)
            # print(vis_value.shape)
            # attention_q =
            # print(len(attention_maps))
            # print(type(attention_maps[0]))
            # print(attention_maps[12].shape)
            # print(type(name0]))
            name_list= name[0].split(r'!')
            # print(name_list)
            category, img_name = name_list[-2], name_list[-1]
            # if self.test_add == False:
            #     # amap = dir_amap*dis_amap
            #     amap = dir_amap*5+dis_amap
            # else:
            #     # print('ok')
            #     # print(np.max(dir_amap))
            #     # print(np.max(dis_amap))
            #     amap = 0.5*(dir_amap/np.max(dir_amap)) + 0.5*(dis_amap/np.max(dis_amap))
            self.new_vis_attention_map([x, attention_maps, x, mask_cpu], os.path.join(self.vis_root, category), img_name)
            # self.new_new_vis_attention_map([x, attention_maps], os.path.join(self.vis_root, category), img_name)

        #     score_list.extend(np.array(np.std(amap)).reshape(1))
        #     score_map_list.extend(amap.reshape((1, 1, 256, 256)))
        #
        #
        # image_level_ROCAUC, pixel_level_ROCAUC, image_level_AP, pixel_level_AP= self.cal_auc(score_list, score_map_list, test_y_list, test_mask_list)
        # # F1_score = self.F1_score(F1_score_map_list, test_mask_list)
        # print('image_auc_roc: {} '.format(image_level_ROCAUC),
        #       'pixel_auc_roc: {} '.format(pixel_level_ROCAUC),
        #       'image_AP: {}'.format(image_level_AP),
        #       'pixel_AP: {}'.format(pixel_level_AP)
        #      )
        # class_rocauc[self.opt.class_name] = (image_level_ROCAUC, pixel_level_ROCAUC, image_level_AP, pixel_level_AP)
        # return image_level_ROCAUC, pixel_level_ROCAUC, image_level_AP, pixel_level_AP
    def vis_attention_map(self, img_list, save_root, idx_name):
        os.makedirs(save_root, exist_ok=True)
        input_frame = denormalize(img_list[0].clone().squeeze(0).cpu().detach().numpy())
        cv2_input = np.array(input_frame, dtype=np.uint8)
        plt.figure()
        plt.subplot(131)
        plt.imshow(cv2_input)
        plt.axis('off')
        p2 = plt.subplot(132)
        attention = img_list[1]
        attention = attention[:12]
        attention = [i[:, :, 257:, 1:257] for i in attention]
        attention = [np.mean(i, axis=1) for i in attention]
        attention = sum(attention)/len(attention)
        # print(attention.shape)
        attention = attention.squeeze(0)
        # attention = attention.transpose(1, 0)
        # L = attention.shape[0]
        # H = W = int(math.sqrt(L))
        # attention = attention.reshape(H, W, H, W).cpu().numpy()
        tx0 = 50
        tx1 = 100
        ty0 = 50
        ty1 = 100
        sx = [tx0, tx1, tx1, tx0, tx0]
        sy = [ty0, ty0, ty1, ty1, ty0]
        p2.plot(sx, sy, "red", linewidth=1)
        plt.imshow(attention)
        plt.axis('off')
        p3 = plt.subplot(133)
        p3.imshow(attention[ty0:ty1+1, tx0:tx1+1])
        p3.axis('off')
        xy = (100, 50)
        xy2 = (0, 0)
        con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                              axesA=p3, axesB=p2, linestyle='--', color='red')
        p3.add_artist(con)
        xy = (100, 100)
        xy2 = (0, 50)
        con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                              axesA=p3, axesB=p2, linestyle='--', color='red')
        p3.add_artist(con)
        plt.savefig(os.path.join(save_root, idx_name))
        plt.close()



    def new_vis_attention_map(self, img_list, save_root, idx_name):
        os.makedirs(save_root, exist_ok=True)
        input_frame = denormalize(img_list[0].clone().squeeze(0).cpu().detach().numpy())
        cv2_input = np.array(input_frame, dtype=np.uint8)
        attention = img_list[1]
        # v = img_list[]
        # attention = attention[:8]
        # attention = [i[:, :, 257:, 1:257] for i in attention]
        # attention = attention[7]
        # attention = attention[0, 0, :, :]
        # attention = attention[0, 0, :, :]
        print(len(attention))
        attention = [np.mean(i, axis=1) for i in attention]
        attention = sum(attention)/len(attention)
        # attention = attention[-1]
        # print(attention.shape)
        # attention = attention[0, 1]
        # print(attention.shape)
        # print(attention.shape)
        # print(attention.shape)
        attention = attention.squeeze(0)
        # attention = attention.transpose(1, 0)
        L = attention.shape[0]
        H = W = int(math.sqrt(L))
        attention = attention.reshape(H, W, H, W)
        self.visualize_correlations(attention, cv2_input, save_root, idx_name)

    def q_vis_attention_map(self, img_list, save_root, idx_name):
        os.makedirs(save_root, exist_ok=True)
        input_frame = denormalize(img_list[0].clone().squeeze(0).cpu().detach().numpy())
        cv2_input = np.array(input_frame, dtype=np.uint8)
        ref_input = np.array(img_list[2].clone().squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)*255, dtype=np.uint8)
        # mask = np.array()
        # attention = img_list[1]
        # print(attention.shape)
        # print(len(attention))
        # attention = attention[:8]
        # attention = [i[:, :, :,] for i in attention]
        # attention = [np.mean(i, axis=1) for i in attention]
        # attention = sum(attention)/len(attention)
        # print(attention.shape)
        # print(attention.shape)
        # attention = attention.squeeze(0)
        # attention = np.mean(attention, axis=-1)
        # attention = attention[:, -1]
        # attention = attention.transpose(1, 0)
        # L = attention.shape[0]
        # H = W = int(math.sqrt(L))
        # attention = np.max(attention, axis=1)
        # attention = attention.reshape(H, W)
        plt.figure(figsize=(4, 1))
        ax1 = plt.subplot(141)
        plt.imshow(cv2_input)
        # plt.axis('off')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        ax.spines['top'].set_linewidth(0.5)
        ax2 = plt.subplot(142)
        plt.imshow(ref_input)
        # plt.axis('off')
        # fig.set_xticks([])
        # plt.tick_params(width=0.1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        ax.spines['top'].set_linewidth(0.5)
        plt.subplot(143)
        plt.imshow(img_list[1])
        # plt.imshow(attention, cmap='jet')
        plt.axis('off')
        plt.subplot(144)
        plt.imshow(img_list[3], cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=0.99, left=0.01, hspace=0, wspace=0.1)
        plt.margins(0, 0)
        plt.savefig(os.path.join(save_root, idx_name), dpi=1000)
        plt.cla()
        plt.close()


        # self.visualize_correlations(attention, cv2_input, save_root, idx_name)
    def new_new_vis_attention_map(self, img_list, save_root, idx_name):
        os.makedirs(save_root, exist_ok=True)
        input_frame = denormalize(img_list[0].clone().squeeze(0).cpu().detach().numpy())
        cv2_input = np.array(input_frame, dtype=np.uint8)
        ref_input = np.array(img_list[2].clone().squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)*255, dtype=np.uint8)
        # mask = np.array()
        attention = img_list[1]
        attention = attention[:12]
        attention = [i[:, :, 257:, 1:257] for i in attention]
        attention = [np.mean(i, axis=1) for i in attention]
        attention = sum(attention)/len(attention)
        # print(attention.shape)
        attention = attention.squeeze(0)
        attention = attention.transpose(1, 0)
        L = attention.shape[0]
        H = W = int(math.sqrt(L))
        attention = np.max(attention, axis=1)
        attention = attention.reshape(H, W)
        plt.figure(figsize=(4, 1))
        ax1 = plt.subplot(141)
        plt.imshow(cv2_input)
        # plt.axis('off')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        ax.spines['top'].set_linewidth(0.5)
        ax2 = plt.subplot(142)
        plt.imshow(ref_input)
        # plt.axis('off')
        # fig.set_xticks([])
        # plt.tick_params(width=0.1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        ax.spines['top'].set_linewidth(0.5)
        plt.subplot(143)
        plt.imshow(attention, cmap='plasma')
        # plt.imshow(attention, cmap='jet')
        plt.axis('off')
        plt.subplot(144)
        plt.imshow(img_list[3], cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=0.99, left=0.01, hspace=0, wspace=0.1)
        plt.margins(0, 0)
        plt.savefig(os.path.join(save_root, idx_name), dpi=1000)
        plt.cla()
        plt.close()


        # self.visualize_correlations(attention, cv2_input, save_root, idx_name)

    def visualize_correlations(self, attn, img, save_root, idx_name):
        factor = 16
        # let's select 4 reference points for visualization
        idxs = [(60, 60), (110, 110), (130, 130), (160, 160)]

        fig = plt.figure(constrained_layout=True, figsize=(25 * 0.5, 8.5 * 0.5))
        # and we add one plot per reference point
        gs = fig.add_gridspec(2, 4)
        # print(gs)
        axs = [fig.add_subplot(gs[0, 0]),
               fig.add_subplot(gs[1, 0]),
               fig.add_subplot(gs[0, 3]),
               fig.add_subplot(gs[1, 3])]
        # print(img.shape)
        # print(attn[60//16, 60//16, ...].shape)
        for idx_o, ax in zip(idxs, axs):
            idx = (idx_o[0] // factor, idx_o[1] // factor)
            x = ((idx_o[1] // factor) + 0.5) * factor - 1
            y = ((idx_o[0] // factor) + 0.5) * factor - 1
            ax.imshow(img)
            # ax.imshow(attn[idx[0], idx[1], ...], cmap='cividis', interpolation='nearest', alpha=0.5)
            ax.imshow(cv2.resize(attn[idx[0], idx[1], ...], (256, 256), interpolation=cv2.INTER_AREA), cmap='cividis', interpolation='nearest', alpha=0.8)
            # cv2.resize(img_list[3], (256, 256), ), alpha = 0.5, cmap = 'jet'
            ax.add_patch(plt.Circle((x , y ), factor // 16, color='r'))
            ax.add_patch(
                plt.Rectangle((x  - factor // 2, y - factor // 2), factor, factor, linewidth=1,
                              edgecolor='r', facecolor='none'))
            ax.axis('off')
            ax.set_title(f'Attention Map {idx_o}')

        fcenter_ax = fig.add_subplot(gs[:, 1:-1])
        fcenter_ax.imshow(img)
        for (y, x) in idxs:
            scale = img.shape[0] / img.shape[0]
            x = ((x // factor) + 0.5) * factor-1
            y = ((y // factor) + 0.5) * factor-1
            fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), factor // 16, color='r'))
            fcenter_ax.add_patch(plt.Rectangle((x * scale-factor//2, y * scale-factor//2), factor, factor,linewidth=1, edgecolor='r', facecolor='none'))
            # fcenter_ax.axis('off')
        os.makedirs(save_root, exist_ok=True)
        fig.savefig(os.path.join(save_root, idx_name))
        plt.cla()
        plt.clf()
        plt.close()

    def vis_img(self, img_list, save_root, idx_name):
        os.makedirs(save_root, exist_ok=True)
        input_frame = denormalize(img_list[0].clone().squeeze(0).cpu().detach().numpy())
        cv2_input = np.array(input_frame, dtype=np.uint8)
        plt.figure()
        plt.subplot(241)
        plt.imshow(cv2_input)
        plt.axis('off')
        plt.subplot(242)
        plt.imshow(img_list[1])
        plt.axis('off')
        plt.subplot(243)
        plt.imshow(img_list[2])
        plt.axis('off')
        plt.subplot(244)
        plt.imshow(img_list[3])
        plt.axis('off')
        plt.subplot(245)
        plt.imshow(img_list[4], cmap='jet')
        plt.axis('off')
        plt.subplot(246)
        plt.imshow(img_list[5],cmap='jet')
        plt.axis('off')
        plt.subplot(247)
        plt.imshow(img_list[6], cmap='jet')
        plt.axis('off')
        plt.subplot(248)
        plt.imshow(img_list[7])
        plt.axis('off')
        plt.savefig(os.path.join(save_root, idx_name))
        plt.close()

    def save_img(self, img_list, save_root, idx_name):
        os.makedirs(save_root, exist_ok=True)
        input_frame = denormalize(img_list[0].clone().squeeze(0).cpu().detach().numpy())
        cv2_input = np.array(input_frame, dtype=np.uint8)
        # plt.figure()
        # plt.subplot(241)
        plt.imsave(os.path.join(save_root, f'{idx_name}_{0}.png'), cv2_input)
        plt.imsave(os.path.join(save_root, f'{idx_name}_{1}.png'), img_list[1])
        plt.imsave(os.path.join(save_root, f'{idx_name}_{2}.png'), img_list[2])
        plt.imsave(os.path.join(save_root, f'{idx_name}_{3}.png'), img_list[3])
        plt.imsave(os.path.join(save_root, f'{idx_name}_{4}.png'), img_list[4], cmap='jet')
        plt.imsave(os.path.join(save_root, f'{idx_name}_{5}.png'), img_list[5], cmap='jet')
        plt.imsave(os.path.join(save_root, f'{idx_name}_{6}.png'), img_list[6], cmap='jet')
        plt.imsave(os.path.join(save_root, f'{idx_name}_{7}.png'),cv2.cvtColor(img_list[7], cv2.COLOR_GRAY2RGB), cmap='gray')
        # plt.axis('off')
        # plt.subplot(242)
        # plt.imwrite()
        # # plt.axis('off')
        # # plt.subplot(243)
        # plt.imshow(img_list[2])
        # # plt.axis('off')
        # # plt.subplot(244)
        # plt.imshow(img_list[3])
        # # plt.axis('off')
        # # plt.subplot(245)
        # plt.imshow(img_list[4], cmap='jet')
        #
        # # plt.axis('off')
        # # plt.subplot(246)
        # plt.imshow(img_list[5],cmap='jet')
        # # plt.axis('off')
        # # plt.subplot(247)
        # plt.imshow(img_list[6], cmap='jet')
        # # plt.axis('off')
        # # plt.subplot(248)
        # plt.imshow(img_list[7])
        # plt.axis('off')
        # plt.savefig(os.path.join(save_root, idx_name))
        # plt.close()

    def tensor_to_np_cpu(self, tensor):
        x_cpu = tensor.squeeze(0).data.cpu().numpy()
        x_cpu = np.transpose(x_cpu, (1, 2, 0))
        return x_cpu

    def check(self, img):
        if len(img.shape) == 2:
            return img
        if img.shape[2] == 3:
            return img
        elif img.shape[2] == 1:
            return img.reshape(img.shape[0], img.shape[1])

MVTec_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                     'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                     'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

class_rocauc = {
                'bottle':(0, 0, 0, 0),
                'cable':(0, 0, 0, 0),
                'capsule':(0, 0, 0, 0),
                'carpet':(0, 0, 0, 0),
                'grid':(0, 0, 0, 0),
                'hazelnut':(0, 0, 0, 0),
                'leather':(0, 0, 0, 0),
                'metal_nut':(0, 0, 0, 0),
                'pill':(0, 0, 0, 0),
                'screw':(0, 0, 0, 0),
                'tile':(0, 0, 0, 0),
                'toothbrush':(0, 0, 0, 0),
                'transistor':(0, 0, 0, 0),
                'wood':(0, 0, 0, 0),
                'zipper':(0, 0, 0, 0)}

# model_name_list = ['VIT_dir', 'OO_AE']

if __name__ == '__main__':
    opt = DefaultConfig()
    test_no_mask = True
    test_add = False
    from datasets.dataset import MVTecDataset
    from torch.utils.data import DataLoader
    # opt.model_name = model_name_list[0]
    for classname in MVTec_CLASS_NAMES:
        opt.class_name = classname
        # opt.class_name = 'capsule'
        # save_name = opt.model_name+'_'+opt.backbone_name}_k={opt.k}_inpaint_num={opt.in_lay_num}
        # opt.resume = fr'D:\IMSN-LW\PAR-Net\result/{save_name}/weight/{opt.class_name}'
        # if test_no_mask == True:
        #     opt.model_name = 'RB_VIT_dir_res_ref'
        print(opt.class_name, opt.model_name)
        # print(opt.referenc_img_file)
        # opt.resume = r'result/RB_VIT_dir_res_ref_VGG/weight/capsule'
        opt.train_dataset = MVTecDataset(dataset_path=opt.data_root, class_name=opt.class_name, is_train=True)
        opt.test_dataset = MVTecDataset(dataset_path=opt.data_root, class_name=opt.class_name, is_train=False)
        opt.trainloader = DataLoader(opt.train_dataset, batch_size=opt.batch_size, shuffle=True)
        opt.testloader = DataLoader(opt.test_dataset, batch_size=1, shuffle=False)
        model = Model(opt, test_no_mask, test_add)
        model.test()

