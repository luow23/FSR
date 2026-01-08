# from models.RB_VIT import RB_VIT
import torch
import os
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
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
import pandas as pd

class Model(object):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.model = eval(opt.model_name)(opt)
        self.device = opt.device
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
        if self.opt.model_name == "VIT_dir":
            self.save_root = f"./fewshotresult_BTAD/{opt.model_name}_{opt.backbone_name}_k={opt.k}_inpaint_num={opt. in_lay_num}_shot={opt.shot}"
        else:
            self.save_root = f"./fewshotresult_BTAD/{opt.model_name}_{opt.backbone_name}_k={opt.k}_inpaint_num={opt.in_lay_num}_shuffle_ratio={opt.shuffle_ratio}_shot={opt.shot}_avg_Ture_noise_scale30"
        if self.opt.is_log == True:
            os.makedirs(self.save_root, exist_ok=True)
            df_loss = pd.DataFrame(columns=['Epoch', 'Train loss'])
            df_loss.to_csv(os.path.join(self.save_root, f'{self.class_name}_train_loss.csv'), index=False)
            df_auroc = pd.DataFrame(columns=['Epoch', 'Image AUROC', 'Pixel AUROC'])
            df_auroc.to_csv(os.path.join(self.save_root, f'{self.class_name}_train_auroc.csv'), index=False)
        self.ckpt_root = os.path.join(self.save_root, "weight/{}".format(self.class_name))
        self.vis_root = os.path.join(self.save_root, "img/{}".format(self.class_name))
    def get_max(self, tensor):
        a_1, _ = torch.max(tensor, dim=1, keepdim=True)
        a_2, _ = torch.max(a_1, dim=2, keepdim=True)
        a_3, _ = torch.max(a_2, dim=3, keepdim=True)
        return a_3
    def train(self):
        loss_now = 100000
        auc_now = 0.
        ap_now = 0.
        patience = 20
        no_update_num = 0
        for epoch in range(self.opt.iter, self.opt.niter):
            self.model.Feature_extractor.eval()
            self.model.Roncon_model.train(True)
            self.model.to(self.device)
            loss_total = 0.
            count = 0
            for index, (x, _, _, _) in enumerate(tqdm(self.trainloader, ncols=80)):
                bs = x.shape[0]
                x = x.to(self.device)
                # ref_x = get_pos_sample(self.opt.referenc_img_file, self.device, bs)
                # arti_x = arti_x.to(self.device)
                # arti_mask = F.interpolate(arti_mask, (64, 64))
                # arti_mask = arti_mask.to(self.device)
                # arti_mask = F.max_pool2d(arti_mask, self.opt.k, self.opt.k)
                # print(arti_mask)

                deep_feature, _, recon_feature, loss = self.model(x, 'train')
                self.loss_scaler(loss, self.optimizer_g, parameters=self.model.Roncon_model.parameters(), update_grad=(index + 1) % 1 == 0)
                loss_total += loss.item()
                count += 1

            loss_total = loss_total / count
            print('the {} epoch is done   loss:{}'.format(epoch + 1, loss_total))
            if self.opt.is_log == True:
                loss_list = [epoch + 1, loss_total]
                loss_data = pd.DataFrame([loss_list])
                loss_data.to_csv(os.path.join(self.save_root, f'{self.class_name}_train_loss.csv'), mode='a',
                                 header=False, index=False)
            # if (epoch+1) == 300:
            #     print('save model')
            #     weight_dir = self.ckpt_root
            #     os.makedirs(weight_dir, exist_ok=True)
            #     torch.save({'epoch': epoch + 1, 'state_dict': self.model.state_dict()},
            #                f'%s/{self.opt.model_name}_300.pth' % (weight_dir))
            if (epoch + 1) % 10  == 0:
                 # self.test_2()
                x1, x2, x3, x4 = self.train_test()
                if self.opt.is_log == True:
                    save_list = [epoch + 1, x1, x2]
                    data = pd.DataFrame([save_list])
                    data.to_csv(os.path.join(self.save_root, f'{self.class_name}_train_auroc.csv'), mode='a',
                             header=False, index=False)
                auc_roc = x1+x2
                if auc_roc > auc_now:
                    # no_update_num = 0
                    auc_now = auc_roc
                    ap_now = x3 + x4
                    class_rocauc[self.opt.class_name] = (x1, x2, x3, x4)
                    print('save model')
                    weight_dir = self.ckpt_root
                    os.makedirs(weight_dir, exist_ok=True)
                    torch.save({'epoch': epoch + 1, 'state_dict': self.model.state_dict()},
                               f'%s/{self.opt.model_name}.pth' % (weight_dir))
                elif auc_roc == auc_now:
                    if x3+x4 > ap_now:
                        ap_now = x3 + x4
                        class_rocauc[self.opt.class_name] = (x1, x2, x3, x4)
                        print('save model')
                        weight_dir = self.ckpt_root
                        os.makedirs(weight_dir, exist_ok=True)
                        torch.save({'epoch': epoch + 1, 'state_dict': self.model.state_dict()},
                                   f'%s/{self.opt.model_name}.pth' % (weight_dir))

                # else:
                #     no_update_num += 1
                #     print('no_update_num:{}'.format(no_update_num))
            # if no_update_num > patience:
            #     break



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
        return round(image_level_ROCAUC, 3), round(pixel_level_ROCAUC, 3), round(image_level_AP, 3), round(pixel_level_AP, 3)
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
        feature_map_list = [torch.mean(i.clone(), dim=1).squeeze(0).cpu().detach().numpy() for i in feature_map_list]
        return feature_map_list

    def train_test(self):
        test_y_list = []
        test_mask_list = []
        score_list = []
        score_map_list = []

        for idx, (x, y, mask, name) in enumerate(tqdm(self.testloader, ncols=80)):
            test_y_list.extend(y.detach().cpu().numpy())
            test_mask_list.extend(mask.detach().cpu().numpy())
            self.model.eval()
            self.model.to(self.device)
            x = x.to(self.device)
            mask = mask.to(self.device)
            mask_cpu = mask.cpu().detach().numpy()[0, :, :, :].transpose((1, 2, 0))
            # ref_x = get_pos_sample(self.opt.referenc_img_file, self.device, 1)
            deep_feature, ref_feature, recon_feature, _ = self.model(x, 'test')

            feature_map_vis_list = self.feature_map_vis([deep_feature, ref_feature, recon_feature])
            cos_mapk1, cos_mapk64, mse_map = self.model.a_map(deep_feature, recon_feature)
            mse_map = gaussian_filter(mse_map, sigma=4)
            cos_mapk1 = gaussian_filter(cos_mapk1, sigma=4)
            # dir_amap = gaussian_filter(dir_amap, sigma=4)
            # ssim_amap = gaussian_filter(ssim_amap, sigma=4)
            # print(type(name0]))
            name_list = name[0].split(r'!')
            # print(name_list)
            category, img_name = name_list[-2], name_list[-1]
            amap = cos_mapk1 * mse_map
            # amap = dir_amap + dis_amap
            self.vis_img(
                [x, *feature_map_vis_list, mse_map, cos_mapk1, amap, mask_cpu],
                os.path.join(self.vis_root, category), img_name)

            score_list.extend(np.array(np.std(amap)).reshape(1))
            score_map_list.extend(amap.reshape((1, 1, 256, 256)))

        image_level_ROCAUC, pixel_level_ROCAUC, image_level_AP, pixel_level_AP= self.cal_auc(score_list, score_map_list, test_y_list, test_mask_list)
        # F1_score = self.F1_score(F1_score_map_list, test_mask_list)
        print('image_auc_roc: {} '.format(image_level_ROCAUC),
              'pixel_auc_roc: {} '.format(pixel_level_ROCAUC),
              'image_AP: {}'.format(image_level_AP),
              'pixel_AP: {}'.format(pixel_level_AP)
             )
        class_rocauc[self.opt.class_name] = (image_level_ROCAUC, pixel_level_ROCAUC, image_level_AP, pixel_level_AP)
        # class_rocauc[name] =
        return image_level_ROCAUC, pixel_level_ROCAUC, image_level_AP, pixel_level_AP

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
        plt.grid('on')
        plt.subplot(246)
        plt.imshow(img_list[5], cmap='jet')
        plt.axis('off')
        plt.subplot(247)
        plt.imshow(img_list[6],cmap='jet')
        plt.axis('off')
        plt.subplot(248)
        plt.imshow(img_list[7], cmap='gray')
        plt.axis('off')
        # plt.subplot(259)
        # plt.imshow(img_list[8], cmap='gray')
        # plt.axis('off')
        # plt.subplot(2,5,10)
        # plt.imshow(img_list[9])
        # plt.axis('off')
        if 'bmp' in idx_name:
            idx_name = idx_name.replace('bmp', 'png')
        plt.savefig(os.path.join(save_root, idx_name))
        plt.close()


    def tensor_to_np_cpu(self, tensor):
        x_cpu = tensor.squeeze(0).data.cpu().numpy()
        x_cpu = np.transpose(x_cpu, (1, 2, 0))
        return x_cpu

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

    def check(self, img):
        if len(img.shape) == 2:
            return img
        if img.shape[2] == 3:
            return img
        elif img.shape[2] == 1:
            return img.reshape(img.shape[0], img.shape[1])

BTAD_CLASS_NAMES = [
    '01', '02', '03'
]

class_rocauc = {
    '01': (0, 0, 0, 0),
    '02': (0, 0, 0, 0),
    '03': (0, 0, 0, 0)
}


if __name__ == '__main__':
    from few_shot_config_btad import DefaultConfig
    opt = DefaultConfig()
    from datasets.BTADdataset import BTADDataset, FewshotBTADDataset
    from torch.utils.data import DataLoader
    for classname in BTAD_CLASS_NAMES:
        opt.class_name = classname
        # opt.class_name = 'capsule'
        # opt.referenc_img_file = f'data/mvtec_anomaly_detection/{opt.class_name}/train/good/000.png'
        # opt.referenc_img_file =  f'data/ref/{opt.class_name}/ref.png'
        # opt.referenc_img_file = f'natrual.JPEG'
        print(opt.class_name, opt.model_name, opt.k, opt.shuffle_ratio, opt.in_lay_num)
        # print(opt.referenc_img_file)
        # opt.resume = r'result/RB_VIT_dir_res_ref_VGG/weight/capsule'
        opt.train_dataset = FewshotBTADDataset(dataset_path=opt.data_root, class_name=opt.class_name, is_train=True, resize=256, k=opt.shot)
        opt.test_dataset = BTADDataset(dataset_path=opt.data_root, class_name=opt.class_name, is_train=False)
        opt.trainloader = DataLoader(opt.train_dataset, batch_size=opt.batch_size, shuffle=True)
        opt.testloader = DataLoader(opt.test_dataset, batch_size=1, shuffle=False)
        model = Model(opt)
        model.train()
    print(class_rocauc)
    value = list(class_rocauc.values())
    img_roc = [i[0] for i in value]
    pixel_roc = [i[1] for i in value]
    img_ap = [i[2] for i in value]
    pixel_ap = [i[3] for i in value]
    mean_img_roc = np.mean(np.array(img_roc))
    mean_pixel_roc = np.mean(np.array(pixel_roc))
    mean_img_ap = np.mean(np.array(img_ap))
    mean_pixel_ap = np.mean(np.array(pixel_ap))
    print(round(mean_img_roc, 4), round(mean_pixel_roc, 4), round(mean_img_ap, 4), round(mean_pixel_ap, 4))
