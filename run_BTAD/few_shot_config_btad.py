import torch

model_name_list = ['VIT_dir', 'OO_AE', 'OO_AE_rec_and_oo']

class DefaultConfig(object):
    class_name = 'bottle'
    data_root = r'D:\IMSN-LW\dataset\BTech_Dataset_transformed'
    device = torch.device('cuda:1')
    model_name = model_name_list[1]
    batch_size = 1
    iter = 0
    niter = 300
    lr = 0.0001
    lr_decay = 0.90
    weight_decay = 1e-5
    momentum = 0.9
    nc = 3
    isTrain = True
    backbone_name = 'WideResnet50'
    # referenc_img_file = f'result/OO_AE_WideResnet50_k=4_inpaint_num=8/weight'
    resume =r''
    # mask_ratio = 0.5
    shuffle_ratio = 0.9
    k = 4
    in_lay_num = 8
    shot = 8
    is_log = False



if __name__ == '__main__':
    opt = DefaultConfig()
    opt.trai = 1
    print(opt.trai)