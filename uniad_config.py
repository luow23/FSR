import torch

model_name_list = ['VIT_dir', 'OO_AE','OO_AE_without_pos','OO_AE_CNN', 'OO_AE_learned_pos', 'OO_AE_rec_and_oo']

class DefaultConfig(object):
    class_name = 'bottle'
    data_root = r'D:\IMSN-LW\dataset\mvtec_anomaly_detection'
    device = torch.device('cuda:0')
    model_name = model_name_list[1]
    batch_size = 16
    iter = 0
    niter = 300
    lr = 0.0001
    lr_decay = 0.90
    weight_decay = 1e-5
    momentum = 0.9
    nc = 3
    isTrain = True
    backbone_name = 'WideResnet50'
    resume =r''
    # mask_ratio = 0.5
    shuffle_ratio = 0.1
    k = 4
    in_lay_num = 8
    is_log = False

if __name__ == '__main__':
    opt = DefaultConfig()
    opt.trai = 1
    print(opt.trai)