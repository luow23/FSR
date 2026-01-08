from sklearn.cluster import k_means
from sklearn.manifold import TSNE
import torch
from models.mvtec import MVTecDataset
from torch.utils.data import DataLoader
from models.pretrained_feature import PretrainedFeature
from matplotlib import pyplot as plt
import numpy as np
pre_feature = PretrainedFeature('vgg16_bn')
cluster_dataset = MVTecDataset(class_name='tile')
cluster_dataloader = DataLoader(cluster_dataset, batch_size=8, shuffle=False, drop_last=False)
feature_vector_list = []
# for i, (x, _, _, _) in enumerate(cluster_dataloader, 0):
#     feature_vector = pre_feature(x)
#     feature_vector_list.append(feature_vector)
#     if i==0:
#         break
#
# feature_vector_all = torch.cat(feature_vector_list, dim=0)

# tsne = TSNE(n_components=2)
# result = tsne.fit_transform(feature_vector_all)

# def plot_embedding(data, title):
#     x_min, x_max = np.min(data, 0), np.max(data, 0)
#     data = (data - x_min) / (x_max - x_min)
#
#     fig = plt.figure()
#     ax = plt.subplot(111)
#     plt.scatter(data[:, 0], data[:, 1])
#     plt.xticks([])
#     plt.yticks([])
#     plt.title(title)
#     return fig
#
# fig = plot_embedding(result, 'tsne')
# plt.show()
feature_vector_all =torch.rand((200, 2))
center, label, _ = k_means(feature_vector_all, 2)
# print(type(center))
# print(label.shape)
# print(np.argwhere(label==0).shape)
a = [1, 2, 3]
a = sorted(a, key=lambda x:x, reverse=True)
print(a)
# print(len(label))