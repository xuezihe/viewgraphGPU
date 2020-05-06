import numpy as np
import torch
eval_list = r'data/4.29_32_img_label_80models_eval.npz'
eval_loader = np.load(eval_list)
eval_img = torch.tensor(eval_loader['img'], requires_grad=False)
eval_label = torch.from_numpy(eval_loader['label'])
print(eval_label.shape)
eval_label = eval_label[:, 0:1]
print(eval_label.shape)
for i in range(80):
    local=eval_label[i]
    print(local.shape)
