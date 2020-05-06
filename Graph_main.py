# GPU use GAT
import torch, math
import time, copy
from torch.utils.data import DataLoader
from dataload import CustomDataSet as Mydataset
from torchvision import transforms
import torchvision.models as models
from torch import nn as nn
from torch import optim
from Graph_Net import Net,build_karate_club_graph
import dgl
from visdom import Visdom
from MyUtiles import caculatecos
import numpy as np
import copy

def collate(samples):
    # samples: a list of pair
    # (graph,label)
    graph, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graph)
    return batched_graph, torch.tensor(labels)


if __name__ == '__main__':

    # 输入是np矩阵
    # 1model
    # train_list = r'C:\Users\Xue\Documents\PycharmProject\viewgraph\without_vgg\4.29_36_img_label_1models.npz'

    # 32view 200models
    train_list = r'data/4.16_32_img_label_200models.npz'

    # 36view 22models
    # train_list = r'C:\Users\Xue\Documents\PycharmProject\viewgraph\without_vgg\4.23_36_img_label_22models.npz'

    # 32view_80models_eval
    eval_list = r'data/4.29_32_img_label_80models_eval.npz'


    # TODO
    # VISDOM
    viz = Visdom(env="viewgraph")
    viz.line([0], [0], win='batchsize_loss', opts=dict(title='train loss'))
    viz.line([0], [0], win='acc', opts=dict(title='eval_acc'))
    global_step = 0
    eval_global_step = 0
    best_acc = 0
    acc_text_window =viz.text('best_acc_history')

    # 1model
    # train_cameraposition = r'C:\Users\Xue\Desktop\ML\3dviewgraph\Big_train_dataset\Big_train_dataset_PNG_32\position1model.txt'
    #32view 200models
    train_cameraposition = r'data/Big_train_dataset_PNG_32position.txt'
    # 32view_80models_eval
    eval_cameraposition = r'data/Big_eval_dataset_PNG_32position.txt'

    device = torch.device('cuda:0')
    train_model_num = 200
    eval_model_num = 80
    Node = 32      #32 views
    Label_num = 2  #label
    N_num = 128
    F_num = 256
    lr = 0.009
    spc_sigma = 10


    # train_load
    train_loader = np.load(train_list)
    train_img =torch.tensor(train_loader['img'],requires_grad=False)
    train_label = torch.from_numpy(train_loader['label'])
    train_label = train_label[:,0:1]
    print('train_img:',train_img.shape)
    print('train_label:',train_label.shape)

    #eval_load
    eval_loader = np.load(eval_list)
    eval_img =torch.tensor(eval_loader['img'],requires_grad=False)
    eval_label = torch.from_numpy(eval_loader['label'])
    eval_label = eval_label[:,0:1]
    print('eval_img',eval_img.shape)
    print("eval_label",eval_label.shape)
    print('finish dataload')

    # Net
    G = build_karate_club_graph(N=Node)
    train_cos = caculatecos(positionpath=train_cameraposition, M=train_model_num, V=Node)
    eval_cos = caculatecos(positionpath=eval_cameraposition, M=eval_model_num, V=Node)

    net = Net(View=Node, N=N_num, G=G, derta=spc_sigma, label=Label_num).to(device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    #Save state
    train_acc_history =[]
    eval_acc_history=[]

    best_model_wts =copy.deepcopy(net.state_dict())
    savepth_path = r'checkpoint_4.30_1_200models_80models_2label.pth'

    for epoch in range(10):
        net.train()
        print("epoch", epoch)
        train_correct =0
        for batchidx, input in enumerate(train_img):
            print('batch',batchidx)
            input =input.to(device)
            label = train_label[batchidx].to(device)
            # print("label", label,label.shape)
            # label:[1]
            local_costheta = train_cos[batchidx].to(device)  # [V,V]([36,36])
            logit = net(input, costheta=local_costheta)
            # logit:[1,L]
            _,train_predict =torch.max(logit,1)
            print('logit',logit.shape,logit)
            print("label",label.shape,label)
            loss = loss_fun(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if train_predict ==label :
                print("epoch:{0};bachidx:{1} is True".format(epoch, batchidx))
                train_correct +=1


            # VISDOM
            global_step += 1
            viz.line([loss.item()], [global_step], win='batchsize_loss', update='append')
            print("train_predict",train_predict)
            print("batch {},loss {}".format(batchidx, loss.item()))
        train_acc =train_correct / train_model_num
        train_acc_history.append(train_acc)
        net.eval()
        print('eval_process')
        with torch.no_grad():
            total_correct = 0
            # total_num = 0
            for eval_batchidx,input in enumerate(eval_img):
                input = input.to(device)

                local_eval_label = eval_label[eval_batchidx].to(device)
                print(eval_label.shape)
                local_eval_cos = eval_cos[eval_batchidx].to(device)
                eval_logit = net(input, costheta=local_eval_cos)
                print('eval_logit', eval_logit.shape, eval_logit)
                # eval_logit:[1,2]
                print("eval_label", local_eval_label.shape, local_eval_label)
                # eval_label:[1]
                _,predic =torch.max(eval_logit,1)
                print("predict",predic)
                total_correct += (predic == local_eval_label).sum().item()
                if predic ==local_eval_label:
                    print("epoch:{0};bachidx:{1} is True".format(epoch,eval_batchidx))
                # correct = torch.eq(eval_predict, label).float().sum().item()
                # total_correct += 1
            epoch_acc = total_correct / eval_model_num


            # EVAL_VISDOM
            eval_global_step += 1
            viz.line([epoch_acc], [eval_global_step], win='acc', update='append')
            print('epoch', epoch, 'test acc:', epoch_acc)
            eval_acc_history.append(epoch_acc)
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())
                viz.text("New best eval acc is{0},epoch:{1}".format(epoch_acc,epoch),win=acc_text_window,append=True)


        # save
        print('save_process')
        print("eval_acc_history",eval_acc_history)
        print('train_acc_history',train_acc_history)
        # net.load_state_dict(best_model_wts)
        torch.save({'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict ': optimizer.state_dict(),
                    'loss': loss,
                    # 'acc': acc,
                    }, savepth_path)
    # finish
    print('finish')
