# 读取dataset，通过txtpath.txt 导入pngpath.txt最后获得png和标签
import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import natsort
import dgl
import numpy as np


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def defaultloader(path):
    return Image.open(path).convert("RGB")


class CustomDataSet(data.Dataset):
    def __init__(self, txt,Node, transform=None, target_transform=None, loader=defaultloader,):
        """TODO
        1.初始化文件路径和文件名列表
        初始化该类的一些基本参数
        """
        super(CustomDataSet, self).__init__()
        # img[0]是png路径path
        # img[1]是label
        # shapes_path 是每个模型的txt文件
        imgs = []       #保存graph
        shapes_path = [] #m个模型path
        label = []      #m个标签
        self.Node = Node
        # 打开输入文档
        fh = open(txt, 'r')
        for line in fh: #22次
            # 删除首尾空格字符
            line = line.strip('\n')
            line = line.rstrip('\n')
            # words[0] = C:\Users\Xue\Documents\PycharmProject\viewgraph\smalldatasetpng\10edit/10edit_bed_0001\pngpath.txt
            # words[1] = 1
            words = line.split()
            # print(words[1])
            label.append(words[1])
            # shapes_path[0] = C:\Users\Xue\Documents\PycharmProject\viewgraph\smalldatasetpng\10edit/10edit_bed_0001\pngpath.txt
            shapes_path.append(words[0])
            # print(len(shapes_path))
            # imgs.append((words[0],int(words[1])))
        fh.close()

        # print(len(imgs))

        # print(imgs)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.shape_path = shapes_path
        self.label = label
        self.TOTENSOR = transforms.ToTensor()


    def __len__(self):
        """返回该数据集的大小（长度）"""
        return len(self.imgs)

    def __getitem__(self, index):
        """
        接收一个index，通常是指一个list的index，这个list包括图片数据的路径和标签信息
        返回一个模型的Graph 和其对应的标签
        TODO
        1.从文件里读取一个数据（如numpy或者PIL.Image.open)
        2.预处理数据（torchvision.Transform)
        3.返回数据对（例如图像和标签）
        :param index:
        :return:img,label
        sample:
        img:[3,224,224] dtype:tensor
        label:0 dtype:int
        """
        # print(index)
        A_shape_path = self.shape_path[index]
        # print(A_shape_path)
        Graph =self.create_graph(A_shapes_path=A_shape_path)
        # TODO
        label = self.label[index]
        # A_shape_path是图片path，label是真实标签，
        # print(img)
        return Graph, label

    def create_graph(self,A_shapes_path):
        # 输入的是一个3D模型的路径，输出这个模型包括V个视图,V*V个边的graph
        PNG_path = []
        PNG_loader = []
        tmp_tensor = []
        TheGraph = dgl.DGLGraph()
        TheGraph.add_nodes(self.Node) #添加Node个点
        for i in range(0, self.Node): #添加Node * Node个边
            for j in range(0, self.Node):
                TheGraph.add_edge(i, j)
        tmptxt = open(A_shapes_path, 'r')
        for line in tmptxt: #36次循环
            # print(line)
            line = line.strip('\n')
            # print(line)
            PNG_path.append(line)
            img = self.loader(line)

            if self.transform is not None:
                img_tensor = self.transform(img)
            # print(img_tensor.shape)
            PNG_loader.append(img_tensor)
        # print(len(PNG_loader))
        for  i in range(len(PNG_loader)):
            tmp_tensor.append(torch.tensor(PNG_loader[i]))
        PNG_tensor = torch.stack(tmp_tensor,dim=0)
        # print(PNG_tensor.shape)
        TheGraph.ndata['img'] = PNG_tensor
        return TheGraph




if __name__ == '__main__':
    path =r'C:\Users\Xue\Documents\PycharmProject\viewgraph\smalldatasetpng\trainCollection.txt'
    dataload =CustomDataSet(path,transform=transform,loader=defaultloader,Node=36)
    print(len(dataload))
    for i in range(22):
        test = dataload.__getitem__(i)
        graph, label = test
        print(graph)
        print(label)

    # print(test)
    # print(label)
    # print(img.shape)

