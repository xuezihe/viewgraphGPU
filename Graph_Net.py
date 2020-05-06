# 2020年4月23日
# GPU ver
import dgl
import math
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.models as models

import networkx as nx
import time
import numpy as np
from MyUtiles import caculatecos
device = torch.device('cuda:0')
def make_vggmodel():
    vggmodel = models.vgg19_bn(pretrained=False)
    vggpre = torch.load(r'.\vgg19_bn-c79401a0.pth')
    vggmodel.load_state_dict(vggpre)
    vggmodel.classifier = nn.Sequential(*[vggmodel.classifier[i] for i in range(4)])
    print(vggmodel)
    vggmodel=vggmodel.eval()
    return(vggmodel)
def make_vggmodel_nobn():
    vggmodel = models.vgg19(pretrained=False)
    vggpre = torch.load(r'.\vgg19-dcbb9e9d.pth')
    vggmodel.load_state_dict(vggpre)
    vggmodel.classifier = nn.Sequential(*[vggmodel.classifier[i] for i in range(4)])
    print(vggmodel)
    vggmodel=vggmodel.eval()
    return(vggmodel)

# 还在用
# 建图
def build_karate_club_graph(N):
    g = dgl.DGLGraph()
    g.add_nodes(N)
    for i in range(0,N):
        for j in range(0,N):
            g.add_edge(i,j)
    return g

# GAT
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim,F=256):
        # g:graph,in_dim:N(256),out_dim:label
        super(GATLayer, self).__init__()
        # self.g = g
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.F = F
        self.fc1 = nn.Sequential(nn.Linear(in_features=self.in_dim,out_features=self.F),
                                 nn.ReLU(inplace=True,),
                                 )
        self.fc2 = nn.Sequential(nn.Linear(in_features=self.in_dim,out_features=1),
                                 nn.ReLU(inplace=True),
                                 )
        # self.attn_fc = nn.Linear(out_dim, 1, bias=True)
        self.WcCijwc_linear()
        self.reset_parameters()


    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.Wc, gain=gain)
        nn.init.xavier_normal_(self.alphaC, gain=gain)
        nn.init.xavier_normal_(self.alphaf, gain=gain)
        nn.init.xavier_normal_(self.b, gain=gain)
        nn.init.xavier_normal_(self.w, gain=gain)
        nn.init.xavier_normal_(self.Wf, gain=gain)
        nn.init.xavier_normal_(self.bias, gain=gain)

        # nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def WcCijwc_linear(self):
        self.Wc = nn.Parameter(torch.ones(self.out_dim,self.in_dim,requires_grad=True))
        self.alphaC = nn.Parameter(torch.ones(self.in_dim,1,requires_grad=True))
        self.alphaf = nn.Parameter(torch.ones(self.F,1,requires_grad=True))
        self.b = nn.Parameter(torch.randn(self.out_dim,1,requires_grad=True))
        self.w = nn.Parameter(torch.randn(1, self.out_dim, requires_grad=True))
        # final
        self.Wf =nn.Parameter(torch.randn(self.out_dim,self.F,requires_grad=True))
        self.bias = nn.Parameter(torch.randn(self.out_dim, 1, requires_grad=True))
        # self.Ci =torch.zeros(self.N,self.N,requires_grad=False)


    def message_func_1(self, edges):
        return {'sijjjcijj': edges.data['tmp_sijjcijj']}

    def reduce_func_1(self, nodes):
        print('message')
        self.View = nodes.mailbox['sijjjcijj'].size(0)
        print("mailbox",nodes.mailbox['sijjjcijj'].shape)
        # self.View是视图数量V
        # print(self.View)
        # print(nodes.mailbox['sijjjcijj'].shape)
        Cij = torch.sum(nodes.mailbox['sijjjcijj'],dim=1)
        print("Cij",Cij.shape)

        # print(Cij.shape)
        return {'correlation': Cij}

    def caculateSijjCijj(self,edges):
        # print(edges.data['Cijj'].shape)
        # print(edges.data['Sij'].shape)
        # tmp_sijjcijj = edges.data['Sij'] * edges.data['Cijj']
        # tmp_sijjcijj = torch.mul(edges.data['Sij'],edges.data['Cijj'])
        SijjCijj_list = []
        for i in range(edges.data['Sij'].size(0)):
            # print("edges.data['Sij'][i].shape",edges.data['Sij'][i].shape)
            # print("edges.data['Cijj'][i].shape",edges.data['Cijj'][i].shape)
            tmp_sijjcijj = torch.mul(edges.data['Sij'][i],edges.data['Cijj'][i])
            SijjCijj_list.append(tmp_sijjcijj)
        # tmp_sijjcijj:[N,N]
        print(len(SijjCijj_list))
        tmp_sijjcijj_new = torch.stack(SijjCijj_list,dim=0)
        # tmp_sijjcijj_new:[1296,128,128](V*V,N,N]
        print("tmp_sijjcijj_new.shape",tmp_sijjcijj_new.shape)
        return {'tmp_sijjcijj' : tmp_sijjcijj_new}

    def attention(self,nodes):
        print('attention')
        correlation = nodes.data['correlation']
        print(correlation[10].shape)
        attention_list=[]
        for i in range(self.View):
            # print(self.Wc.shape)
            WcCij = torch.mm(self.Wc,correlation[i])
            WcCijalphac = torch.mm(WcCij,self.alphaC)
            WfalphaF = torch.mm(self.Wf,self.alphaf)
            att_add = torch.add(WcCijalphac,WfalphaF)
            # print(att_add.shape)
            # attention = self.attn_fc(att_add)
            attention = self.w @(torch.add(att_add,self.b))
            attention_list.append(attention)
        attention_total = torch.stack(attention_list,dim=0)

        print("attention_total",attention_total.shape)
        attention_total = attention_total.view(self.View,1)
        attention_total = F.softmax(attention_total,dim=0)
        # print(attention_total)
        print('attention_total',attention_total.shape)
        print("attention_total",attention_total)
        return {'attention' : attention_total}

    def caculateCi(self,nodes):
        Ci_list=[]
        for i in range(self.View):
            tmp_Ci = torch.mul(nodes.data['attention'][i],nodes.data['correlation'][i])
            Ci_list.append(tmp_Ci)
        Ci_total = torch.stack(Ci_list,dim=0)
        # print(Ci_total.shape)
        return {'Ci' : Ci_total}
    def Ci_sum(self,):
        print("CI_sum")
        all_nodes_Ci = self.g.nodes[:].data['Ci']
        print("all_nodes",all_nodes_Ci.shape)
        # self.g.nodes[:].data['Ci']:[V,N,N]
        Ci = torch.sum(all_nodes_Ci,dim=0)
        return Ci
    def forward(self, G):
        # G:graph
        # TODO
        self.g = G
        # print(self.g.nodes[0].data['features'].shape)
        self.g.apply_edges(self.caculateSijjCijj)
        self.g.update_all(self.message_func_1, self.reduce_func_1)
        self.g.apply_nodes(self.attention)
        self.g.apply_nodes(self.caculateCi)
        print('after_attention')
        Ci = self.Ci_sum() #[256,256]
        # print(Ci.shape)
        Fi = self.fc1(Ci)
        Fi =torch.sigmoid(Fi)
        # print('1',Fi.shape)
        Fi = self.fc2(Fi.t())
        # print('2',Fi.shape)
        P = self.Wf @ Fi +self.bias
        P = P.t()
        print('P',P.shape)
        # print(P)
        # return F.softmax(P,dim=1)
        return P

# SPC
class SPC(nn.Module):
    def __init__(self,View,derta=10):
        super(SPC, self).__init__()
        self.V = View
        self.derta = derta
    def forward(self,G,costheta):
        # input:[32,256]([V,N])
        for i in range(self.V):
            # Eijj = 0
            # Sijj = 0
            for j in range(self.V):
                tmp_list=[]
                Eijj = 0.5 * (1 - costheta[i][j].float())
                # print(Eijj.shape)
                G.edata['Eij'][G.edge_id(i, j)] = Eijj
                Sijj_float = math.exp(-self.derta * Eijj)
                # print("Sijj_float",Sijj_float)
                tmp_list.append(Sijj_float)
                Sijj = torch.tensor(tmp_list,requires_grad=False)
                # print('sijj',type(Sijj),Sijj.shape)
                G.edata['Sij'][G.edge_id(i, j)] = Sijj
                # print(G.edata['Sij'][G.edge_id(i, j)].shape)
            for j_ in range(self.V):
                cijj = torch.mm(G.nodes[i].data['features'].t(),G.nodes[j_].data['features'])
                # print(cijj.shape)
                G.edata['Cijj'][G.edge_id(i,j_)] = cijj
                # Cijj:[N,N]
        return G


# embedding
class embedding(nn.Module):
    def __init__(self,V,N):
        super(embedding, self).__init__()
        self.weigh = nn.Parameter(torch.ones(4096,N,requires_grad=True))
        self.bias = nn.Parameter(torch.randn(V,N,requires_grad=True))
        torch.nn.init.kaiming_normal_(self.weigh)
        torch.nn.init.kaiming_normal_(self.bias)

    def forward(self,inputs):
        output = inputs @ self.weigh + self.bias
        output = F.softmax(output,dim=1)
        return output

# put everything together
class Net(nn.Module):
    def __init__(self,View,N,derta,G,label):
        super(Net, self).__init__()
        self.N = N
        self.G = G
        self.View = View
        # self.vgglayer = make_vggmodel_nobn()
        self.embeddinglayer = embedding(V=View,N=N)
        # self.build_karate_club_graph =build_karate_club_graph(N =View)
        self.SPC = SPC(View=View,derta=derta)
        self.GAT = GATLayer(in_dim=self.N ,out_dim=label,F=256)
    def forward(self,input,costheta):
        '''input:[32(V),4096]'''
        # vgg_input:[b,3,224,224]
        # after_vgg = self.vgglayer(input)
        # print('VGG_finish')
        after_emb = self.embeddinglayer(input)
        print('embedding_finish')
        # after_emb:[32,256]([V,N])
        self.G.ndata['features'] =after_emb
        # print(self.G.ndata['features'].shape)
        self.G.edata['Cijj'] = torch.zeros(self.View*self.View,self.N,self.N).to(device)
        self.G.edata['Eij'] = torch.zeros(self.View *self.View,1).to(device)
        self.G.edata['Sij'] = torch.zeros(self.View *self.View,1).to(device)
        self.G.edata['tmp_sijjcijj'] =torch.zeros(self.View *self.View,self.N,self.N).to(device)
        self.G.ndata['correlation'] = torch.zeros(self.View, self.N, self.N).to(device)
        self.G.ndata['Ci'] = torch.zeros(self.View,self.N,self.N).to(device)
        after_SPC = self.SPC(G=self.G,costheta=costheta)
        # after_SPC:[V,N*N]
        print('PSC_finish')
        logits = self.GAT(after_SPC)
        # print("logits",logits)
        print('GAT_finish')

        return logits

def this_0():
    make_vggmodel_nobn()
if __name__ == '__main__':
    G = build_karate_club_graph(N=36)
    # input = torch.randn(36,4096)
    input =np.load(r'C:\Users\Xue\Documents\PycharmProject\viewgraph\without_vgg\4.23_36_img_label_22models.npz')
    img=input['img']
    label = input['label']
    print(img.shape)
    print(label.shape)
    img_tensor = torch.tensor(img)
    cameraposition = r'C:\Users\Xue\Desktop\ML\3dviewgraph\datasetpath\position.txt'
    costheta = caculatecos(cameraposition,M=22,V=36) #[b(m),v,v] cos值
    # costheta = costheta[0]
    print(costheta[0].shape)
    net = Net(View=36, N=128, derta=1, G=G, label=2)
    # for bachidx,x in enumerate(input):
    #TODO
    for idx ,img in enumerate(img_tensor):
        local_costheta = costheta[idx] #[V,V]
        logits = net(img,costheta=local_costheta)
        print(logits.shape)
        print(logits)
        print('finish')



