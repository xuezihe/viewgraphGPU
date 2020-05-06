import math
import torch

# 计算cos（theta）
def caculatetheta(list1,list2):
    '''list1:theta
    list2:phi
    返回的Thetalist是list套list，Thetalist[0][0]
    '''
    dict_ = {}
    Thetalist = []

    def topi(theta):
        # print(theta,type(theta))
        tmp = theta/180 *math.pi
        return tmp

    for i in range(len(list1)): #36个

        theta = topi(int(list1[i]))
        phi = topi(int(list2[i]))
        loc_x = math.cos(theta) * math.sin(phi)
        loc_y = math.cos(theta) * math.cos(phi)
        loc_z = math.sin(theta)
        tmp = []
        tmp.append(loc_x)
        tmp.append(loc_y)
        tmp.append(loc_z)
        dict_[i] = tmp
        # print('dict_num',len(dict_))
    for i in dict_.keys():
        thetalist = []
        for j in dict_.keys():
            a2 = math.pow(dict_[i][0], 2) + math.pow(dict_[i][1], 2) + math.pow(dict_[i][2], 2)
            b2 = math.pow(dict_[j][0], 2) + math.pow(dict_[i][1], 2) + math.pow(dict_[j][2], 2)
            c2 = math.pow(dict_[i][0] - dict_[j][0], 2) + math.pow(dict_[i][1] - dict_[j][1], 2) + math.pow(
                dict_[i][2] - dict_[j][2], 2)
            tmptheta = (a2 + b2 - c2) / (2 * math.sqrt(a2) * math.sqrt(b2))
            if tmptheta > 0:
                tmptheta = min(tmptheta, 1.0)
            else:
                tmptheta = max(tmptheta, -1)
            thetalist.append(tmptheta)
        Thetalist.append(thetalist)
    # print('caculatetheta数量',Thetalist[2][2],len(Thetalist))
    return Thetalist


# 获得theta角,png，模型和theta绑定
def getTheta(path, modelname, r=1,):
    '''
        path:list
        modelname:list
        output是dict,是字典！，用模型名字查找，第v号视角的m个模型对应的36个角度的cos值
    '''
    name = []
    output ={}
    file =open(path,'r')
    strlines =file.readlines()
    print('begin getTheta')
    for i in strlines: #m个
        # print('strline',i)
        strArray =i.split('[',2)
        # print("str[0]",strArray[0])

        name.append(strArray[0])
        strArray[0] =strArray[0].replace(' ','')

        # print(strArray[0],type(strArray[0]))

        # output.append(strArray[1])
        for j in modelname:#22循环
            # print(j,type(j))
            if strArray[0] == j:
                # print('yes')
                str = strArray[1]
                # print(str)
                str = str.replace('(','')
                str = str.replace(')','')
                str = str.replace(' ','')
                str = str.replace(']','')
                str = str.rstrip()

                # print(str)

                thetalist =str.split(',')
                # print(thetalist)
                tmplist_A=thetalist[::2]
                tmplist_B =thetalist[1::2]
                # print(len(tmplist_B))
                # print(len(tmplist_A))
                # for vidx in range(v-1):#36次

                thetaoutput = caculatetheta(tmplist_A, tmplist_B) #list[a][b],a是第一个角j，b是第二角j'
                # print('thetaoutput',thetaoutput[0])
                output[j] = thetaoutput
            else:
                # print("no")
                continue
    return output

# 获得model的名字，返回list
def getmodelname(path):
    modelnameList = []
    file = open(path, 'r+')
    for i in file:
        # print(i)
        tmp = i.split(' ')
        modelnameList.append(tmp[0])
    # print(modelnameList)
    file.close()
    return modelnameList
# position计算
def caculatecos(positionpath,M,V):
    '''input:path
    output:和所有其他视图的cos值tensor[b(等于m）,V,V]


    '''
    tmp_cos_list =[]
    file = open(positionpath,'r')
    strlines =file.readlines()
    for i in strlines: #m个循环
        strArray = i.split('][')
        strArray[0].replace('[','')
        strArray[-1].replace(']','')
        tmp_m_list = []
        for j in strArray: #V 次循环

            xyz =j.replace('[',' ')
            xyz =xyz.replace(']',' ')
            xyz = xyz.strip()
            x_y_z = xyz.split(',')
            x_y_z=list(map(float,x_y_z))
            tmp_m_list.append(x_y_z)
        tmp_m_tensor =torch.tensor(tmp_m_list)
        # print(tmp_m_tensor.shape)
        # print(len(tmp_m_tensor))
        for i in range(len(tmp_m_tensor)):#V 次循环
            i_2 =tmp_m_tensor[i] * tmp_m_tensor[i]
            i_mor = (i_2.sum()) ** 0.5
            # print(i_mor)
            for j in range(len(tmp_m_tensor)):
                j_2 = tmp_m_tensor[j] * tmp_m_tensor[j]
                j_mor = (j_2.sum()) ** 0.5
                ij_dot = tmp_m_tensor[i].mul(tmp_m_tensor[j])
                ij_dot_sum = ij_dot.sum()
                cos_ij = ij_dot_sum /(i_mor * j_mor)
                # print(cos_ij)
                tmp_cos_list.append(cos_ij)
    # print(len(tmp_cos_list))
    output = torch.tensor(tmp_cos_list)
    output = output.view(M,V,V)
    print('costensor',output.shape)
    return output







if __name__ == '__main__':

    # train_list = r'C:\Users\Xue\Documents\PycharmProject\viewgraph\smalldatasetpng\trainCollection.txt'
    # thetapath = r'C:\Users\Xue\Documents\PycharmProject\viewgraph\theta.txt'
    # modelList =getmodelname(thetapath)
    # print(modelList)
    # Thetalist =getTheta(thetapath,modelList)
    # print(Thetalist['10edit_toilet_0013'])
    # print(type(Thetalist))
    # test ='10edit_toilet_0001'
    # print(len(Thetalist[test]))
    cameraposition = r'C:\Users\Xue\Desktop\ML\3dviewgraph\datasetpath\position.txt'
    cos = caculatecos(cameraposition,M=22,V=36)
    for i in range(20):
        cos1 = cos[1]
        print(cos1.shape)


