# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))                # (2708, 1)
    r_inv = np.power(rowsum, -1).flatten()      # (2708,)
    r_inv[np.isinf(r_inv)] = 0.                 # 处理除数为0导致的inf
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# def create_adj( H, W, C, neibour):
# 	"""
# 	功能：
# 		根据featuremap的高和宽建立对应的空域邻接矩阵,
# 	输入：
# 		h featuremap的高度
# 		w featuremap的宽
# 		C featuremap的通道数 
# 		neibour  4或8决定空域adj的邻居数   2 决定计算channel的adj


# 	"""
# 	h = H
# 	w = W
# 	n = h*w
# 	x = [] #保存节点
# 	y = [] #保存对应的邻居节点
# 	#判断是生成8邻居还是4邻居
# 	if neibour==8:
# 		for i in range(n):
# 			# 分为5类第一行，最后一行，第一列，最后一列，内部
# 			if i<w:                  #第一行
# 				if (i-1)>=0: #左邻
# 					x.append(i)
# 					y.append(i-1)
# 				if (i+1)<w: #右邻
# 					x.append(i)
# 					y.append(i+1)
# 				if i+w-1>w-1: #左下邻
# 					x.append(i)
# 					y.append(i+w-1) 
# 				if i+w+1<2*w: #右下邻
# 					x.append(i)
# 					y.append(i+w+1)                        
# 				x.append(i) #下邻
# 				y.append(i+w)
# 			elif i%w==0:               #第一列
# 				if (i-w)>=0: #上邻
# 					x.append(i)
# 					y.append(i-w)
# 				if (i-w+1)>0: #右上邻
# 					x.append(i)
# 					y.append(i-w+1)
# 				if (i+w)<=(h-1)*w: #下邻
# 					x.append(i)
# 					y.append(i+w)
# 				if (i+w+1)<=(h-1)*w+1: #右下邻
# 					x.append(i)
# 					y.append(i+w+1)
# 				x.append(i) #右邻
# 				y.append(i+1)
# 			elif (i+1)%w==0:        #最后一列
# 				if (i-w)>=0: #上邻
# 					x.append(i)
# 					y.append(i-w)
# 				if (i-w-1)>=0: #左上邻
# 					x.append(i)
# 					y.append(i-w-1)
# 				if (i+w)<n:  #下邻
# 					x.append(i)
# 					y.append(i+w)
# 				if (i+w-1)<n:  #左下邻
# 					x.append(i)
# 					y.append(i+w-1)
# 				x.append(i) #左邻
# 				y.append(i-1)               
# 			elif i>=n-w:            #最后一行的数
# 				if (i-1)>=n-w: #左邻
# 					x.append(i)
# 					y.append(i-1)
# 				if (i-w-1)>=n-2*w: #左上邻
# 					x.append(i)
# 					y.append(i-w-1)
# 				if (i+1)<n: #右邻
# 					x.append(i)
# 					y.append(i+1)
# 				if (i-w+1)<n-w: #右上邻
# 					x.append(i)
# 					y.append(i-w+1)
# 				x.append(i) #上邻
# 				y.append(i-w)  
# 			else:                   #内部的数
# 				x+=[i,i,i,i,i,i,i,i] #上邻
# 				y+=[i-w-1,i-w,i-w+1,i-1,i+1,i+w-1,i+w,i+w+1]
# 	elif neibour==4:       #4邻居
# 		for i in range(n):
# 			# 分为5类第一行，最后一行，第一列，最后一列，内部
# 			if i<w:                  #第一行
# 				if (i-1)>=0: #左邻
# 					x.append(i)
# 					y.append(i-1)
# 				if (i+1)<w: #右邻
# 					x.append(i)
# 					y.append(i+1)
# 				x.append(i) #下邻
# 				y.append(i+w)
# 			elif i%w==0:               #第一列
# 				if (i-w)>=0: #上邻
# 					x.append(i)
# 					y.append(i-w)
# 				if (i+w)<=(h-1)*w: #下邻
# 					x.append(i)
# 					y.append(i+w)
# 				x.append(i) #右邻
# 				y.append(i+1)
# 			elif (i+1)%w==0:        #最后一列
# 				if (i-w)>=0: #上邻
# 					x.append(i)
# 					y.append(i-w)
# 				if (i+w)<n:  #下邻
# 					x.append(i)
# 					y.append(i+w)
# 				x.append(i) #左邻
# 				y.append(i-1)               
# 			elif i>=n-w:            #最后一行的数
# 				if (i-1)>=n-w: #左邻
# 					x.append(i)
# 					y.append(i-1)
# 				if (i+1)<n: #右邻
# 					x.append(i)
# 					y.append(i+1)
# 				x.append(i) #上邻
# 				y.append(i-w)  
# 			else:                   #内部的数
# 				x+=[i,i,i,i] #上邻
# 				y+=[i-w,i-1,i+1,i+w]
# 	elif neibour==2: #2邻居是对channel进行计算
# 		n = C
# 		l =np.arange(n)

# 		#每个元素的上一个邻居
# 		x = np.append(x,l[1:]).astype(int) #0没有上一个邻居
# 		y = np.append(y,(l-1)[1:]).astype(int)
# 		#每个元素的下一个邻居
# 		x = np.append(x,l[:-1]).astype(int) #最后一个没有下一个邻居
# 		y = np.append(y,(l+1)[:-1]).astype(int) 
# 	adj = np.array((x,y)).T  #生成的两列合并得到节点及其邻居的矩阵
# 	#print(adj)
# 	#使用sp.coo_matrix() 和 np.ones() 共同生成临界矩阵，右边的

# 	adj = sp.coo_matrix((np.ones(adj.shape[0]), (adj[:, 0], adj[:, 1])),shape=(n, n),dtype=np.float32)

# 	# build symmetric adjacency matrix 堆成矩阵
# 	adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
# 	#adj = np.hstack((x,y)).rehsape(-1,2) 这样reshape得到的是临近两个一组变化成两列，不符合条件
# 	adj =normalize( adj + sp.eye(adj.shape[0]))
# 	adj = np.array(adj.todense())
# 	''''      保存adj的数据查看是什么形状      	'''
# 	# np.save('./adj.txt',x) 
# 	adj = torch.tensor(adj).cuda()

# 	#adj = torch.FloatTensor(x).cuda()
# 	return adj

def create_adj( H, W, C, neibour):
    """
    功能：
        根据featuremap的高和宽建立对应的空域邻接矩阵,
    输入：
        h featuremap的高度
        w featuremap的宽
        C featuremap的通道数 
        neibour  4或8决定空域adj的邻居数   2 决定计算channel的adj
    """
    h = H
    w = W
    n = h*w
    x = [] #保存节点
    y = [] #保存对应的邻居节点
    #判断是生成8邻居还是4邻居
    if neibour==8:
        l =np.reshape(np.arange(n),(h,w))
        # print(l)
        # print(((l[:,2])+w)[:1])
        #print(l[:,2])
        for i in range(h): 
            #邻界条件需要考虑，故掐头去尾先做中间再两边
            r = l[i,:]
            #左邻
            x = np.append(x,r[1:]).astype(int) #0没有上一个邻居
            y = np.append(y,(r-1)[1:]).astype(int)
            #每个元素的右邻居
            x = np.append(x,r[:-1]).astype(int) #最后一个没有下一个邻居
            y = np.append(y,(r+1)[:-1]).astype(int) 
            if i >0:
                #上邻
                x = np.append(x,r).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r-w)).astype(int) 
                #左上
                x = np.append(x,r[1:]).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r-w-1)[1:]).astype(int) 
                #右上
                x = np.append(x,r[:-1]).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r-w+1)[:-1]).astype(int) 
            if i <h-1:
                #下邻
                x = np.append(x,r).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r+w)).astype(int) 
                #左下
                x = np.append(x,r[1:]).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r+w-1)[1:]).astype(int) 
                #右上
                x = np.append(x,r[:-1]).astype(int) #最后一个没有下一个邻居
                y = np.append(y,(r+w+1)[:-1]).astype(int)                           
    elif neibour==4:       #4邻居
        l =np.reshape(np.arange(n),(h,w))
        for i in range(h): 
            v = l[i,:]
            x = np.append(x,v[1:]).astype(int) #0没有上一个邻居
            y = np.append(y,(v-1)[1:]).astype(int)
            #每个元素的右邻居
            x = np.append(x,v[:-1]).astype(int) #最后一个没有下一个邻居
            y = np.append(y,(v+1)[:-1]).astype(int) 

        for i in range(w):
            p = l[:,i]
            #上邻
            x = np.append(x,p[1:]).astype(int) #0没有上一个邻居
            y = np.append(y,(p-w)[1:]).astype(int)
            #下邻  
            x = np.append(x,p[:-1]).astype(int) #0没有上一个邻居
            y = np.append(y,(p+w)[:-1]).astype(int)
    elif neibour==2:       #4邻居
        n = C
        l =np.arange(n)

        #每个元素的上一个邻居
        x = np.append(x,l[1:]).astype(int) #0没有上一个邻居
        y = np.append(y,(l-1)[1:]).astype(int)
        #每个元素的下一个邻居
        x = np.append(x,l[:-1]).astype(int) #最后一个没有下一个邻居
        y = np.append(y,(l+1)[:-1]).astype(int) 
    adj = np.array((x,y)).T  #生成的两列合并得到节点及其邻居的矩阵
    #print(adj)
    #使用sp.coo_matrix() 和 np.ones() 共同生成临界矩阵，右边的

    adj = sp.coo_matrix((np.ones(adj.shape[0]), (adj[:, 0], adj[:, 1])),shape=(n, n),dtype=np.float32)

    # build symmetric adjacency matrix 堆成矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #adj = np.hstack((x,y)).rehsape(-1,2) 这样reshape得到的是临近两个一组变化成两列，不符合条件
    #adj =normalize( adj + sp.eye(adj.shape[0]))
    adj = np.array(adj.todense())
    ''''      保存adj的数据查看是什么形状      	'''
    # np.save('./adj.txt',x) 
    adj = torch.tensor(adj).cuda()

    #adj = torch.FloatTensor(x).cuda()
    return adj


class GraphAttentionLayer(nn.Module):
	"""
	描述：
		再MPGA中，单层的GAT，输入输出的维度相同，attention的计算方式使用softmax
		in_features：输入的维度，
		down_ratio:降维的比例
		out_feature:在多头注意力之中需要用
	"""
	def __init__(self, in_features,down_ratio=8,sgat_on=True,cgat_on=True):
		super(GraphAttentionLayer, self).__init__()
		#self.dropout = dropout
		self.in_features = in_features
		self.hid_features = in_features//down_ratio #数据降维，使用//保证输出的结果为整数
		#alpha sigma两次降维后，做矩阵运算获得att，类似GAT中的先用w再用a获得注意力。
		self.use_sgat = sgat_on
		self.use_cgat = cgat_on
		if self.use_sgat:
			self.down_alpha = nn.Sequential( #无论是通道还是空域都需要先降维 图卷积的#默认使用hid_feature是
				nn.Conv2d(in_channels=self.in_features, out_channels=self.hid_features, 
						kernel_size=1, stride=1, padding=0, bias=False), #输入 in_features 2048 输出hid_features  in_features//down_ratio
				nn.BatchNorm2d(self.hid_features),
				nn.ReLU()
			)

			self.down_sigma = nn.Sequential( #无论是通道还是空域都需要先降维 图卷积的
				nn.Conv2d(in_channels=self.in_features, out_channels=self.hid_features, 
						kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(self.hid_features),
				nn.ReLU()
			)
	

	def forward(self, x):#v 是输入的各个节点， b c h w是输入feature map的shape
		#输入后都是先降维计算注意力，concat=false需要聚合特征前需要将输入的维度降低再聚合
		b,c,h,w = x.size()

		if self.use_sgat:
			adj = create_adj(h,w,self.in_features,8)
			#print('图片的维度：',x.size())
			alpha = self.down_alpha(x)#concat的时候不太一样
			#print('alpha :',alpha.shape)
			#print('alpha.shape:',alpha.shape)
			sigma = self.down_sigma(x)
			#print('sigma :',sigma.shape)
			alpha = alpha.view(b, self.hid_features, -1).permute(0, 2, 1) #8 32 64*32	
			#print('转换后alpha :',alpha.shape)
			sigma = sigma.view(b, self.hid_features, -1)
			#print('转换后sigma :',sigma.shape)
			att = torch.matmul(alpha, sigma) #这就是每个图的自注意力机制
			#print('alpha乘sigma得到大的att shape:',att.shape)
			zero_vec = -9e15*torch.ones_like(att)
			attention = torch.where(adj.expand_as(att)> 0, att,zero_vec)
			#print('attention shape:',attention.shape)
			attention = F.softmax(attention, dim=2)  # 计算节点，临近节点的关系，因为attention为3维 b h w 按行求则为
			#print('softmax(attention) shape:',attention.shape)
			h_s = torch.matmul(attention, x.view(b, c, -1).permute(0, 2, 1)).permute(0,2,1).view(b,c,h,w)  #聚合临近节点的信息表示该节点
			#print('图上传播后',h_prime.shape)
		if self.use_cgat:
			cadj = create_adj(h,w,c,2)#2表示通道的adj未进行节点维度的变化，直接点乘和sigmod计算的att
			theta_xc = x.view(b, c, -1)
			phi_xc = x.view(b, c, -1).permute(0, 2, 1) # 8 2048 256 1  batch_size 节点 channel
			Gc = torch.matmul(theta_xc, phi_xc) # bactchsiz n n   通道之间的关系 
			zero_vec = -9e15*torch.ones_like(Gc)
			catt = torch.where(cadj.expand_as(Gc)> 0, Gc, zero_vec)
			cattention = F.softmax(catt, dim=2)  # 计算节点，临近节点的关系，因为attention为3维 b h w 按行求则为
			h_prime = torch.matmul(cattention, x.view(b, c, -1)).view(b,c,h,w)   #聚合临近节点的信息表示该节点
		if self.use_cgat and self.use_sgat:
			return torch.add(h_s, h_prime)
		if self.use_cgat:
			return h_prime #残差

		return  h_s


class GAT(nn.Module):
    def __init__(self, nfeat, nclass, nheads,down_ratio=8,sgat_on=True,cgat_on=True):
        """Dense version of GAT.
        描述：
            nfeat :输入的维度
            nclass ：非concat时使用的输出维度
            nheads：几头
            Height Width：输入的图片的大小
            dow_ratio:concat时输入数据的维度降低倍数
        """
        super(GAT, self).__init__()
        
        print ('Use_SGAT_Att: {};\tUse_CGAT_Att: {}.'.format(sgat_on, cgat_on))

        self.heads = nheads
        self.attentions= nn.ModuleList([GraphAttentionLayer(nfeat,down_ratio=down_ratio,sgat_on=sgat_on,cgat_on=cgat_on) for _ in range(nheads)])
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        
        self.gama = nn.Parameter(torch.zeros(1))

        self.output_down = nn.Sequential( 
        nn.Conv2d(in_channels=nfeat*nheads, out_channels=nfeat, 
                kernel_size=1, stride=1, padding=0, bias=False), #输入 in_features 2048 输出hid_features  in_features//down_ratio
        nn.BatchNorm2d(nfeat),
		nn.ReLU()

		)

    def forward(self, x):#resnet输入的维度为 b 2048 16 8 
        b,c,h,w = x.size()
        #x = F.dropout(x, self.dropout, training=self.training)
        if  self.heads > 1 :
            h_prime = torch.cat([att(x) for att in self.attentions], dim=1)  #通道直接拼接上还是 如果维度变为1024 则最终 b n 4096 考虑
            h_prime =  self.output_down(h_prime)#到这一步已经算是进行了两次的GCN  在这一部分中再次将维度恢复到2048
        else:
            h_prime =self.attention_0(x)
        #x = F.dropout(x, self.dropout, training=self.training) 
        h_prime = F.elu(h_prime) #因为输入x是经果relu的所有这里也需要经过这个然后输入res中
        h_prime = (1-self.gama)*x+self.gama*h_prime #残差
        return h_prime 


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PGA_resnte(nn.Module):
	def __init__(self, last_stride=1,nclass=751,block=Bottleneck, layers=[3, 4, 6, 3],sgat_on=True,cgat_on=True,nheads=1,model_name=None):
		self.inplanes = 64
		super().__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
								bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		# self.relu = nn.ReLU(inplace=True)   # add missed relu
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
		self.gat11 = GAT(256,nclass=nclass,nheads=nheads, sgat_on=sgat_on,cgat_on=cgat_on)
		self.gat12 = GAT(256,nclass=nclass,nheads=nheads, sgat_on=sgat_on,cgat_on=cgat_on)
		self.gat13 = GAT(256,nclass=nclass,nheads=nheads, sgat_on=sgat_on,cgat_on=cgat_on)

		self.gat21 = GAT(512,nclass=nclass,nheads=nheads, sgat_on=sgat_on,cgat_on=cgat_on)
		self.gat22 = GAT(512,nclass=nclass,nheads=nheads, sgat_on=sgat_on,cgat_on=cgat_on)
		self.gat23 = GAT(512,nclass=nclass,nheads=nheads, sgat_on=sgat_on,cgat_on=cgat_on)

		self.gat31 = GAT(1024,nclass=nclass,nheads=nheads, sgat_on=sgat_on,cgat_on=cgat_on)
		self.gat32 = GAT(1024,nclass=nclass,nheads=nheads, sgat_on=sgat_on,cgat_on=cgat_on)
		self.gat33 = GAT(1024,nclass=nclass,nheads=nheads, sgat_on=sgat_on,cgat_on=cgat_on)

		self.gat41 = GAT(2048,nclass=nclass,nheads=nheads, sgat_on=sgat_on,cgat_on=cgat_on)
		self.gat42 = GAT(2048,nclass=nclass,nheads=nheads, sgat_on=sgat_on,cgat_on=cgat_on)
		self.gat43 = GAT(2048,nclass=nclass,nheads=nheads, sgat_on=sgat_on,cgat_on=cgat_on)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
							kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		# x = self.relu(x)    # add missed relu
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.gat11(x)#[64,256,h,w]
		x = self.gat12(x)#[64,256,h,w]
		x = self.gat13(x)#[64,256,h,w]

		x = self.layer2(x)
		x = self.gat21(x)#[64,512,h,w]
		x = self.gat22(x)#[64,512,h,w]
		x = self.gat23(x)#[64,512,h,w]

		x = self.layer3(x)
		x = self.gat31(x)#[64,1024,h,w]
		x = self.gat32(x)#[64,1024,h,w]
		x = self.gat33(x)#[64,1024,h,w]

		x = self.layer4(x)
		x = self.gat41(x)#[64,2048,h,w]
		x = self.gat42(x)#[64,2048,h,w]
		x = self.gat43(x)#[64,2048,h,w]

		return x

	def load_param(self, model_path):
		param_dict = torch.load(model_path)
		for i in param_dict:
			if 'fc' in i:
				continue
			self.state_dict()[i].copy_(param_dict[i])
            
	def random_init(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

