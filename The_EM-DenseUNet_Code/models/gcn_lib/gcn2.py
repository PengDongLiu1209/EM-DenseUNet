import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F
from timm.layers import DropPath
# from .pvtv2 import pvt_v2_b2



class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, kernel_size=1, padding=0, groups=4):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias, kernel_size=1, padding=0, groups=4)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        print(f"After MRConv2d: {x.shape}")  # 打印MRConv2d的输出特征形状
        return self.nn(x)
    
class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True, kernel_size=1, padding=0, groups=4):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias, kernel_size=kernel_size, padding=padding, groups=groups)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)

        
class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1, padding=4, groups=4):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias, kernel_size=kernel_size, padding=padding, groups=groups)
        
        self.k = kernel_size
        self.d = dilation
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        print(f"Input to DyGraphConv2d: {x.shape}")  # 打印输入特征形状
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
        print(f"After pooling and reshaping: {x.shape}, y: {y.shape if y is not None else 'None'}")  # 打印pooling后的特征形状

        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        print(f"Edge index shape: {edge_index.shape}")  # 打印邻域图形状

        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        x = x.reshape(B, -1, H, W).contiguous()
        print(f"Output of DyGraphConv2d: {x.shape}")  # 打印卷积后的输出特征形状
        return x

class Grapher(nn.Module):
    """
    Grapher with graph convolution and fc layers
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False, padding=4):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.n = n
        self.r = r
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r, padding=padding, groups=in_channels) #in_channels * 2
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None
        if relative_pos:
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x
        print(f"Input to Grapher: {x.shape}")  # 打印输入特征形状
        x = self.fc1(x)
        B, C, H, W = x.shape
        print(f"After fc1 in Grapher: {x.shape}")  # 打印经过fc1后的特征形状

        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)
        print(f"After graph_conv in Grapher: {x.shape}")  # 打印经过graph_conv后的特征形状

        x = self.fc2(x)
        print(f"After fc2 in Grapher: {x.shape}")  # 打印经过fc2后的特征形状

        x = self.drop_path(x) + _tmp
        print(f"Output of Grapher: {x.shape}")  # 打印最终输出特征形状
        return x

class GCUP(nn.Module):
    def __init__(self, channels=[768, 384, 192, 96], img_size=224, drop_path_rate=0.0, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu'):
        super(GCUP, self).__init__()
        
        # Graph convolution block (GCB) parameters
        self.padding = padding
        self.k = k  # neighbor num (default: 9)
        self.conv = conv  # graph conv layer {edge, mr, sage, gin}, default is 'mr'
        self.gcb_act = gcb_act  # activation layer for graph convolution block
        self.gcb_norm = 'batch'  # normalization for GCB
        self.bias = True  # bias for conv layers
        self.drop_path = drop_path_rate  # dropout rate
        self.reduce_ratios = [1, 1, 4, 2]
        self.dpr = [drop_path_rate] * 4  # drop path rate
        self.num_knn = [self.k] * 4  # number of knn's k
        self.max_dilation = 18 // max(self.num_knn)
        self.HW = img_size // 4 * img_size // 4
        self.use_dilation = True
        self.use_stochastic = False # stochastic for gcn, True or False
        self.epsilon = 0.2 # stochastic epsilon for gcn
        
        # Define the Grapher layers with the new channel sizes
        self.gcb4 = nn.Sequential(Grapher(channels[0], self.num_knn[0], min(0 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[0], n=self.HW//(4*4*4), drop_path=self.dpr[0],
                                    relative_pos=True, padding=self.padding),
        )
	
        self.gcb3 = nn.Sequential(Grapher(channels[1], self.num_knn[1], min(3 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[1], n=self.HW//(4*4), drop_path=self.dpr[1],
                                    relative_pos=True, padding=self.padding),
        )

        self.gcb2 = nn.Sequential(Grapher(channels[2], self.num_knn[2], min(8 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[2], n=self.HW//(4), drop_path=self.dpr[2],
                                    relative_pos=True, padding=self.padding),
        )
        self.gcb1 = nn.Sequential(Grapher(channels[3], self.num_knn[3],  min(11 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                    self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[3], n=self.HW, drop_path=self.dpr[3],
                                    relative_pos=True, padding=self.padding),
        )


    def forward(self, x, k):
        # Grapher forward pass
        # print(x.shape)
        if (k==1):
            d = self.gcb1(x)
        elif (k==2):
            d = self.gcb2(x)
        elif (k==3):
            d = self.gcb3(x)
        elif (k==4):
            d = self.gcb4(x)           

        print(d.shape)
        return d

if __name__ == '__main__':
    # Initialize the backbone model and load the pre-trained weights

    # backbone = pvt_v2_b2()  # Backbone model with updated architecture
    # path = './pvt_v2_b2.pth'
    # save_model = torch.load(path)
    # model_dict = backbone.state_dict()
    # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # backbone.load_state_dict(model_dict)

    # Initialize the decoder model
    decoder = GCUP(channels=[768, 384, 192, 96])  # Updated channels

    x4 = torch.randn(4, 768, 8, 8)
    x3 = torch.randn(4, 384, 16, 16)
    x2 = torch.randn(4, 192, 32, 32)
    x1 = torch.randn(4, 96, 64, 64)

    gbc1=GCUP(x1,1) 
    gbc2=GCUP(x2,2) 
    gbc3=GCUP(x3,3) 
    gbc4=GCUP(x4,4) 
    print("gbc4 shape:", gbc4.shape)# torch.Size([4, 768, 8, 8])
    print("gbc3 shape:", gbc3.shape)# torch.Size([4, 384, 16, 16])
    print("gbc1 shape:", gbc1.shape)# torch.Size([4, 192, 32, 32])
    print("gbc2 shape:", gbc2.shape)# torch.Size([4, 96, 64, 64])




        
        
        
        