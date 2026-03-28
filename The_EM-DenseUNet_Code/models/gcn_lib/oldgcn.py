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
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()            
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y, relative_pos)
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()

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
        print(x.shape,1)
        x = self.fc1(x)
        B, C, H, W = x.shape

        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x, relative_pos)        
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp
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

        # Define the Grapher layers with the new channel sizes
        self.gcb4 = nn.Sequential(
            Grapher(channels[0], self.num_knn[0], min(0 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                    self.bias, False, 0.2, self.reduce_ratios[0], n=self.HW // (4 * 4 * 4), drop_path=self.dpr[0], relative_pos=True, padding=self.padding),
        )

    def forward(self, x, skips):
        # Grapher forward pass
        # print(x.shape)
        d4 = self.gcb4(x)
        # print(d4.shape)
        return d4

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

    # # Simulate input tensor (batch size of 4 and 3-channel input image with 256x256 resolution)
    # channels = torch.randn(4, 3, 256, 256)

    # # Pass through the backbone to get the feature maps
    # x1, x2, x3, x4 = backbone(channels)

    # Print the output shapes from the backbone
    # print('x4:', x4.shape, 'x3:', x3.shape, 'x2:', x2.shape, 'x1:', x1.shape)# x1 [4, 96, 64, 64] 
    # x2 [4, 192, 32, 32]) 
    # x3 [4, 384, 16, 16] 
    # x4 [4, 768, 8, 8] 
    x4 = torch.randn(4, 768, 8, 8)
    x3 = torch.randn(4, 384, 16, 16)
    x2 = torch.randn(4, 192, 32, 32)
    x1 = torch.randn(4, 96, 64, 64)
    
    # Pass the output feature map x4 and skip connections (x3, x2, x1) to the decoder
    _ = decoder(x4, [x3, x2, x1])
    # d4torch.Size([4, 768, 8, 8])