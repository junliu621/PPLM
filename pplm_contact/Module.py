import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super(ResNet, self).__init__()
        self.channels = channels
        self.n_layers = 5
        self.dilations = [1, 2, 4, 2, 1]
        self.dropout = dropout

        self.blocks = nn.ModuleList()

        for layer in range(self.n_layers):
            block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, dilation=self.dilations[layer], padding=self.dilations[layer]),
                nn.InstanceNorm2d(channels, affine=True, track_running_stats=True),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Dropout(dropout),
                nn.Conv2d(channels, channels, kernel_size=3, dilation=self.dilations[layer], padding=self.dilations[layer]),
                nn.InstanceNorm2d(channels, affine=True, track_running_stats=True),
                nn.LeakyReLU(negative_slope=0.01, inplace=True)
            )
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            _residual = x
            x = block(x)
            x = x + _residual

        return x


class Intra_ResNet(nn.Module):
    def __init__(self, dim_1d=768+20, dim_2d=144+2+64, channels=64, dropout=0):
        super(Intra_ResNet, self).__init__()

        self.dim_1d = dim_1d
        self.dim_2d = dim_2d
        self.channels = channels
        self.dropout = dropout

        self.pre_norm_1d = nn.InstanceNorm1d(self.dim_1d)
        self.pre_norm_2d = nn.InstanceNorm2d(self.dim_2d)

        self.pair_conv1 = nn.Sequential(
            nn.Conv2d(self.dim_1d * 2, self.channels, kernel_size=1, stride=1, padding="same", dilation=1, bias=False),
            nn.InstanceNorm2d(self.channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.pair_conv2 = nn.Sequential(
            nn.Conv2d(self.dim_2d, self.channels, kernel_size=1, stride=1, padding="same", dilation=1, bias=False),
            nn.InstanceNorm2d(self.channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.pair_conv3 = nn.Sequential(
            nn.Conv2d(self.channels * 2, self.channels, kernel_size=1, stride=1, padding="same", dilation=1, bias=False),
            nn.InstanceNorm2d(self.channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.resnet = ResNet(channels=self.channels, dropout=self.dropout)

    def forward(self, x_1d, x_2d):

        x_1d = self.pre_norm_1d(x_1d)
        x_2d = self.pre_norm_2d(x_2d)

        b, d, L = x_1d.size()
        x_1d_row = x_1d.unsqueeze(-1).expand(-1, -1, L, L)
        x_1d_col = x_1d.unsqueeze(-2).expand(-1, -1, L, L)
        pair_1 = torch.cat([x_1d_row, x_1d_col], dim=1)
        pair_1 = self.pair_conv1(pair_1)
        pair_2 = self.pair_conv2(x_2d)

        pair = torch.cat([pair_1, pair_2], dim=1)
        pair = self.pair_conv3(pair)
        pair = self.resnet(pair)

        return pair


class Inter_ResNet(nn.Module):
    def __init__(self, dim_1d=768+20, dim_2d=768 + 20, channels=64, dropout=0):
        super(Inter_ResNet, self).__init__()

        self.dim_1d = dim_1d
        self.dim_2d = dim_2d
        self.channels = channels
        self.dropout = dropout

        # self.pre_norm_1d = nn.InstanceNorm1d(self.dim_1d)
        self.pre_norm_2d = nn.InstanceNorm2d(self.dim_2d)

        self.pair_conv2 = nn.Sequential(
            nn.Conv2d(self.dim_2d, self.channels, kernel_size=1, stride=1, padding="same", dilation=1, bias=False),
            nn.InstanceNorm2d(self.channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.pair_conv3 = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding="same", dilation=1, bias=False),
            nn.InstanceNorm2d(self.channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.resnet = ResNet(channels=self.channels, dropout=self.dropout)

    def forward(self, inter_2d):

        inter_2d = self.pre_norm_2d(inter_2d)
        pair = self.pair_conv2(inter_2d)
        pair = self.pair_conv3(pair)
        pair = self.resnet(pair)

        return pair


class InterTriangleMultiplication_S(nn.Module):
    def __init__(self, channel_z=64, channel_c=64, transpose=False):
        super(InterTriangleMultiplication_S, self).__init__()

        self.dz = channel_z
        self.dc = channel_c
        self.transpose = transpose

        # init norm
        self.norm_init = nn.LayerNorm(self.dz)

        # linear * gate for com_rec, com_lig
        self.Linear_inter = nn.Linear(self.dz, self.dc)
        self.gate_inter = nn.Linear(self.dz, self.dc)

        self.Linear_intra = nn.Linear(self.dz, self.dc)
        self.gate_intra = nn.Linear(self.dz, self.dc)

        # final
        self.norm_final = nn.LayerNorm(self.dc)
        self.Linear_final = nn.Linear(self.dc, self.dz)
        self.gate_final = nn.Linear(self.dz, self.dz)

    def forward(self, z_inter, z_intra, transpose=False):
        if self.transpose or transpose:
            z_inter = z_inter.permute(0, 2, 1, 3)
            z_intra = z_intra.permute(0, 2, 1, 3)

        z_inter_init = self.norm_init(z_inter)
        z_intra = self.norm_init(z_intra)

        z_inter = self.Linear_inter(z_inter_init) * self.gate_inter(z_inter_init).sigmoid()
        z_intra = self.Linear_intra(z_intra) * self.gate_intra(z_intra).sigmoid()

        z_inter_update = torch.einsum('bikc,bkjc->bijc', z_intra, z_inter)

        z_inter_update = self.norm_final(z_inter_update)
        z_inter = self.Linear_final(z_inter_update) * self.gate_final(z_inter_init).sigmoid()

        if self.transpose or transpose:
            z_inter = z_inter.permute(0, 2, 1, 3)

        return z_inter


class InterCrossAttention_S(nn.Module):
    def __init__(self, channel_z=64, num_head=4, dim_head=8, bias=True, transpose=False):
        super(InterCrossAttention_S, self).__init__()

        self.dz = channel_z
        self.dhc = dim_head
        self.num_head = num_head
        self.dc = self.num_head * self.dhc
        self.transpose = transpose
        self.bias = bias

        self.norm_init = nn.LayerNorm(self.dz)
        self.Linear_Q = nn.Linear(self.dz, self.dc)
        self.Linear_K = nn.Linear(self.dz, self.dc)
        self.Linear_V = nn.Linear(self.dz, self.dc)
        if self.bias:
            self.Linear_bias = nn.Linear(self.dz, self.num_head)

        self.softmax = nn.Softmax(dim=-1)

        self.gate_final = nn.Linear(self.dz, self.dc)
        self.Linear_final = nn.Linear(self.dc, self.dz)


    def reshape_dim(self, x):
        new_shape = x.size()[:-1] + (self.num_head, self.dhc)
        return x.view(*new_shape)

    def forward(self, z_inter, z_intra, transpose=False):
        if self.transpose or transpose:
            z_inter = z_inter.permute(0, 2, 1, 3)
            z_intra = z_intra.permute(0, 2, 1, 3)


        B, row, col, _ = z_inter.shape
        z_inter = self.norm_init(z_inter)
        z_intra = self.norm_init(z_intra)

        q = self.reshape_dim(self.Linear_Q(z_inter))
        k = self.reshape_dim(self.Linear_K(z_intra))
        v = self.reshape_dim(self.Linear_V(z_intra))

        scalar = 1.0 / torch.sqrt(torch.tensor(self.dhc, dtype=q.dtype, device=q.device))

        attn = torch.einsum('b l j h c, b l i h c -> b l j h i', q * scalar, k)

        if self.bias:
            bias = self.Linear_bias(z_inter).unsqueeze(-1)
            attn = attn + bias

        if attn.dtype is torch.bfloat16:
            with torch.cuda.amp.autocast(enabled=False):
                attn_weights = torch.nn.functional.softmax(attn, -1)
        else:
            attn_weights = torch.nn.functional.softmax(attn, -1)

        attn = torch.einsum('b l j h i, b l i h c -> b l j h c', attn_weights, v)

        gate_final = self.reshape_dim(self.gate_final(z_inter)).sigmoid()
        z_inter = (attn * gate_final).contiguous().view(gate_final.size()[:-2] + (-1,))

        z_inter = self.Linear_final(z_inter)

        if self.transpose or transpose:
            z_inter = z_inter.permute(0, 2, 1, 3)

        return z_inter


class InterSelfAttention_S(nn.Module):
    def __init__(self, channel_z=64, num_head=4, dim_head=8, bias=False, transpose=False):
        super(InterSelfAttention_S, self).__init__()

        self.dz = channel_z
        self.dhc = dim_head
        self.num_head = num_head
        self.dc = self.num_head * self.dhc
        self.transpose = transpose
        self.bias = bias

        self.norm_init = nn.LayerNorm(self.dz)
        self.Linear_Q = nn.Linear(self.dz, self.dc)
        self.Linear_K = nn.Linear(self.dz, self.dc)
        self.Linear_V = nn.Linear(self.dz, self.dc)
        if self.bias:
            self.Linear_bias = nn.Linear(self.dz, self.num_head)

        self.softmax = nn.Softmax(-1)
        self.gate_v = nn.Linear(self.dz, self.dc)
        self.Linear_final = nn.Linear(self.dc, self.dz)


    def reshape_dim(self, x):
        new_shape = x.size()[:-1] + (self.num_head, self.dhc)
        return x.view(*new_shape)

    def forward(self, z_inter, dist=None, transpose=False):
        if self.transpose or transpose:
            z_inter = z_inter.permute(0, 2, 1, 3)

        B, row, col, _ = z_inter.shape
        z_inter = self.norm_init(z_inter)

        q = self.reshape_dim(self.Linear_Q(z_inter))
        k = self.reshape_dim(self.Linear_K(z_inter))
        v = self.reshape_dim(self.Linear_V(z_inter))

        scalar = 1.0 / torch.sqrt(torch.tensor(self.dhc, dtype=q.dtype, device=q.device))

        attn = torch.einsum('b l i h c, b l j h c -> b l i h j', q * scalar, k)

        if self.bias:
            bias = self.Linear_bias(z_inter)
            # print("bias:", bias.shape, attn.shape, z_inter.shape)
            attn = attn + bias

        if dist != None:
            coef = torch.exp(-(dist/8.0)**2.0/2.0).unsqueeze(3).type_as(q)
            attn = attn * coef

        if attn.dtype is torch.bfloat16:
            with torch.cuda.amp.autocast(enabled=False):
                attn_weights = torch.nn.functional.softmax(attn, -1)
        else:
            attn_weights = torch.nn.functional.softmax(attn, -1)

        v_avg = torch.einsum('b l i h j, b l j h c -> b l i h c', attn_weights, v)

        gate_v = self.reshape_dim(self.gate_v(z_inter)).sigmoid()
        z_com = (v_avg * gate_v).contiguous().view( v_avg.size()[:-2] + (-1,) )

        z_final = self.Linear_final(z_com)

        if self.transpose or transpose:
            z_final = z_final.permute(0, 2, 1, 3)

        return z_final


class Transition(nn.Module):
    def __init__(self, channel_z=64, transition_n=4):
        super(Transition, self).__init__()

        self.dz = channel_z
        self.n = transition_n

        self.norm = nn.LayerNorm(self.dz)
        self.transition = nn.Sequential(
            nn.Linear(self.dz, self.dz*self.n),
            nn.Linear(self.dz*self.n, self.dz)
        )

    def forward(self, z_com):
        z_com = self.norm(z_com)
        z_com = self.transition(z_com)

        return z_com

