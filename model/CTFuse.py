import torch
from torch import nn
import torch.nn.functional as F

#conv_attention block
def drop_path_f(x, drop_prob: float = 0.1, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize得到下界值
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class Mlp_c(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Spatial_Channel_Attn_Cnn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1)
        self.avg_pool=nn.AdaptiveAvgPool2d((1,1))
        # self.avg_pool=F.adaptive_avg_pool2d(output_size=(1,1))
        self.conv1=nn.Conv2d(dim,1,1)
        self.conv2=nn.Conv2d(dim,dim,1)
        self.act=nn.Sigmoid()
        self.norm=nn.BatchNorm2d(dim)
        self.relu=nn.ReLU()


    def forward(self, x):
        shortcut=x.clone()
        x=self.conv0(x)
        x1=self.avg_pool(x)
        x2=self.conv1(x)
        x3=x1*x2
        x3=self.conv2(x3)
        x=x*x3
        x=self.act(x)
        x=x+shortcut
        x=self.relu(self.norm(x))
        return x




class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x



class Block_c(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU
                 ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_c(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.sc_attn_cnn = Spatial_Channel_Attn_Cnn(dim)

    def forward(self, x):
        # B, N, C = x.shape
        # x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * (self.attn(self.norm1(x))+self.sc_attn_cnn(self.norm1(x))))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))
        # x = x.view(B, C, N).permute(0, 2, 1)
        return x

#vit block
class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        #Patch Merging先将图片concat成四个channel，然后再Linear成两个channel
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):#传入输入的数据，当前特征层的高度和宽度
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            #pad是从最后一个维度向前设置的
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]，-1代表最后一个维度
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]，-1代表从其他维度推断

        x = self.norm(x)#通过layer_norm进行处理
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]#刚刚创建的全连接层将4*c映射为2*c

        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block_t(nn.Module):
    def __init__(self,
                 dim,
                 L,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.1,
                 drop_path_ratio=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block_t, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.sc_attn_t=Spatial_Channel_Attn_Trans(dim,L)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))+self.sc_attn_t(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class attentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            Block_c(dim=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class attentionTrans(nn.Module):
    def __init__(self, in_channels,head,L):
        super().__init__()
        self.double_conv = Block_t(dim=in_channels,num_heads=head,L=L)

    def forward(self, x):
        return self.double_conv(x)

class Down_c(nn.Module):
    def __init__(self, in_channels, out_channels,depth=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for i_layer in range(depth):
            layer = attentionConv(out_channels, out_channels)
            self.layers.append(layer)

    def forward(self, x):
        x=self.maxpool_conv(x)
        for layer in self.layers:
            x=x+layer(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down_t(nn.Module):
    def __init__(self, embed_dim=768, head=6, down_ratio=1,depth=2,L=384):
        super().__init__()
        self.patch_merging=PatchMerging(embed_dim//2)
        self.pos_embed=nn.Parameter(torch.zeros(1, 256*256//down_ratio//down_ratio, embed_dim))
        self.layers = nn.ModuleList()
        for i_layer in range(depth):
            layer = attentionTrans(embed_dim,head,L)
            self.layers.append(layer)
        self.norm=nn.LayerNorm(embed_dim)

    def forward(self, x, H,W):
        x=self.patch_merging(x,H,W)
        for layer in self.layers:
            x=x+layer(x)
        x=self.norm(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Spatial_Channel_Attn_Trans(nn.Module):
    def __init__(self, dim, L):
        super().__init__()
        self.linear_c1=nn.Linear(dim,dim)
        self.linear_c2=nn.Linear(dim,dim)
        self.linear1=nn.Linear(dim,dim)
        self.linear_l1=nn.Linear(L,L)
        self.linear_l2=nn.Linear(L,L)
        self.linear2 = nn.Linear(L, L)
        self.proj=nn.Linear(dim,dim)
        self.norm=nn.LayerNorm(dim)

    def forward(self, x):
        B,L,C=x.shape
        shortcut=x.clone()
        x_BLC1=self.linear_c1(x)
        x_BLC2=self.linear_c2(x)
        x_BCL=x.permute(0,2,1).contiguous()
        x_BCL1=self.linear_l1(x_BCL)
        x_BCL2=self.linear_l2(x_BCL)
        map_l=x_BLC1@x_BLC2.transpose(-2,-1)
        map_c=x_BCL1@x_BCL2.transpose(-2,-1)
        x=x@map_c
        x=self.linear1(x)
        x=x.permute(0,2,1)@map_l
        x=self.linear2(x)
        x = x.softmax(dim=-1)
        x=x.permute(0,2,1).contiguous()
        x = x.softmax(dim=-1)
        x=shortcut+x
        x=self.norm(x)
        return x

class CTFuse(nn.Module):
    def __init__(self, n_classes, n_channels=3, dim=48,bilinear=True):
        super(CTFuse, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, dim)
        self.fuse=attentionConv(dim,dim)
        self.down1 = Down_c(dim, 2*dim)
        self.down2 = Down_c(2*dim, 4*dim)
        self.pos_embed=nn.Parameter(torch.zeros(1, 256//4*256//4, dim*4))
        self.pos_embed1=nn.Parameter(torch.zeros(1, 256//8*256//8, dim*8))
        self.down3 = Down_t(8*dim,8,8,depth=2,L=32*32)
        factor = 2 if bilinear else 1
        self.down4 = Down_t(16*dim,16,16,L=16*16)
        #self.sc_attn4=SC_attn(dim=16*dim,L=256)
        self.ln=nn.Linear(dim*16,dim*8)
        self.up1 = Up(16*dim, 8*dim // factor, bilinear)
        self.up2 = Up(8*dim, 4*dim// factor, bilinear)
        self.up3 = Up(4*dim, 2*dim // factor, bilinear)
        self.up4 = Up(2*dim, dim, bilinear)
        #self.dropout=nn.Dropout(0.1)
        self.outc = OutConv(dim, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        xr=self.fuse(x1)
        x1=x1+xr
        shortcut1=x1.clone()
        x2 = self.down1(x1)
        shortcut2=x2.clone()
        x3 = self.down2(x2)
        shortcut3=x3.clone()
        x3=x3.flatten(2).permute(0,2,1).contiguous()
        x3=x3+self.pos_embed
        x4 = self.down3(x3,256//4,256//4)
        shortcut4=x4.permute(0,2,1).reshape(x4.shape[0],-1,256//8,256//8)
        x4=x4+self.pos_embed1
        x5 = self.down4(x4,256//8,256//8)
        #x5=self.sc_attn4(x5)
        x5=self.ln(x5)
        x5=x5.permute(0,2,1).reshape(x5.shape[0],-1,256//16,256//16)
        x = self.up1(x5, shortcut4)
        x = self.up2(x, shortcut3)
        x = self.up3(x, shortcut2)
        x = self.up4(x, shortcut1)
        #x= self.dropout(x)
        logits = self.outc(x)
        return {'out':logits}