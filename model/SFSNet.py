import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import timm

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class FFT_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FFT_Conv, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2,1)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.act = nn.GELU() 

    def forward(self, x):
        batch, _, h, w = x.size()

        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.conv_layer(ffted) 
        ffted = self.act(self.bn(ffted))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        return output

class FourierUnit(nn.Module):
    def __init__(self, dim,scale):
        super().__init__()

        self.act = nn.GELU()
        self.fft_conv = FFT_Conv(dim, dim)
        self.scale = scale

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.ca_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_max_pool = nn.AdaptiveMaxPool2d(1)
        self.ca_mlp = nn.Sequential(
            nn.Conv2d(dim, dim//4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim//4, dim, kernel_size=1),
        )
        self.ca_sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.act(self.spatial_fft(x, self.scale))
        x = self.fft_conv(x)
        x = self.act(self.channel_fft(x, self.scale))
        return x
    
    def spatial_fft(self, x, scale):
        assert scale > 2
        x = torch.fft.fft2(x, norm="ortho") 

        x = torch.fft.fftshift(x, dim=[-2, -1])
        _, _, h, w = x.shape  
        x[:,:,h // 2 - h // scale:h // 2 + h // scale,w // 2 - w // scale:w // 2 + w // scale,] = 0.0  
        x = torch.fft.ifftshift(x, dim=[-2, -1])

        mag = torch.abs(x)                      
        avg_out = mag.mean(dim=1, keepdim=True)       
        max_out, _ = mag.max(dim=1, keepdim=True)     
        sa_in = torch.cat([avg_out, max_out], dim=1)  
        attn = self.spatial_attn(sa_in)               
        x = x * attn                       

        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = self.act(x)
        return x

    def channel_fft(self, x, scale):
        assert scale > 2
        x = torch.fft.fft(x, dim=1, norm="ortho")
        x = torch.fft.fftshift(x, dim=1)
        _, c, _, w = x.shape
        x[:, c // 2 - c // scale:c // 2 + c // scale, :, :] = 0.0
        x = torch.fft.ifftshift(x, dim=1)

        mag = torch.abs(x)                                  
        avg_pool = self.ca_avg_pool(mag)                    
        max_pool = self.ca_max_pool(mag)                   
        ca = self.ca_mlp(avg_pool) + self.ca_mlp(max_pool)  
        w = self.ca_sigmoid(ca)                             
        x = x * w               

        x = torch.fft.ifft(x, dim=1, norm="ortho")
        x = torch.real(x)
        x = self.act(x)

        return x

class NonLocalBlock(nn.Module):
    def __init__(self, planes):
        super(NonLocalBlock, self).__init__()
        self.query_conv = nn.Conv2d(planes, planes, kernel_size=1)
        self.key_conv = nn.Conv2d(planes, planes, kernel_size=1)
        self.value_conv = nn.Conv2d(planes, planes, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.ln = LayerNorm(planes, eps=1e-6, data_format="channels_first")
        self.conv = nn.Sequential(
            nn.Conv2d(planes,planes,3,padding=1),
            nn.BatchNorm2d(planes),
            nn.GELU()  
        )
        self.low_pass = ALPF(64,3,group=8)
        
    def forward(self, x):
        _, _, h, w = x.size()
        shorcut = x.clone()
        x = self.ln(x)

        proj_query = self.low_pass(self.query_conv(x))
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        q = rearrange(proj_query,'b (head c) h w -> b head h (w c)',head = 8)
        k = rearrange(proj_key,'b (head c) h w -> b head h (w c)',head = 8)
        v = rearrange(proj_value,'b (head c) h w -> b head h (w c)',head = 8)
        q = torch.nn.functional.normalize(q,dim=-1)
        k = torch.nn.functional.normalize(k,dim=-1)

        d_k = q.size(-1)  
        scale = 1.0 / (d_k ** 0.5)

        attn = (q @ k.transpose(-2,-1)) * scale
        attn = self.softmax(attn)
        out = attn @ v

        out = rearrange(out,'b head h (w c) -> b (head c) h w', head = 8 ,h = h, w = w)
        out = self.conv(out) + shorcut

        return out

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

class ALPF(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect', group=2):
        super(ALPF, self).__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size, kernel_size=kernel_size, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size*kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)

        n,c,h,w = sigma.shape

        sigma = sigma.reshape(n,1,c,h*w)

        n,c,h,w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))

        n,c1,p,q = x.shape
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)

        n,c2,p,q = sigma.shape
        sigma = sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size,n,c2,q)).permute(2,0,3,1,4)

        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)
        return x

class SCEAM(nn.Module):
    def __init__(self, dim,scale):
        super(SCEAM, self).__init__()
        self.low = NonLocalBlock(dim)
        self.high = FourierUnit(dim,scale)
        self.fusion = ConvBNReLU(dim,dim,3)

    def forward(self, x):    
        low = self.low(x)
        high = self.high(x)
        fusion = self.fusion(low + high)

        return fusion

class Block(nn.Module):
    def __init__(self, dim=256, mlp_ratio=4., drop=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, scale = 4):
        super().__init__()

        self.attn = SCEAM(dim,scale = scale)
        self.norm1 = norm_layer(dim)

        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)


    def forward(self, x):

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class CAIF(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CAIF, self).__init__()

        self.stdpool = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.stdpool(x) + self.avg_pool(x)
        y = self.mlp(y)
        return x * y

class FeatureInteract(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.caif = CAIF(in_ch * 2)

        self.rssc_cscf = RSSC_CSCF(128,128)
        
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)

        x_o = self.rssc_cscf(self.caif(x))
    
        return x_o

class RSSC_CSCF(nn.Module):
    def __init__(self, in_channels, out_channels, conv=nn.Conv2d):
        super(RSSC_CSCF, self).__init__()
        kwargs = {'padding': 1}

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3,  **kwargs)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3,padding=2,dilation=2)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3,dilation=3,padding=3)

        self.c4 = conv(self.remaining_channels, self.dc, kernel_size=3, **kwargs)
        self.act = nn.GELU()

        self.out_s = nn.Conv2d(4,4,1)

        self.dc = nn.Sequential(
            nn.Conv2d(self.dc * 4, self.dc * 4, 3, 1,1,groups=self.dc * 4),
            nn.BatchNorm2d(self.dc * 4),
            nn.Conv2d(self.dc * 4, 64, kernel_size=1, bias=False),
            nn.ReLU()
        )

    def forward(self, input):

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        split_tensors = []

        for channel in range(distilled_c1.size(1)):
            channel_tensors = [tensor[:, channel:channel + 1, :, :] for tensor in [distilled_c1, distilled_c2, distilled_c3, r_c4]]
            concatenated_channel = self.out_s(torch.cat(channel_tensors, dim=1))
            split_tensors.append(concatenated_channel)

        out = torch.cat(split_tensors, dim=1)

        out = self.dc(out)

        return out  

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )

class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SAFM(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128):
        super(SAFM, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.fusion = FeatureInteract(64)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        res = self.pre_conv(res)

        x = self.fusion(x,res)

        return x

class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.fusion = FeatureInteract(64)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        res = self.pre_conv(res)

        x = self.fusion(x,res)
        
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x

class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat

class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, scale= 4)

        self.b3 = Block(dim=decode_channels, scale= 4)
        self.p3 = SAFM(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, scale= 4)
        self.p2 = SAFM(encoder_channels[-3], decode_channels)

        if self.training:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            self.aux_head = AuxHead(decode_channels, num_classes)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):
        if self.training:
            x = self.b4(self.pre_conv(res4))
            h4 = self.up4(x)

            x = self.p3(x, res3)
            x = self.b3(x)
            h3 = self.up3(x)

            x = self.p2(x, res2)
            x = self.b2(x)
            h2 = x
            x = self.p1(x, res1)
            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            ah = h4 + h3 + h2
            ah = self.aux_head(ah, h, w)

            return x, ah
        else:
            x = self.b4(self.pre_conv(res4))
            x = self.p3(x, res3)
            x = self.b3(x)

            x = self.p2(x, res2)
            x = self.b2(x)

            x = self.p1(x, res1)

            x = self.segmentation_head(x)
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SFSNet(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnet18',
                 pretrained=True,
                 num_classes=6
                 ):
        super().__init__()

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, num_classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        if self.training:
            x, ah = self.decoder(res1, res2, res3, res4, h, w)
            return x, ah
        else:
            x = self.decoder(res1, res2, res3, res4, h, w)
            return x

if __name__ == '__main__':

    net = SFSNet(num_classes=6).to('cuda')

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total number of parameters: {total_params}")
    
    x = torch.rand(2,3,1024,1024).to('cuda')
    print(net(x)[0].shape)