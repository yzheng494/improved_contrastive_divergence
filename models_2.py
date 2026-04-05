from torch.nn import ModuleList
import torch.nn.functional as F
import torch.nn as nn
import torch
from easydict import EasyDict
from torch.nn.utils import spectral_norm
from utils import WSConv2d, GaussianSmoothing
from downsample import Downsample
from torch.nn import Dropout


def swish(x):
    return x * torch.sigmoid(x)


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out,attention


class CondResBlock(nn.Module):
    def __init__(self, args, downsample=True, rescale=True, filters=64, latent_dim=64, im_size=64):
        super(CondResBlock, self).__init__()

        self.filters = filters
        self.latent_dim = latent_dim
        self.im_size = im_size
        self.downsample = downsample

        if filters <= 128:
            # self.bn1 = nn.GroupNorm(filters // 4, filters)
            self.bn1 = nn.InstanceNorm2d(filters, affine=True)
            # self.bn1 = None
        else:
            self.bn1 = nn.GroupNorm(32, filters)
        # self.bn1 = nn.InstanceNorm2d(filters, affine=True)
        self.args = args

        # self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.conv1 = WSConv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.InstanceNorm2d(filters, affine=True)

        if filters <= 128:
            # self.bn2 = nn.GroupNorm(filters // 4, filters)
            self.bn2 = nn.InstanceNorm2d(filters, affine=True)
            # self.bn2 = None
        else:
            self.bn2 = nn.GroupNorm(32, filters)
        # self.bn2 = nn.GroupNorm(32, filters)
        # self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = WSConv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        self.dropout = Dropout(0.2)

        # Upscale to an mask of image
        # self.latent_fc1 = nn.Linear(10, 256)
        self.latent_map = nn.Linear(512, 2*filters)
        self.latent_map_2 = nn.Linear(512, 2*filters)
        # self.dropout = torch.nn.Dropout(p=0.1, inplace=False)

        self.relu = torch.nn.ReLU(inplace=True)
        # self.act = self.relu
        self.act = swish

        # Upscale to mask of image
        if downsample:
            if rescale:
                self.conv_downsample = nn.Conv2d(filters, 2 * filters, kernel_size=3, stride=1, padding=1)

                if args.alias:
                    self.avg_pool = Downsample(channels=2*filters)
                else:
                    self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
            else:
                self.conv_downsample = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

                if args.alias:
                    self.avg_pool = Downsample(channels=filters)
                else:
                    self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)


    def forward(self, x, y):
        x_orig = x

        # latent_1 = self.act(self.latent_fc1(y))
        # latent_2 = self.latent_fc2(latent_1).view(-1, 2*self.filters, 1, 1)

        if y is not None:
            latent_map = self.latent_map(y).view(-1, 2*self.filters, 1, 1)

            gain = latent_map[:, :self.filters]
            bias = latent_map[:, self.filters:]

        x = self.conv1(x)

        if self.bn1 is not None:
            x = self.bn1(x)

        if y is not None:
            x = gain * x + bias

        x = self.act(x)
        # x = self.dropout(x)

        if y is not None:
            latent_map = self.latent_map_2(y).view(-1, 2*self.filters, 1, 1)
            gain = latent_map[:, :self.filters]
            bias = latent_map[:, self.filters:]

        x = self.conv2(x)

        if self.bn2 is not None:
            x = self.bn2(x)

        # x = self.dropout(x)

        if y is not None:
            x = gain * x + bias

        x = self.act(x)
        # x_out = x
        x_out = x_orig + x

        if self.downsample:
            x_out = self.conv_downsample(x_out)
            x_out = self.act(self.avg_pool(x_out))

        return x_out

class ResNetModel(nn.Module):
    def __init__(self, args):
        super(ResNetModel, self).__init__()
        self.act = swish

        self.args = args
        self.init_main_model()

        if args.multiscale:
            self.init_mid_model()
            self.init_small_model()

        self.init_label_map()

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = Downsample(channels=3)
        self.heir_weight = nn.Parameter(torch.Tensor([1.0, 1.0, 1.0]))
        # self.act = self.relu
        self.cond = args.cond

    def init_main_model(self):
        args = self.args
        filter_dim = args.filter_dim
        latent_dim = args.filter_dim
        im_size = args.im_size

        self.conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)
        self.res_1a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False)
        self.res_1b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=False)

        self.res_2a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False)
        self.res_2b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True)

        self.res_3a = CondResBlock(args, filters=2*filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False)
        self.res_3b = CondResBlock(args, filters=2*filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True)

        self.res_4a = CondResBlock(args, filters=4*filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False)
        self.res_4b = CondResBlock(args, filters=4*filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True)

        self.self_attn = Self_Attn(2 * filter_dim, self.act)

        # self.fc1 = nn.Linear(filter_dim*8, 128)
        self.energy_map = nn.Linear(filter_dim*8, 1)

    def init_mid_model(self):
        args = self.args
        filter_dim = args.filter_dim
        latent_dim = args.filter_dim
        im_size = args.im_size

        self.mid_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)
        self.mid_res_1a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False)
        self.mid_res_1b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=False)

        self.mid_res_2a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False)
        self.mid_res_2b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True)

        self.mid_res_3a = CondResBlock(args, filters=2*filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False)
        self.mid_res_3b = CondResBlock(args, filters=2*filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True)

        # self.mid_fc1 = nn.Linear(filter_dim*4, 128)
        self.mid_energy_map = nn.Linear(filter_dim*4, 1)
        self.avg_pool = Downsample(channels=3)

    def init_small_model(self):
        args = self.args
        filter_dim = args.filter_dim
        latent_dim = args.filter_dim
        im_size = args.im_size

        self.small_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)
        self.small_res_1a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False)
        self.small_res_1b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=False)

        self.small_res_2a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False)
        self.small_res_2b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True)

        # self.small_fc1 = nn.Linear(filter_dim*2, 128)
        self.small_energy_map = nn.Linear(filter_dim*2, 1)

    def init_label_map(self):
        args = self.args
        filter_dim = args.filter_dim
        latent_dim = args.filter_dim
        im_size = args.im_size

        self.map_fc1 = nn.Linear(10, 512)
        self.map_fc2 = nn.Linear(512, 512)
        self.map_fc3 = nn.Linear(512, 512)
        self.map_fc4 = nn.Linear(512, 512)

    def main_model(self, x, latent, compute_feat=False):
        x = self.act(self.conv1(x))

        x = self.res_1a(x, latent)
        x = self.res_1b(x, latent)

        x = self.res_2a(x, latent)
        x = self.res_2b(x, latent)

        if self.args.self_attn:
            x, _ = self.self_attn(x)

        x = self.res_3a(x, latent)
        x = self.res_3b(x, latent)

        x = self.res_4a(x, latent)
        x = self.res_4b(x, latent)
        x = self.act(x)

        x = x.sum(dim=2).sum(dim=2)

        if compute_feat:
            return x

        x = x.view(x.size(0), -1)
        # x = self.act(self.fc1(x))
        energy = self.energy_map(x)

        if self.args.square_energy:
            energy = torch.pow(energy, 2)

        return energy

    def mid_model(self, x, latent):
        # x = F.avg_pool2d(x, 3, stride=2, padding=1)
        x = self.downsample(x)
        # x = F.interpolate(x, size=16, mode='bilinear')

        x = self.act(self.mid_conv1(x))

        x = self.mid_res_1a(x, latent)
        x = self.mid_res_1b(x, latent)

        x = self.mid_res_2a(x, latent)
        x = self.mid_res_2b(x, latent)

        x = self.mid_res_3a(x, latent)
        x = self.mid_res_3b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)
        # x = self.act(self.mid_fc1(x))
        energy = self.mid_energy_map(x)

        if self.args.square_energy:
            energy = torch.pow(energy, 2)

        return energy

    def small_model(self, x, latent):
        # x = F.avg_pool2d(x, 3, stride=2, padding=1)
        # x = F.avg_pool2d(x, 3, stride=2, padding=1)
        x = self.downsample(x)
        x = self.downsample(x)
        # x = F.interpolate(x, size=8, mode='bilinear')

        x = self.act(self.small_conv1(x))

        x = self.small_res_1a(x, latent)
        x = self.small_res_1b(x, latent)

        x = self.small_res_2a(x, latent)
        x = self.small_res_2b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)
        # x = self.act(self.small_fc1(x))
        energy = self.small_energy_map(x)

        if self.args.square_energy:
            energy = torch.pow(energy, 2)

        return energy

    def label_map(self, latent):
        x = swish(self.map_fc1(latent))
        x = swish(self.map_fc2(x))
        x = swish(self.map_fc3(x))
        x = swish(self.map_fc4(x))

        return x

    def forward(self, x, latent):
        args = self.args

        if self.cond:
            latent = self.label_map(latent)
        else:
            latent = None

        energy = self.main_model(x, latent)

        if args.multiscale:
            # energy = energy + 0.2 * self.mid_model(x, None) + 0.1 * self.small_model(x, None)
            energy = self.heir_weight[0] * energy + self.heir_weight[1] * self.mid_model(x, latent) + self.heir_weight[2] * self.small_model(x, latent)

        return energy

    def compute_feat(self, x, latent):
        return self.main_model(x, None, compute_feat=True)


class CelebAModel(nn.Module):
    def __init__(self, args):
        super(CelebAModel, self).__init__()
        self.act = swish
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cond = args.cond

        self.args = args
        self.init_main_model()

        self.init_label_map()

        if args.multiscale:
            self.init_mid_model()
            self.init_small_model()

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = Downsample(channels=3)
        self.heir_weight = nn.Parameter(torch.Tensor([1.0, 1.0, 1.0]))

    def init_main_model(self):
        args = self.args
        filter_dim = args.filter_dim
        latent_dim = args.filter_dim
        im_size = args.im_size

        self.conv1 = nn.Conv2d(3, filter_dim // 2, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(filter_dim, filter_dim, kernel_size=5, stride=2, padding=1)

        self.res_1a = CondResBlock(args, filters=filter_dim // 2, latent_dim=latent_dim, im_size=im_size, downsample=True)
        self.res_1b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=False)

        self.res_2a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=True, rescale=False)
        self.res_2b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True)

        self.res_3a = CondResBlock(args, filters=2*filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False)
        self.res_3b = CondResBlock(args, filters=2*filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True)

        self.res_4a = CondResBlock(args, filters=4*filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False)
        self.res_4b = CondResBlock(args, filters=4*filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True)

        self.self_attn = Self_Attn(4 * filter_dim, self.act)

        self.energy_map = nn.Linear(filter_dim*8, 1)

    def init_mid_model(self):
        args = self.args
        filter_dim = args.filter_dim
        latent_dim = args.filter_dim
        im_size = args.im_size

        self.mid_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)

        self.mid_res_1a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=True, rescale=False)
        self.mid_res_1b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=False)

        self.mid_res_2a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=True, rescale=False)
        self.mid_res_2b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True)

        self.mid_res_3a = CondResBlock(args, filters=2*filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=False)
        self.mid_res_3b = CondResBlock(args, filters=2*filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True)

        # self.mid_fc1 = nn.Linear(filter_dim*4, 128)
        self.mid_energy_map = nn.Linear(filter_dim*4, 1)
        self.avg_pool = Downsample(channels=3)

    def init_small_model(self):
        args = self.args
        filter_dim = args.filter_dim
        latent_dim = args.filter_dim
        im_size = args.im_size

        self.small_conv1 = nn.Conv2d(3, filter_dim, kernel_size=3, stride=1, padding=1)

        self.small_res_1a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=True, rescale=False)
        self.small_res_1b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=False)

        self.small_res_2a = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, downsample=True, rescale=False)
        self.small_res_2b = CondResBlock(args, filters=filter_dim, latent_dim=latent_dim, im_size=im_size, rescale=True)

        self.small_energy_map = nn.Linear(filter_dim*2, 1)

    def init_label_map(self):
        args = self.args
        filter_dim = args.filter_dim
        latent_dim = args.filter_dim
        im_size = args.im_size

        self.map_fc1 = nn.Linear(2, 512)
        self.map_fc2 = nn.Linear(512, 512)
        self.map_fc3 = nn.Linear(512, 512)
        self.map_fc4 = nn.Linear(512, 512)

    def main_model(self, x, latent):
        x = self.act(self.conv1(x))
        # x = self.act(self.conv2(x))
        # x = self.act(self.conv3(x))
        # x = self.max_pool(x)

        x = self.res_1a(x, latent)
        x = self.res_1b(x, latent)

        x = self.res_2a(x, latent)
        x = self.res_2b(x, latent)


        x = self.res_3a(x, latent)
        x = self.res_3b(x, latent)

        if self.args.self_attn:
            x, _ = self.self_attn(x)

        x = self.res_4a(x, latent)
        x = self.res_4b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)
        # x = self.act(self.fc1(x))
        energy = self.energy_map(x)

        if self.args.square_energy:
            energy = torch.pow(energy, 2)

        return energy

    def mid_model(self, x, latent):
        # x = F.avg_pool2d(x, 3, stride=2, padding=1)
        x = self.downsample(x)
        # x = F.interpolate(x, size=16, mode='bilinear')

        x = self.act(self.mid_conv1(x))

        x = self.mid_res_1a(x, latent)
        x = self.mid_res_1b(x, latent)

        x = self.mid_res_2a(x, latent)
        x = self.mid_res_2b(x, latent)

        x = self.mid_res_3a(x, latent)
        x = self.mid_res_3b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)
        # x = self.act(self.mid_fc1(x))
        energy = self.mid_energy_map(x)

        if self.args.square_energy:
            energy = torch.pow(energy, 2)

        return energy

    def small_model(self, x, latent):
        # x = F.avg_pool2d(x, 3, stride=2, padding=1)
        # x = F.avg_pool2d(x, 3, stride=2, padding=1)
        x = self.downsample(x)
        x = self.downsample(x)
        # x = F.interpolate(x, size=8, mode='bilinear')

        x = self.act(self.small_conv1(x))

        x = self.small_res_1a(x, latent)
        x = self.small_res_1b(x, latent)

        x = self.small_res_2a(x, latent)
        x = self.small_res_2b(x, latent)
        x = self.act(x)

        x = x.mean(dim=2).mean(dim=2)

        x = x.view(x.size(0), -1)
        # x = self.act(self.small_fc1(x))
        energy = self.small_energy_map(x)

        if self.args.square_energy:
            energy = torch.pow(energy, 2)

        return energy

    def label_map(self, latent):
        x = self.act(self.map_fc1(latent))
        x = self.act(self.map_fc2(x))
        x = self.act(self.map_fc3(x))
        x = self.act(self.map_fc4(x))

        return x

    def forward(self, x, latent):
        args = self.args

        if self.cond:
            latent = self.label_map(latent)
        else:
            latent = None

        energy = self.main_model(x, latent)

        if args.multiscale:
            # energy = energy + 0.2 * self.mid_model(x, None) + 0.1 * self.small_model(x, None)
            energy = self.heir_weight[0] * energy + self.heir_weight[1] * self.mid_model(x, latent) + self.heir_weight[2] * self.small_model(x, latent)

        return energy


class ModelLinear(nn.Module):
    def __init__(self, args):
        super(ModelLinear, self).__init__()
        self.fc1 = nn.Linear(args.filter_dim * 8, 10)

    def forward(self, x):
        return self.fc1(x)
