import torch
import torch.nn as nn

import utils.batch_norm
import utils.whitening


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def compute_bn_stats(state_dict):
    # state_dict = state_dict = torch.load(path) #'/home/sroy/.torch/models/resnet50-19c8e357.pth'

    bn_key_names = []
    for name, param in state_dict.items():
        if name.find('bn') != -1:
            bn_key_names.append(name)
        elif name.find('downsample') != -1:
            bn_key_names.append(name)

    # keeping only the batch norm specific elements in the dictionary
    bn_dict = {k: v for k, v in state_dict.items() if k in bn_key_names}

    return bn_dict


class whitening_scale_shift(nn.Module):
    def __init__(self, planes, group_size, running_mean, running_variance, track_running_stats=True, affine=True):
        super(whitening_scale_shift, self).__init__()
        self.planes = planes
        self.group_size = group_size
        self.track_running_stats = track_running_stats
        self.affine = affine
        self.running_mean = running_mean
        self.running_variance = running_variance

        self.wh = utils.whitening.WTransform2d(self.planes,
                                               self.group_size,
                                               running_m=self.running_mean,
                                               running_var=self.running_variance,
                                               track_running_stats=self.track_running_stats)
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.planes, 1, 1))
            self.beta = nn.Parameter(torch.zeros(self.planes, 1, 1))

    def forward(self, x):
        out = self.wh(x)
        if self.affine:
            out = out * self.gamma + self.beta
        return out
# class Bottleneck_rt(nn.Module):


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, layer, sub_layer, bn_dict, group_size=4, stride=1, downsample=None, rt = False):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.conv1 = conv1x1(inplanes, planes)
        if layer == 1:
            self.bns1 = whitening_scale_shift(planes=planes,
                                              group_size=group_size,
                                              running_mean=bn_dict['layer' + str(layer) + '.' + str(
                                                  sub_layer) + '.bn1.wh.running_mean'],
                                              running_variance=bn_dict['layer' + str(layer) + '.' + str(
                                                  sub_layer) + '.bn1.wh.running_variance'],
                                              affine=False)
            self.bnt1 = whitening_scale_shift(planes=planes,
                                              group_size=group_size,
                                              running_mean=bn_dict['layer' + str(layer) + '.' + str(
                                                  sub_layer) + '.bn1.wh.running_mean'],
                                              running_variance=bn_dict['layer' + str(layer) + '.' + str(
                                                  sub_layer) + '.bn1.wh.running_variance'],
                                              affine=False)
            self.bnt1_aug = whitening_scale_shift(planes=planes,
                                                  group_size=group_size,
                                                  running_mean=bn_dict['layer' + str(layer) + '.' + str(
                                                      sub_layer) + '.bn1.wh.running_mean'],
                                                  running_variance=bn_dict['layer' + str(layer) + '.' + str(
                                                      sub_layer) + '.bn1.wh.running_variance'],
                                                  affine=False)
            self.gamma1 = nn.Parameter(
                bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.gamma'])
            self.beta1 = nn.Parameter(
                bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.beta'])
        else:
            self.bns1 = utils.batch_norm.BatchNorm2d(num_features=planes,
                                                     running_m=bn_dict['layer' + str(layer) + '.' + str(
                                                         sub_layer) + '.bn1.running_mean'],
                                                     running_v=bn_dict['layer' + str(layer) + '.' + str(
                                                         sub_layer) + '.bn1.running_var'],
                                                     affine=False)
            self.bnt1 = utils.batch_norm.BatchNorm2d(num_features=planes,
                                                     running_m=bn_dict['layer' + str(layer) + '.' + str(
                                                         sub_layer) + '.bn1.running_mean'],
                                                     running_v=bn_dict['layer' + str(layer) + '.' + str(
                                                         sub_layer) + '.bn1.running_var'],
                                                     affine=False)
            self.bnt1_aug = utils.batch_norm.BatchNorm2d(num_features=planes,
                                                         running_m=bn_dict['layer' + str(layer) + '.' + str(
                                                             sub_layer) + '.bn1.running_mean'],
                                                         running_v=bn_dict['layer' + str(layer) + '.' + str(
                                                             sub_layer) + '.bn1.running_var'],
                                                         affine=False)
            self.gamma1 = nn.Parameter(
                bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.weight'].view(-1, 1, 1))
            self.beta1 = nn.Parameter(
                bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn1.bias'].view(-1, 1, 1))

        self.conv2 = conv3x3(planes, planes, stride)
        if layer == 1:
            self.bns2 = whitening_scale_shift(planes=planes,
                                              group_size=group_size,
                                              running_mean=bn_dict['layer' + str(layer) + '.' + str(
                                                  sub_layer) + '.bn2.wh.running_mean'],
                                              running_variance=bn_dict['layer' + str(layer) + '.' + str(
                                                  sub_layer) + '.bn2.wh.running_variance'],
                                              affine=False)
            self.bnt2 = whitening_scale_shift(planes=planes,
                                              group_size=group_size,
                                              running_mean=bn_dict['layer' + str(layer) + '.' + str(
                                                  sub_layer) + '.bn2.wh.running_mean'],
                                              running_variance=bn_dict['layer' + str(layer) + '.' + str(
                                                  sub_layer) + '.bn2.wh.running_variance'],
                                              affine=False)
            self.bnt2_aug = whitening_scale_shift(planes=planes,
                                                  group_size=group_size,
                                                  running_mean=bn_dict['layer' + str(layer) + '.' + str(
                                                      sub_layer) + '.bn2.wh.running_mean'],
                                                  running_variance=bn_dict['layer' + str(layer) + '.' + str(
                                                      sub_layer) + '.bn2.wh.running_variance'],
                                                  affine=False)
            self.gamma2 = nn.Parameter(
                bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.gamma'])
            self.beta2 = nn.Parameter(
                bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.beta'])
        else:
            self.bns2 = utils.batch_norm.BatchNorm2d(num_features=planes,
                                                     running_m=bn_dict['layer' + str(layer) + '.' + str(
                                                         sub_layer) + '.bn2.running_mean'],
                                                     running_v=bn_dict['layer' + str(layer) + '.' + str(
                                                         sub_layer) + '.bn2.running_var'],
                                                     affine=False)
            self.bnt2 = utils.batch_norm.BatchNorm2d(num_features=planes,
                                                     running_m=bn_dict['layer' + str(layer) + '.' + str(
                                                         sub_layer) + '.bn2.running_mean'],
                                                     running_v=bn_dict['layer' + str(layer) + '.' + str(
                                                         sub_layer) + '.bn2.running_var'],
                                                     affine=False)
            self.bnt2_aug = utils.batch_norm.BatchNorm2d(num_features=planes,
                                                         running_m=bn_dict['layer' + str(layer) + '.' + str(
                                                             sub_layer) + '.bn2.running_mean'],
                                                         running_v=bn_dict['layer' + str(layer) + '.' + str(
                                                             sub_layer) + '.bn2.running_var'],
                                                         affine=False)
            self.gamma2 = nn.Parameter(
                bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.weight'].view(-1, 1, 1))
            self.beta2 = nn.Parameter(
                bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn2.bias'].view(-1, 1, 1))

        self.conv3 = conv1x1(planes, planes * self.expansion)
        if layer == 1:
            self.bns3 = whitening_scale_shift(planes=planes * self.expansion,
                                              group_size=group_size,
                                              running_mean=bn_dict['layer' + str(layer) + '.' + str(
                                                  sub_layer) + '.bn3.wh.running_mean'],
                                              running_variance=bn_dict['layer' + str(layer) + '.' + str(
                                                  sub_layer) + '.bn3.wh.running_variance'],
                                              affine=False)
            self.bnt3 = whitening_scale_shift(planes=planes * self.expansion,
                                              group_size=group_size,
                                              running_mean=bn_dict['layer' + str(layer) + '.' + str(
                                                  sub_layer) + '.bn3.wh.running_mean'],
                                              running_variance=bn_dict['layer' + str(layer) + '.' + str(
                                                  sub_layer) + '.bn3.wh.running_variance'],
                                              affine=False)
            self.bnt3_aug = whitening_scale_shift(planes=planes * self.expansion,
                                                  group_size=group_size,
                                                  running_mean=bn_dict['layer' + str(layer) + '.' + str(
                                                      sub_layer) + '.bn3.wh.running_mean'],
                                                  running_variance=bn_dict['layer' + str(layer) + '.' + str(
                                                      sub_layer) + '.bn3.wh.running_variance'],
                                                  affine=False)
            self.gamma3 = nn.Parameter(
                bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.gamma'])
            self.beta3 = nn.Parameter(
                bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.beta'])
        else:
            self.bns3 = utils.batch_norm.BatchNorm2d(num_features=planes * self.expansion,
                                                     running_m=bn_dict['layer' + str(layer) + '.' + str(
                                                         sub_layer) + '.bn3.running_mean'],
                                                     running_v=bn_dict['layer' + str(layer) + '.' + str(
                                                         sub_layer) + '.bn3.running_var'],
                                                     affine=False)
            self.bnt3 = utils.batch_norm.BatchNorm2d(num_features=planes * self.expansion,
                                                     running_m=bn_dict['layer' + str(layer) + '.' + str(
                                                         sub_layer) + '.bn3.running_mean'],
                                                     running_v=bn_dict['layer' + str(layer) + '.' + str(
                                                         sub_layer) + '.bn3.running_var'],
                                                     affine=False)
            self.bnt3_aug = utils.batch_norm.BatchNorm2d(num_features=planes * self.expansion,
                                                         running_m=bn_dict['layer' + str(layer) + '.' + str(
                                                             sub_layer) + '.bn3.running_mean'],
                                                         running_v=bn_dict['layer' + str(layer) + '.' + str(
                                                             sub_layer) + '.bn3.running_var'],
                                                         affine=False)
            self.gamma3 = nn.Parameter(
                bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.weight'].view(-1, 1, 1))
            self.beta3 = nn.Parameter(
                bn_dict['layer' + str(layer) + '.' + str(sub_layer) + '.bn3.bias'].view(-1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if self.downsample is not None:
            if layer == 1:
                self.downsample_bns = whitening_scale_shift(planes=planes * self.expansion,
                                                            group_size=group_size,
                                                            running_mean=bn_dict['layer' + str(
                                                                layer) + '.0.downsample_bn.wh.running_mean'],
                                                            running_variance=bn_dict['layer' + str(
                                                                layer) + '.0.downsample_bn.wh.running_variance'],
                                                            affine=False)
                self.downsample_bnt = whitening_scale_shift(planes=planes * self.expansion,
                                                            group_size=group_size,
                                                            running_mean=bn_dict['layer' + str(
                                                                layer) + '.0.downsample_bn.wh.running_mean'],
                                                            running_variance=bn_dict['layer' + str(
                                                                layer) + '.0.downsample_bn.wh.running_variance'],
                                                            affine=False)
                self.downsample_bnt_aug = whitening_scale_shift(planes=planes * self.expansion,
                                                                group_size=group_size,
                                                                running_mean=bn_dict['layer' + str(
                                                                    layer) + '.0.downsample_bn.wh.running_mean'],
                                                                running_variance=bn_dict['layer' + str(
                                                                    layer) + '.0.downsample_bn.wh.running_variance'],
                                                                affine=False)
                self.downsample_gamma = nn.Parameter(
                    bn_dict['layer' + str(layer) + '.0.downsample_bn.gamma'])
                self.downsample_beta = nn.Parameter(
                    bn_dict['layer' + str(layer) + '.0.downsample_bn.beta'])
            else:
                self.downsample_bns = utils.batch_norm.BatchNorm2d(num_features=planes * self.expansion,
                                                                   running_m=bn_dict['layer' + str(
                                                                       layer) + '.0.downsample_bn.running_mean'],
                                                                   running_v=bn_dict['layer' + str(
                                                                       layer) + '.0.downsample_bn.running_var'],
                                                                   affine=False)
                self.downsample_bnt = utils.batch_norm.BatchNorm2d(num_features=planes * self.expansion,
                                                                   running_m=bn_dict['layer' + str(
                                                                       layer) + '.0.downsample_bn.running_mean'],
                                                                   running_v=bn_dict['layer' + str(
                                                                       layer) + '.0.downsample_bn.running_var'],
                                                                   affine=False)
                self.downsample_bnt_aug = utils.batch_norm.BatchNorm2d(num_features=planes * self.expansion,
                                                                       running_m=bn_dict['layer' + str(
                                                                           layer) + '.0.downsample_bn.running_mean'],
                                                                       running_v=bn_dict['layer' + str(
                                                                           layer) + '.0.downsample_bn.running_var'],
                                                                       affine=False)
                self.downsample_gamma = nn.Parameter(
                    bn_dict['layer' + str(layer) + '.0.downsample_bn.weight'].view(-1, 1, 1))
                self.downsample_beta = nn.Parameter(
                    bn_dict['layer' + str(layer) + '.0.downsample_bn.bias'].view(-1, 1, 1))

    def forward(self, x):
        if self.training:
            # to do
            identity = x
            out = self.conv1(x)
            out_s, out_t, out_t_dup = torch.split(
                out, split_size_or_sections=out.shape[0] // 3, dim=0)
            out = torch.cat((self.bns1(out_s), torch.cat((self.bnt1(out_t), self.bnt1_aug(
                out_t_dup)), dim=0)), dim=0) * self.gamma1 + self.beta1
            out = self.relu(out)

            out = self.conv2(out)
            out_s, out_t, out_t_dup = torch.split(
                out, split_size_or_sections=out.shape[0] // 3, dim=0)
            out = torch.cat((self.bns2(out_s), torch.cat((self.bnt2(out_t), self.bnt2_aug(
                out_t_dup)), dim=0)), dim=0) * self.gamma2 + self.beta2
            out = self.relu(out)

            out = self.conv3(out)
            out_s, out_t, out_t_dup = torch.split(
                out, split_size_or_sections=out.shape[0] // 3, dim=0)
            out = torch.cat((self.bns3(out_s), torch.cat((self.bnt3(out_t), self.bnt3_aug(
                out_t_dup)), dim=0)), dim=0) * self.gamma3 + self.beta3

            if self.downsample is not None:
                identity = self.downsample(x)
                identity_s, identity_t, identity_t_dup = torch.split(
                    identity, split_size_or_sections=identity.shape[0] // 3, dim=0)
                identity = torch.cat((self.downsample_bns(identity_s),
                                      torch.cat((self.downsample_bnt(identity_t), self.downsample_bnt_aug(identity_t_dup)), dim=0)), dim=0) * self.downsample_gamma + self.downsample_beta

            out = out.clone() + identity
            out = self.relu(out)
        else:
            identity = x

            out = self.conv1(x)
            out = self.bnt1(out) * self.gamma1 + self.beta1
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bnt2(out) * self.gamma2 + self.beta2
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bnt3(out) * self.gamma3 + self.beta3

            if self.downsample is not None:
                identity = self.downsample(x)
                identity = self.downsample_bnt(
                    identity) * self.downsample_gamma + self.downsample_beta

            out = out.clone() + identity
            out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, state_dict, num_classes=65, zero_init_residual=False, group_size=4, rt=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        if rt:
            self.bn_dict = state_dict
        else:
            self.bn_dict = compute_bn_stats(state_dict)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)

        if rt:
            self.bns1 = whitening_scale_shift(planes=64,
                                              group_size=group_size,
                                              running_mean=self.bn_dict['bns1.wh.running_mean'],
                                              running_variance=self.bn_dict['bns1.wh.running_variance'],
                                              affine=False)
            self.bnt1 = whitening_scale_shift(planes=64,
                                              group_size=group_size,
                                              running_mean=self.bn_dict['bnt1.wh.running_mean'],
                                              running_variance=self.bn_dict['bnt1.wh.running_variance'],
                                              affine=False)
            self.bnt1_aug = whitening_scale_shift(planes=64,
                                                  group_size=group_size,
                                                  running_mean=self.bn_dict['bnt1.wh.running_mean'],
                                                  running_variance=self.bn_dict['bnt1.wh.running_variance'],
                                                  affine=False)
            self.gamma1 = nn.Parameter(self.bn_dict['gamma1'])
            self.beta1 = nn.Parameter(self.bn_dict['beta1'])

        else:
            self.bns1 = whitening_scale_shift(planes=64,
                                              group_size=group_size,
                                              running_mean=self.bn_dict['bn1.wh.running_mean'],
                                              running_variance=self.bn_dict['bn1.wh.running_variance'],
                                              affine=False)
            self.bnt1 = whitening_scale_shift(planes=64,
                                              group_size=group_size,
                                              running_mean=self.bn_dict['bn1.wh.running_mean'],
                                              running_variance=self.bn_dict['bn1.wh.running_variance'],
                                              affine=False)
            self.bnt1_aug = whitening_scale_shift(planes=64,
                                                  group_size=group_size,
                                                  running_mean=self.bn_dict['bn1.wh.running_mean'],
                                                  running_variance=self.bn_dict['bn1.wh.running_variance'],
                                                  affine=False)
            self.gamma1 = nn.Parameter(self.bn_dict['bn1.gamma'])
            self.beta1 = nn.Parameter(self.bn_dict['bn1.beta'])

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], self.bn_dict, layer=1)
        self.layer2 = self._make_layer(
            block, 128, layers[1], self.bn_dict, stride=2, layer=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], self.bn_dict, stride=2, layer=3)
        self.layer4 = self._make_layer(
            block, 512, layers[3], self.bn_dict, stride=2, layer=4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, bn_dict, layer=1, group_size=4, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                # nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, layer, 0,
                            bn_dict, group_size, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                layer, i, bn_dict, group_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            x = self.conv1(x)
            x_s, x_t, x_t_dup = torch.split(
                x, split_size_or_sections=x.shape[0] // 3, dim=0)
            x = torch.cat((self.bns1(x_s), torch.cat((self.bnt1(x_t), self.bnt1_aug(
                x_t_dup)), dim=0)), dim=0) * self.gamma1 + self.beta1
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc_out(x)
        else:
            x = self.conv1(x)
            x = self.bnt1(x) * self.gamma1 + self.beta1
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc_out(x)

        return x
