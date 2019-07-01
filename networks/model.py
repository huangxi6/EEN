import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
affine_par = True
import functools
import pdb

import sys, os

from libs import InPlaceABN, InPlaceABNSync


BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
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

        out = out + residual      
        out = self.relu_inplace(out)

        return out


class Edge_Enhancement_module(nn.Module):

    def __init__(self):
        super(Edge_Enhancement_module, self).__init__()

        self.layer_denseaspp = DenseASPPModule(3072, 1024, 512)
        self.conv1 =  nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
            ) 
        self.conv2 =  nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
            )
        self.conv3 = nn.Conv2d(256,2, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv4 = nn.Conv2d(512,2, kernel_size=3, padding=1, dilation=1, bias=True)
        self.bn_relu = InPlaceABNSync(256)
        self.conv5 = nn.Conv2d(6,2, kernel_size=1, padding=0, dilation=1, bias=True)
        
    def forward(self, x1, x2, x3, x4):   
        _, _, h, w = x1.size()

        edge_feat_x1 = self.conv1(x1)
        edge_x1 = self.conv3(edge_feat_x1)

        edge_feat_x2 = self.conv2(x2)
        edge_feat_x2 = F.interpolate(edge_feat_x2, size=(h, w), mode='bilinear', align_corners=True)
        edge_feat_x2 = self.bn_relu(edge_feat_x2)
        edge_x2 = self.conv3(edge_feat_x2)

        x = torch.cat((x3, x4), dim = 1)
        edge_feat_x3 = self.layer_denseaspp(x)
        edge_feat_x3 = F.interpolate(edge_feat_x3, size=(h, w), mode='bilinear', align_corners=True)
        edge_x3 = self.conv4(edge_feat_x3)


        edge_feat = torch.cat([edge_feat_x1,edge_feat_x2,edge_feat_x3],dim=1)
        edge = torch.cat([edge_x1,edge_x2,edge_x3],dim=1)
        edge = self.conv5(edge)
         
        return edge, edge_feat


class DenseASPPModule(nn.Module):
    
    def __init__(self, features=2048, temp_features=1024, out_features=512, dilations=(3, 6, 12, 18, 24)):
    #def __init__(self, features=2048, temp_features=1024, out_features=512, dilations=(2, 4, 8, 12, 16)):
        super(DenseASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(features, temp_features, kernel_size=1, padding=0),
                                   InPlaceABNSync(temp_features),
                                   nn.Conv2d(temp_features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   InPlaceABNSync(out_features),
                                   nn.Dropout2d(0.1))
        self.conv2 = nn.Sequential(nn.Conv2d(features + out_features * 1, temp_features, kernel_size=1, padding=0),
                                   InPlaceABNSync(temp_features),
                                   nn.Conv2d(temp_features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   InPlaceABNSync(out_features),
                                   nn.Dropout2d(0.1))
        self.conv3 = nn.Sequential(nn.Conv2d(features + out_features * 2, temp_features, kernel_size=1, padding=0),
                                   InPlaceABNSync(temp_features),
                                   nn.Conv2d(temp_features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   InPlaceABNSync(out_features),
                                   nn.Dropout2d(0.1))
        self.conv4 = nn.Sequential(nn.Conv2d(features + out_features * 3, temp_features, kernel_size=1, padding=0),
                                   InPlaceABNSync(temp_features),
                                   nn.Conv2d(temp_features, out_features, kernel_size=3, padding=dilations[3], dilation=dilations[3], bias=False),
                                   InPlaceABNSync(out_features),
                                   nn.Dropout2d(0.1))
        self.conv5 = nn.Sequential(nn.Conv2d(features + out_features * 4, temp_features, kernel_size=1, padding=0),
                                   InPlaceABNSync(temp_features),
                                   nn.Conv2d(temp_features, out_features, kernel_size=3, padding=dilations[4], dilation=dilations[4], bias=False),
                                   InPlaceABNSync(out_features),
                                   nn.Dropout2d(0.1))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features)
            )
        
    def forward(self, x):

        _, _, h, w = x.size()
        feature = x #2048

        feat1 = self.conv1(feature)
        feature = torch.cat((feat1, feature), 1) #2048+512
       
        feat2 = self.conv2(feature) #512       
        feature = torch.cat((feat2, feature), 1) #2048+512*2        
        
        feat3 = self.conv3(feature) #512    
        feature = torch.cat((feat3, feature), 1)#2048+512*3        
     
        feat4 = self.conv4(feature)#512      
        feature = torch.cat((feat4, feature), 1)#2048+512*4        

        feat5 = self.conv5(feature)#512    
        feature = torch.cat((feat5, feature), 1)#2048+512*5
 
 
        out = feature     
        bottle = self.bottleneck(out)

        return bottle


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        
        low_level_inplanes = 256

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256)
            )
        self.conv3 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       InPlaceABNSync(256),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       InPlaceABNSync(256),
                                       nn.Dropout(0.1))
        self.last_conv = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        x = self.conv2(x)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.conv3(x)
        #seg = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1,1,1))
        

        self.layer5 = DenseASPPModule(3072,1024,512)
        self.edge_layer = Edge_Enhancement_module()
        self.layer6 = Decoder(num_classes)

        self.layer7 = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
            )
        

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x4_5 = torch.cat((x4, x5), dim = 1)
        x6 = self.layer5(x4_5)
        edge,edge_fea = self.edge_layer(x2,x3,x4,x5)
        x = self.layer6(x6,x2)
        x = torch.cat([x, edge_fea], dim=1)
        seg = self.layer7(x)
    
        return [[seg], [edge]]

def EEN(num_classes=21):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model



