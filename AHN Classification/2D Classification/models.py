# Import Libraries
# PyTorch
import torch
import torch.nn as nn
from torch import cuda 
from torchvision import models
from torchsummary import summary
# Data Science tools
import timm
import numpy as np
from turtle import forward
# Warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# SelfONN Blocks
from selfonn import SelfONNLayer


def reset_function_generic(m):
    if hasattr(m,'reset_parameters') or hasattr(m,'reset_parameters_like_torch'): 
        # print(m) 
        if isinstance(m, SelfONNLayer):
            m.reset_parameters_like_torch() 
        else:
            m.reset_parameters()


class SqueezeLayer(nn.Module):
    def forward(self,x):
        x = x.squeeze(2)
        x = x.squeeze(2)
        return x 


class UnSqueezeLayer(nn.Module):
    def forward(self,x):
        x = x.unsqueeze(2).unsqueeze(3)
        return x 


class CNN_Classifier(nn.Module):
    def __init__(self, in_channels, class_num, final_activation):
        super().__init__()
        if final_activation == 'LogSigmoid':
            self.classifier = nn.Sequential(
                nn.Linear(in_channels, class_num),
                nn.LogSigmoid()
                )
        elif final_activation == 'LogSoftmax':
            self.classifier = nn.Sequential(
                nn.Linear(in_channels, class_num), 
                nn.LogSoftmax(dim=1)
                )
        elif final_activation == 'Sigmoid':
            self.classifier = nn.Sequential(
                nn.Linear(in_channels, class_num),
                nn.Sigmoid()
                )
        elif final_activation == 'Softmax':
            self.classifier = nn.Sequential(
                nn.Linear(in_channels, class_num),
                nn.Softmax(dim=1)
                )
        elif final_activation == 'Softsign':
            self.classifier = nn.Sequential(
                nn.Linear(in_channels, class_num),
                nn.Softsign()
                )
        elif final_activation == 'Tanh':
            self.classifier = nn.Sequential(
                nn.Linear(in_channels, class_num),
                nn.Tanh()
                )
        # torch.nn.init.xavier_uniform_(self.classifier[0].weight)
        # self.classifier[0].bias.data.fill_(0.01) 
     
    def forward(self,x):
        x = self.classifier(x)
        return x


class Self_B_ResBlock(nn.Module):
    def __init__(self, in_channels=3, channel1=8, channel2=16, channel3=8, resConnection=False,q_order=3):
        super().__init__()
        self.layer1 = SelfONNLayer(in_channels=in_channels,out_channels=channel1,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q_order,mode='fast')
        self.Batch1 = nn.BatchNorm2d(channel1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.resConnection = resConnection
        if self.resConnection:
            self.layer2 = SelfONNLayer(in_channels=channel1,out_channels=channel2,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast')
            self.Batch2 = nn.BatchNorm2d(channel2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        else:
            self.layer2 = SelfONNLayer(in_channels=channel1,out_channels=channel2,kernel_size=3,stride=2,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast')
            self.Batch2 = nn.BatchNorm2d(channel2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3 = SelfONNLayer(in_channels=channel2,out_channels=channel3,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q_order,mode='fast')
        self.Batch3 = nn.BatchNorm2d(channel3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.tanh = nn.Tanh()

    def forward(self,x):
        input = x.clone()
        x = self.tanh(self.Batch1(self.layer1(x)))
        x = self.tanh(self.Batch2(self.layer2(x)))
        x = self.tanh(self.Batch3(self.layer3(x)))
        if self.resConnection:
            x= self.tanh(x.clone()+ input)
            
            # x = torch.cat((x,input), 1)
        return x


class Self_MobileNet(nn.Module):
    def __init__(self, input_channel=3, last_layer_channel=32, class_num=10):
        super().__init__()
        self.class_num = class_num
        self.selfONN = SelfONNLayer(in_channels=input_channel,out_channels=32,kernel_size=3,stride=2,padding=1,dilation=1,groups=1,bias=True,q=3,mode='fast')
        self.batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.tanh = nn.Tanh()
        self.BottleneckResidual1 = Self_B_ResBlock(in_channels=32, channel1=16, channel2=48, channel3=32, resConnection=True)
        self.NonResidual1 = Self_B_ResBlock(in_channels=32, channel1=32, channel2=48, channel3=16, resConnection=False)
        self.BottleneckResidual2 = Self_B_ResBlock(in_channels=16, channel1=24, channel2=48, channel3=16, resConnection=True)
        self.NonResidual2 = Self_B_ResBlock(in_channels=16, channel1=24, channel2=32, channel3=48, resConnection=False)
        self.BottleneckResidual3 = Self_B_ResBlock(in_channels=48, channel1=16, channel2=56, channel3=48, resConnection=True)
        self.NonResidual3 = Self_B_ResBlock(in_channels=48, channel1=8, channel2=48, channel3=36, resConnection=False)
        self.BottleneckResidual4 = Self_B_ResBlock(in_channels=36, channel1=16, channel2=32, channel3=36, resConnection=True)
        self.BottleneckResidual5 = Self_B_ResBlock(in_channels=36, channel1=8, channel2=32, channel3=36, resConnection=True)
        self.NonResidual4 = Self_B_ResBlock(in_channels=36, channel1=8, channel2=32, channel3=last_layer_channel, resConnection=False)
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((7,7))
        self.Flatten = nn.Flatten()
        self.Dropout2d = nn.Dropout(p=0.1)
        self.self_MLP = CNN_Classifier(in_channels = int(49*last_layer_channel),class_num = self.class_num)
        
    def forward(self, x):
        x= self.tanh(self.batchnorm(self.selfONN(x)))
        x= self.BottleneckResidual1(x)
        x= self.NonResidual1(x)
        x= self.BottleneckResidual2(x)
        x= self.NonResidual2(x) 
        x= self.BottleneckResidual3(x)
        x= self.NonResidual3(x)
        x= self.BottleneckResidual4(x)
        x= self.BottleneckResidual5(x)
        x= self.NonResidual4(x)
        x= self.AdaptiveAvgPool2d(x)
        x = self.Flatten(x)
        x = self.Dropout2d(x)
        x= self.self_MLP(x)
        return x


class Self_DenseMobileNet(nn.Module):
    def __init__(self, input_channel=3, last_layer_channel=32, class_num=10, q_order=3):
        super().__init__()
        self.class_num = class_num
        self.InputMLP = 10
        self.selfONN = SelfONNLayer(in_channels=input_channel,out_channels=32,kernel_size=3,stride=2,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast')
        self.batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.tanh = nn.Tanh()
        self.BottleneckResidual1 = Self_B_ResBlock(in_channels=32, channel1=64, channel2=64, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual2 = Self_B_ResBlock(in_channels=32, channel1=72, channel2=72, channel3=32, resConnection=True,q_order=q_order)
        self.maxpool = nn.MaxPool2d(2)
        self.BottleneckResidual3 = Self_B_ResBlock(in_channels=32, channel1=84, channel2=84, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual4 = Self_B_ResBlock(in_channels=32, channel1=96, channel2=96, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual5 = Self_B_ResBlock(in_channels=32, channel1=96, channel2=96, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual6 = Self_B_ResBlock(in_channels=32, channel1=84, channel2=84, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual7 = Self_B_ResBlock(in_channels=32, channel1=72, channel2=72, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual8 = Self_B_ResBlock(in_channels=32, channel1=64, channel2=64, channel3=32, resConnection=True,q_order=q_order)
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        self.Flatten = nn.Flatten()
        self.Dropout = nn.Dropout(p=0.2)
        self.self_MLP = CNN_Classifier(in_channels =5*32,class_num = self.class_num)
    
    def forward(self,x):
        x = self.tanh(self.batchnorm(self.selfONN(x)))
        x = self.BottleneckResidual1(x)
        x = self.BottleneckResidual2(x)
        inMLP1 = self.Flatten(self.AdaptiveAvgPool2d(x)) # input for MLP
        #
        x = self.BottleneckResidual3(self.maxpool(x))
        x = self.BottleneckResidual4(x)
        inMLP2 = self.Flatten(self.AdaptiveAvgPool2d(x)) # input for MLP
        #
        x = self.BottleneckResidual5(self.maxpool(x))
        x = self.BottleneckResidual6(x)
        inMLP3 = self.Flatten(self.AdaptiveAvgPool2d(x)) # input for MLP
        #
        x = self.BottleneckResidual7(self.maxpool(x))
        x = self.BottleneckResidual8(x)
        inMLP4 = self.Flatten(self.AdaptiveAvgPool2d(x)) # input for MLP
        #
        x = self.BottleneckResidual7(self.maxpool(x))
        x = self.BottleneckResidual8(x)
        inMLP5 = self.Flatten(self.AdaptiveAvgPool2d(x)) # input for MLP
        #
        inMLP = torch.cat((inMLP1, inMLP2, inMLP3, inMLP4, inMLP5), dim=1)
        #
        x = self.Flatten(inMLP)
        x = self.Dropout(x)
        x = self.self_MLP(x)
        return x


class cnn_V5(nn.Module):
    def __init__(self, input_ch, class_num): 
        super(cnn_V5, self).__init__() 
        # 1st layer (conv)
        self.conv1 = cnn_V5.conv_block(in_channels=input_ch, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        # Average pooling 
        self.AvgPool = torch.nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.flatten = torch.nn.Flatten()
        # 2nd layer (MLP) 
        # conv_output = 7*7*20= 980
        self.MLP2 = torch.nn.Linear(in_features=980, out_features=class_num, bias=True)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        layer1 = self.conv1(x)
        layer1 = self.pool1(layer1)
        Pool_layer = self.AvgPool(layer1)
        Pool_layer = self.flatten(Pool_layer)
        Output_layer = self.MLP2(Pool_layer) 
        return self.softmax(Output_layer) 

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):     
        return nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), 
            torch.nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True)
            )


class Modified_DenseNet201_AUX(nn.Module):
    def __init__(self, class_num):
        super(Modified_DenseNet201_AUX, self).__init__()
        model = models.densenet201(weights='DEFAULT')
        self.ExtractedDenseBlock1 = nn.Sequential(*list(model.features.children())[0:5])
        self.ExtractedDenseBlock2 = nn.Sequential(*list(model.features.children())[0:7])
        self.ExtractedDenseBlock3 = nn.Sequential(*list(model.features.children())[0:9])
        self.ExtractedDenseBlock4 = nn.Sequential(*list(model.features.children())[0:])
        self.downSamp1 = nn.MaxPool2d(8)
        self.downSamp2 = nn.MaxPool2d(4)
        self.downSamp3 = nn.MaxPool2d(2)
        self.aux1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=3),
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(),
            nn.Conv2d(64, 256, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, class_num),
            nn.LogSoftmax(dim=1)
            )
        self.aux2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(),
            nn.Conv2d(128, 512, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, class_num),
            nn.LogSoftmax(dim=1)
            )
        self.aux3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=3),
            nn.Conv2d(1792, 224, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(),
            nn.Conv2d(224, 1792, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(1792, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1792, class_num),
            nn.LogSoftmax(dim=1)
            )
        self.final_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Dropout(p=0.2),
            nn.Flatten(),
            nn.Linear(4480, class_num),
            nn.LogSoftmax(dim=1)
            )
    
    def forward(self, x):
        out1 = self.ExtractedDenseBlock1(x)
        out2 = self.ExtractedDenseBlock2(x)
        out3 = self.ExtractedDenseBlock3(x)
        out4 = self.ExtractedDenseBlock4(x)
        out1_1 = self.downSamp1(out1)
        out2_1 = self.downSamp2(out2)
        out3_1 = self.downSamp3(out3)
        out_cat = torch.cat((out1_1, out2_1, out3_1, out4), dim=1)
        final_out1 = self.aux1(out1)
        final_out2 = self.aux2(out2)
        final_out3 = self.aux3(out3)
        final_out4 = self.final_classifier(out_cat)
        return final_out4, final_out3, final_out2, final_out1


class Modified_InceptionV3(nn.Module):
    def __init__(self, class_num):
        super(Modified_InceptionV3, self).__init__()
        model = models.inception_v3(weights='DEFAULT')
        self.ExtractedInceptionBlocks1 = nn.Sequential(*list(model.children())[0:10])
        self.ExtractedInceptionBlocks2 = nn.Sequential(*list(model.children())[0:15])
        self.ExtractedInceptionBlocks3 = nn.Sequential(*list(model.children())[0:15],*list(model.children())[16:19])
        self.downSamp1 = nn.MaxPool2d(4)
        self.downSamp2 = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Dropout(p=0.2),
            nn.Flatten(),
            nn.Linear(3104, class_num),
            nn.LogSoftmax(dim=1)
            )
    
    def forward(self, x):
        out1 = self.ExtractedInceptionBlocks1(x)
        out2 = self.ExtractedInceptionBlocks2(x)
        out3 = self.ExtractedInceptionBlocks3(x)
        out1 = self.downSamp1(out1)
        out2 = self.downSamp2(out2)
        out_cat = torch.cat((out1,out2,out3), dim=1)
        out = self.classifier(out_cat)
        return out


class Modified_InceptionV3_AUX(nn.Module):
    def __init__(self, class_num):
        super(Modified_InceptionV3_AUX, self).__init__()
        model = models.inception_v3(weights='DEFAULT')
        self.ExtractedInceptionBlocks1 = nn.Sequential(*list(model.children())[0:10])
        self.ExtractedInceptionBlocks2 = nn.Sequential(*list(model.children())[0:15])
        self.ExtractedInceptionBlocks3 = nn.Sequential(*list(model.children())[0:15],*list(model.children())[16:19])
        self.downSamp1 = nn.MaxPool2d(4)
        self.downSamp2 = nn.MaxPool2d(2)
        self.aux1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=3),
            nn.Conv2d(288, 72, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(72, eps=1e-03, momentum=0.1, affine=True),
            nn.ReLU6(),
            nn.Conv2d(72, 288, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(288, eps=1e-03, momentum=0.1, affine=True),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(288, class_num),
            nn.LogSoftmax(dim=1)
            )
        self.aux2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(768, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128, eps=1e-03, momentum=0.1, affine=True),
            nn.ReLU6(),
            nn.Conv2d(128, 768, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(768, eps=1e-03, momentum=0.1, affine=True),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(768, class_num),
            nn.LogSoftmax(dim=1)
            )
        self.final_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Dropout(p=0.2),
            nn.Flatten(),
            nn.Linear(3104, class_num),
            nn.LogSoftmax(dim=1)
            )
    
    def forward(self, x):
        out1 = self.ExtractedInceptionBlocks1(x)
        out2 = self.ExtractedInceptionBlocks2(x)
        out3 = self.ExtractedInceptionBlocks3(x)
        out1_1 = self.downSamp1(out1)
        out2_1 = self.downSamp2(out2)
        out_cat = torch.cat((out1_1,out2_1,out3), dim=1)
        final_out1 = self.aux1(out1)
        final_out2 = self.aux2(out2)
        final_out3 = self.final_classifier(out_cat)
        return final_out3, final_out2, final_out1


class DenseInception_AUX(nn.Module):
    def __init__(self, class_num):
        super(DenseInception_AUX, self).__init__()
        model_densenet201 = models.densenet201(weights='DEFAULT')
        model_inceptionv3 = models.inception_v3(weights='DEFAULT')
        self.ExtractedDenseBlock = nn.Sequential(
          *list(model_densenet201.features.children()),
          nn.MaxPool2d(2, stride=None, padding=0, dilation=1)
          )
        self.ExtractedInceptionBlock = nn.Sequential(
          *list(model_inceptionv3.children())[0:15],*list(model_inceptionv3.children())[16:19],
          nn.MaxPool2d(2, stride=None, padding=0, dilation=1)
          )
        self.final_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Dropout(p=0.2),
            nn.Flatten(),
            nn.Linear(3968, class_num),
            nn.LogSoftmax(dim=1)
            )
        self.aux1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1920, class_num),
            nn.LogSoftmax(dim=1)
            )
        self.aux2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048, class_num),
            nn.LogSoftmax(dim=1)
            )
    
    def forward(self, x):
        out_densenet201 = self.ExtractedDenseBlock(x)
        out_inceptionv3 = self.ExtractedInceptionBlock(x)
        out_inceptionv3 = nn.ZeroPad2d((1,0,1,0))(out_inceptionv3)
        out_cat = torch.cat((out_densenet201, out_inceptionv3), dim=1)
        final_out1 = self.final_classifier(out_cat)
        final_out2 = self.aux1(out_densenet201)
        final_out3 = self.aux2(out_inceptionv3)
        return final_out1, final_out2, final_out3


class DenseMobileNet_AUX(nn.Module):
    def __init__(self, class_num):
        super(DenseMobileNet_AUX, self).__init__()
        model_densenet201 = models.densenet201(weights='DEFAULT')
        model_mobilenet_v2 = models.mobilenet_v2(weights='DEFAULT')
        self.ExtractedDenseBlock = nn.Sequential(model_densenet201.features)
        self.ExtractedMobileNetBlock = nn.Sequential(model_mobilenet_v2.features)
        self.final_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Dropout(p=0.2),
            nn.Flatten(),
            nn.Linear(3200, class_num),
            nn.LogSoftmax(dim=1)
            )
        self.aux1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1920, class_num),
            nn.LogSoftmax(dim=1)
            )
        self.aux2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1280, class_num),
            nn.LogSoftmax(dim=1)
            )
    
    def forward(self, x):
        out_densenet201 = self.ExtractedDenseBlock(x)
        out_mobilenetv2 = self.ExtractedMobileNetBlock(x)
        out_mobilenetv2 = nn.MaxPool2d(2, stride=2)(out_mobilenetv2)
        out_cat = torch.cat((out_densenet201, out_mobilenetv2), dim=1)
        final_out1 = self.final_classifier(out_cat)
        final_out2 = self.aux1(out_densenet201)
        final_out3 = self.aux2(out_mobilenetv2)
        return final_out1, final_out2, final_out3


class PHCNet(nn.Module):
    def __init__(self, class_num):
        super(PHCNet, self).__init__()
        model1 = models.densenet201(weights='DEFAULT')
        model2 = models.inception_v3(weights='DEFAULT')
        self.ExtractedDenseBlock1 = nn.Sequential(*list(model1.features.children())[0:7])
        self.ExtractedDenseBlock2 = nn.Sequential(*list(model1.features.children())[0:9])
        self.ExtractedDenseBlock3 = nn.Sequential(*list(model1.features.children())[0:])
        self.ExtractedInceptionBlocks1 = nn.Sequential(*list(model2.children())[0:10])
        self.ExtractedInceptionBlocks2 = nn.Sequential(*list(model2.children())[0:15])
        self.ExtractedInceptionBlocks3 = nn.Sequential(*list(model2.children())[0:15],*list(model2.children())[16:19])
        self.zeropad1 = nn.ZeroPad2d((1, 1, 1, 1))
        self.zeropad2 = nn.ZeroPad2d((1, 0, 1, 0))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.aux1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(800, 200, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(),
            nn.Conv2d(200, 800, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(800, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(800, class_num),
            nn.LogSoftmax(dim=1)
            )
        self.aux2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=3),
            nn.Conv2d(2560, 320, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(),
            nn.Conv2d(320, 2560, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(2560, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2560, class_num),
            nn.LogSoftmax(dim=1)
            )
        self.final_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(3968, class_num),
            nn.LogSoftmax(dim=1)
            )
    def forward(self, x):
        Dout1 = self.ExtractedDenseBlock1(x)
        Dout2 = self.ExtractedDenseBlock2(x)
        Dout3 = self.ExtractedDenseBlock3(x)
        Iout1 = self.upsample(self.zeropad1(torch.max_pool2d(self.ExtractedInceptionBlocks1(x),2)))
        Iout2 = self.upsample(self.zeropad2(torch.max_pool2d(self.ExtractedInceptionBlocks2(x),2)))
        Iout3 = self.zeropad1(self.ExtractedInceptionBlocks3(x))
        out1 = torch.cat((Dout1, Iout1), dim=1)
        out2 = torch.cat((Dout2, Iout2), dim=1)
        out3 = torch.cat((Dout3, Iout3), dim=1)
        final_out1 = self.aux1(out1)
        final_out2 = self.aux2(out2)
        final_out3 = self.final_classifier(out3)
        return final_out3, final_out2, final_out1
    

class DenseInceptionResNet_AUX(nn.Module):
    def __init__(self, class_num):
        super(DenseInceptionResNet_AUX, self).__init__()
        model1 = timm.create_model('densenet201', pretrained=True)
        model2 = timm.create_model('inception_resnet_v2', pretrained=True)
        self.ExtractedDenseBlock1 = nn.Sequential(*list(model1.features.children())[0:7])
        self.ExtractedDenseBlock2 = nn.Sequential(*list(model1.features.children())[0:9])
        self.ExtractedDenseBlock3 = nn.Sequential(*list(model1.features.children())[0:])
        self.ExtractedInceptionResNetV2Block1 = nn.Sequential(*list(model2.children())[0:9])
        self.ExtractedInceptionResNetV2Block2 = nn.Sequential(*list(model2.children())[0:11])
        self.ExtractedInceptionResNetV2Block3 = nn.Sequential(*list(model2.children())[0:14])
        self.zeropad1 = nn.ZeroPad2d((1, 1, 1, 1))
        self.zeropad2 = nn.ZeroPad2d((1, 0, 1, 0))
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')
        self.aux1 = nn.Sequential(
            nn.Conv2d(576, 144, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(144, class_num, kernel_size=1, stride=1, padding=0),
            nn.Flatten(),
            nn.LogSoftmax(dim=1)
            )
        self.aux2 = nn.Sequential(
            nn.Conv2d(1984, 496, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(496, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(496, class_num, kernel_size=1, stride=1, padding=0),
            nn.Flatten(),
            nn.LogSoftmax(dim=1)
            )
        self.final_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(4000, class_num),
            nn.LogSoftmax(dim=1)
            )
    def forward(self, x):
        Dout1 = self.ExtractedDenseBlock1(x)
        Dout2 = self.ExtractedDenseBlock2(x)
        Dout3 = self.ExtractedDenseBlock3(x)
        Iout1 = self.zeropad2(torch.max_pool2d(self.ExtractedInceptionResNetV2Block1(x),2))
        Iout2 = self.zeropad2(torch.max_pool2d(self.ExtractedInceptionResNetV2Block2(x),2))
        Iout3 = self.zeropad2(self.ExtractedInceptionResNetV2Block3(x))
        out1 = torch.cat((Dout1, Iout1), dim=1)
        out2 = torch.cat((Dout2, Iout2), dim=1)
        out3 = torch.cat((Dout3, Iout3), dim=1)
        final_out1 = self.aux1(out1)
        final_out2 = self.aux2(out2)
        final_out3 = self.final_classifier(out3)
        return final_out3, final_out2, final_out1


def cnn_V1(input_ch, class_num): 
    model = torch.nn.Sequential(
        # 1st layer (conv) 
        torch.nn.Conv2d(input_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 2nd layer (conv)
        torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 3rd layer (conv)
        torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
        torch.nn.ReLU(inplace=True),
        # 4th layer (conv)
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 5th layer (conv)
        torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 6th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 7th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 8th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # Average pooling 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(), 
        # 9th layer (MLP)
        torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.5, inplace=False),
        # 10th layer (MLP)
        torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.5, inplace=False),
        # 11th layer (MLP)  
        torch.nn.Linear(in_features=4096, out_features=class_num, bias=True), 
        torch.nn.LogSoftmax(dim=1) 
    )  
    #
    return model 


def cnn_V2(input_ch, class_num): 
    model = torch.nn.Sequential(
        # 1st layer (conv) 
        torch.nn.Conv2d(input_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 2nd layer (conv)
        torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 3rd layer (conv)
        torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
        torch.nn.ReLU(inplace=True),
        # 4th layer (conv)
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 5th layer (conv)
        torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 6th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 7th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 8th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # Average pooling 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(), 
        # 9th layer (MLP)
        torch.nn.Linear(in_features=25088, out_features=256, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.2, inplace=False),
        # 10th layer (MLP)   
        torch.nn.Linear(in_features=256, out_features=class_num, bias=True), 
        torch.nn.LogSoftmax(dim=1) 
    )
    #
    return model 


def cnn_V3(input_ch, class_num): 
    model = torch.nn.Sequential(
        # 1st layer (conv)
        torch.nn.Conv2d(input_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 2nd layer (conv)
        torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 3rd layer (conv)
        torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
        torch.nn.ReLU(inplace=True),
        # 4th layer (conv)
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 5th layer (conv)
        torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 6th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 7th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 8th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # Average pooling 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(), 
        # 9th layer (MLP)
        torch.nn.Linear(in_features=25088, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    )
    #
    return model 


def cnn_V4(input_ch, class_num): 
    model = torch.nn.Sequential(
        # 1st layer (conv)
        torch.nn.Conv2d(input_ch, 16, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        # Average pooling 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(), 
        # 2nd layer (MLP)
        # conv_output = 7*7*20= 980 
        torch.nn.Linear(in_features=3136, out_features=class_num, bias=True),  
        torch.nn.LogSigmoid()
    )
    #
    return model 


def SelfONN_1(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   

        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=75,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.BatchNorm2d(75, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=75,out_channels=56,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(),  
    
        # Output layer (MLP)  
        
        torch.nn.Linear(in_features=2744, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 


def SelfONN_2(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(4),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
        # Output layer (MLP)  
        torch.nn.Linear(in_features=784, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 


def SelfONN_2_Dense(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # Output layer (Self-MLP) 
        SelfONNLayer(in_channels=16,out_channels=class_num,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q_order,mode='fast'), 
        SqueezeLayer(),
        torch.nn.LogSigmoid()
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 


def SelfONN_3(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast',dropout=0.2),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast',dropout=0.2),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(), 
        # 5th layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        #torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 6th layer (conv)
        SelfONNLayer(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
        # Output layer (MLP)
        torch.nn.Dropout(p=0.2, inplace=False),  
        torch.nn.Linear(in_features=1568, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 


def SelfONN_3_SelfDense(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(), 
        # 5th layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 6th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # Output layer (Self-MLP) 
        SelfONNLayer(in_channels=16,out_channels=class_num,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q_order,mode='fast'), 
        SqueezeLayer(),
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 


def SelfONN_4(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(), 
        # 5th layer (conv) 
        SelfONNLayer(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 6th layer (conv)
        SelfONNLayer(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(3),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
        # Output layer (MLP)  
        torch.nn.Linear(in_features=512, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 


def SelfONN_5(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(), 
        # 5th layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 6th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(), 
        # 7th layer (conv) 
        SelfONNLayer(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 8th layer (conv)
        SelfONNLayer(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(3),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
        # Output layer (MLP)  
        torch.nn.Linear(in_features=512, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 


def dwise_conv(ch_in, stride=1):
    return (
        nn.Sequential(
            #depthwise
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU6(inplace=True),
        )
    )


def conv1x1(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )


def conv3x3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )
    
    
class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        self.stride = stride
        assert stride in [1,2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride==1 and ch_in==ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        layers.extend([
            #dw
            dwise_conv(hidden_dim, stride=stride),
            #pw
            conv1x1(hidden_dim, ch_out)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)
        
        
class RODNet(nn.Module):
    def __init__(self, input_ch=3, class_num=10):
        super(RODNet, self).__init__()

        self.configs=[
            # t, c, n, s
            [1, 16, 2, 1],
            [6, 24, 3, 2],
            [6, 48, 4, 2],
            [6, 16, 2, 1],
        ]

        self.stem_conv = conv3x3(input_ch, 16, stride=2)

        layers = []
        input_channel = 16
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
                input_channel = c

        self.layers = nn.Sequential(*layers)
        self.pool1 = nn.MaxPool2d(4)
        self.conv1 = conv3x3(32, 64, stride=2)
        self.last_conv = conv1x1(64, 1280)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(1280, class_num, bias=True),
                                        nn.LogSigmoid()
                                        )

    def forward(self, x):
        x1 = self.stem_conv(x)
        x2 = self.layers(x1)
        x3 = self.pool1(x1)
        x_cat = torch.cat((x2,x3), 1)
        x4 = self.conv1(x_cat)
        x5 = self.last_conv(x4)
        x6 = self.avg_pool(x5).view(-1, 1280)
        out = self.classifier(x6)
        return out
    

def get_pretrained_model(parentdir, model_name, model_mode, isPretrained, input_ch, class_num, final_activation_func, train_on_gpu, multi_gpu, q_order):
    if model_mode == "custom_CNN":
        if model_name == 'cnn_V1': 
            model = cnn_V1(input_ch,class_num)  
        
        elif model_name == 'cnn_V2': 
            model = cnn_V2(input_ch,class_num)  

        elif model_name == 'cnn_V3': 
            model = cnn_V3(input_ch,class_num)  

        elif model_name == 'cnn_V4': 
            model = cnn_V4(input_ch,class_num)  

        elif model_name == 'cnn_V5':   
            model = cnn_V5(input_ch,class_num)

        elif model_name == 'Modified_DenseNet201_AUX':   
            model = Modified_DenseNet201_AUX(class_num)

        elif model_name == 'Modified_InceptionV3':   
            model = Modified_InceptionV3(class_num)

        elif model_name == 'Modified_InceptionV3_AUX':   
            model = Modified_InceptionV3_AUX(class_num)

        elif model_name == 'DenseInception_AUX':   
            model = DenseInception_AUX(class_num)
        
        elif model_name == 'DenseMobileNet_AUX':   
            model = DenseMobileNet_AUX(class_num)

        elif model_name == 'PHCNet':   
            model = PHCNet(class_num)
            
        elif model_name == 'DenseInceptionResNet_AUX':   
            model = DenseInceptionResNet_AUX(class_num)
            
        elif model_name == 'RODNet':
            model = RODNet(input_ch, class_num)

        else:
            raise ValueError('The requested Custom CNN model is not currently available or there is a typo. Please also check the configs.') 
    
    elif model_mode == "custom_ONN":
        if model_name == 'SelfONN_1': 
            model = SelfONN_1(input_ch, class_num, q_order) 
        
        elif model_name == 'SelfONN_2': 
            model = SelfONN_2(input_ch, class_num, q_order) 

        elif model_name == 'SelfONN_2_Dense':  
            model = SelfONN_2_Dense(input_ch, class_num, q_order) 
        
        elif model_name == 'SelfONN_3':  
            model = SelfONN_3(input_ch, class_num, q_order) 

        elif model_name == 'SelfONN_3_SelfDense':  
            model = SelfONN_3_SelfDense(input_ch, class_num, q_order)

        elif model_name == 'SelfONN_4':  
            model = SelfONN_4(input_ch, class_num, q_order)

        elif model_name == 'SelfONN_5':  
            model = SelfONN_5(input_ch, class_num, q_order)

        elif model_name == 'Self_MobileNet':
            model = Self_MobileNet(input_channel=input_ch, last_layer_channel=32, class_num=class_num)

        elif model_name == 'Self_DenseMobileNet':
            model = Self_DenseMobileNet(input_channel=input_ch, last_layer_channel=32, class_num=class_num, q_order=q_order)

        else:
            raise ValueError('The requested Custom Self-ONN model is not currently available or there is a typo. Please also check the configs.') 

    elif model_mode == "import_Torch":
        if (isPretrained == True):
            pretrained_weights = 'DEFAULT'  # 'IMAGENET1K_V2' or 'DEFAULT'
        else:
            pretrained_weights = None
        
        if model_name == 'alexnet':
            model = models.alexnet(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)
        
        elif model_name == 'convnext_tiny':
            model = models.convnext_tiny(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'convnext_small':
            model = models.convnext_small(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'convnext_base':
            model = models.convnext_base(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'convnext_large':
            model = models.convnext_large(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)
        
        elif model_name == 'vgg11':
            model = models.vgg11(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'vgg11_bn':
            model = models.vgg11_bn(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'vgg13':
            model = models.vgg13(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'vgg13_bn':
            model = models.vgg13_bn(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'vgg16':
            model = models.vgg16(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'vgg16_bn':
            model = models.vgg16_bn(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'vgg19':
            model = models.vgg19(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'vgg19_bn':
            model = models.vgg19_bn(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)
        
        elif model_name == 'resnet18':
            model = models.resnet18(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'resnet34':
            model = models.resnet34(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'resnet50':
            model = models.resnet50(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'resnet101':
            model = models.resnet101(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'resnet152':
            model = models.resnet152(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'regnet_y_400mf':
            model = models.regnet_y_400mf(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'regnet_y_800mf':
            model = models.regnet_y_800mf(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'regnet_y_1_6gf':
            model = models.regnet_y_1_6gf(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'regnet_y_3_2gf':
            model = models.regnet_y_3_2gf(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'regnet_y_8gf':
            model = models.regnet_y_8gf(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'regnet_y_16gf':
            model = models.regnet_y_16gf(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'regnet_y_32gf':
            model = models.regnet_y_32gf(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'regnet_y_128gf':
            model = models.regnet_y_128gf(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'regnet_x_400mf':
            model = models.regnet_x_400mf(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'regnet_x_800mf':
            model = models.regnet_x_800mf(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'regnet_x_1_6gf':
            model = models.regnet_x_1_6gf(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'regnet_x_3_2gf':
            model = models.regnet_x_3_2gf(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'regnet_x_8gf':
            model = models.regnet_x_8gf(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'regnet_x_16gf':
            model = models.regnet_x_16gf(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'regnet_x_32gf':
            model = models.regnet_x_32gf(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'densenet121':
            model = models.densenet121(weights=pretrained_weights)
            num_ftrs = model.classifier.in_features
            model.classifier = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'densenet161':
            model = models.densenet161(weights=pretrained_weights)
            num_ftrs = model.classifier.in_features
            model.classifier = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'densenet169':
            model = models.densenet169(weights=pretrained_weights)
            num_ftrs = model.classifier.in_features
            model.classifier = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'densenet201':
            model = models.densenet201(weights=pretrained_weights)
            num_ftrs = model.classifier.in_features
            model.classifier = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'inception_v3':
            model = models.inception_v3(weights=pretrained_weights)
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'mobilenet_v3_small':
            model = models.mobilenet_v3_small(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'mobilenet_v3_large':
            model = models.mobilenet_v3_large(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)
        
        elif model_name == 'shufflenet_v2_x0_5':
            model = models.shufflenet_v2_x0_5(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'shufflenet_v2_x1_0':
            model = models.shufflenet_v2_x1_0(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'shufflenet_v2_x1_5':
            model = models.shufflenet_v2_x1_5(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'shufflenet_v2_x2_0':
            model = models.shufflenet_v2_x2_0(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'googlenet':
            model = models.googlenet(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'squeezenet1_0':
            model = models.squeezenet1_0(weights=pretrained_weights)
            model.classifier[-1] = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            CNN_Classifier(in_channels=1000, class_num=class_num, final_activation=final_activation_func))

        elif model_name == 'squeezenet1_1':
            model = models.squeezenet1_1(weights=pretrained_weights)
            model.classifier[-1] = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            CNN_Classifier(in_channels=1000, class_num=class_num, final_activation=final_activation_func))

        elif model_name == 'resnext-50-32x4d':
            model = models.resnext50_32x4d(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'resnext-101-32x8d':
            model = models.resnext101_32x8d(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'wide_resnet-50-2':
            model = models.wide_resnet50_2(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'wide_resnet-101-2':
            model = models.wide_resnet101_2(weights=pretrained_weights)
            num_ftrs = model.fc.in_features
            model.fc = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'mnasnet0_5':
            model = models.mnasnet0_5(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'mnasnet0_75':
            model = models.mnasnet0_75(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'mnasnet1_0':
            model = models.mnasnet1_0(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'mnasnet1_3':
            model = models.mnasnet1_3(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'maxvit_t':
            model = models.maxvit_t(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'efficientnet_b1':
            model = models.efficientnet_b1(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'efficientnet_b2':
            model = models.efficientnet_b2(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'efficientnet_b3':
            model = models.efficientnet_b3(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'efficientnet_b4':
            model = models.efficientnet_b4(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'efficientnet_b5':
            model = models.efficientnet_b5(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'efficientnet_b6':
            model = models.efficientnet_b6(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'efficientnet_b7':
            model = models.efficientnet_b7(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'efficientnet_v2_s':
            model = models.efficientnet_v2_s(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'efficientnet_v2_m':
            model = models.efficientnet_v2_m(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'efficientnet_v2_l':
            model = models.efficientnet_v2_l(weights=pretrained_weights)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'swin_t':
            model = models.swin_t(weights=pretrained_weights)
            num_ftrs = model.head.in_features
            model.head = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'swin_s':
            model = models.swin_s(weights=pretrained_weights)
            num_ftrs = model.head.in_features
            model.head = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'swin_b':
            model = models.swin_b(weights=pretrained_weights)
            num_ftrs = model.head.in_features
            model.head = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'swin_v2_t':
            model = models.swin_v2_t(weights=pretrained_weights)
            num_ftrs = model.head.in_features
            model.head = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'swin_v2_s':
            model = models.swin_v2_s(weights=pretrained_weights)
            num_ftrs = model.head.in_features
            model.head = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

        elif model_name == 'swin_v2_b':
            model = models.swin_v2_b(weights=pretrained_weights)
            num_ftrs = model.head.in_features
            model.head = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)
        
        else:
            raise ValueError('The requested Pretrained CNN model is not currently available in the PyTorch repository or there is a typo. Please also check the configs.') 

    elif model_mode == "import_TIMM":
        model = timm.create_model(model_name, pretrained=isPretrained, in_chans=input_ch, num_classes=class_num)
        num_ftrs = model.get_classifier().in_features
        model.classifier = CNN_Classifier(in_channels=num_ftrs, class_num=class_num, final_activation=final_activation_func)

    else:
        raise ValueError('The requested MODE is not currently available in the pipeline or there is a typo. Please also check the configs.') 
    
    # Move to gpu and parallelize
    if train_on_gpu:
        model = model.to('cuda')
    if multi_gpu:
        model = nn.DataParallel(model)

    return model
