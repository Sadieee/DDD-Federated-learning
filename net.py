import torch
from math import sqrt
import torch.nn as nn
#from torchvision.models.utils import load_state_dict_from_url (這行沒辦法用 改下行)
from torch.hub import load_state_dict_from_url
from torchvision import models
# https://arxiv.org/abs/1409.1556

#-----------lrcn----------------#
#vgg-16
class SE_VGG(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super(SE_VGG, self).__init__()
        self.num_classes = num_classes
        # define an empty for Conv_ReLU_MaxPool
        net = []
        
        # block 1 5
        net.append(nn.Conv3d(in_channels=3, out_channels=64, padding='same', kernel_size=(3, 3, 3)))
        net.append(nn.ReLU())
        net.append(nn.Conv3d(in_channels=64, out_channels=64, padding='same', kernel_size=3))
        net.append(nn.ReLU())
        net.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))

        # block 2 5
        net.append(nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding='same'))
        net.append(nn.ReLU())
        net.append(nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding='same'))
        net.append(nn.ReLU())
        net.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        
        # block 3 7
        net.append(nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding='same'))
        net.append(nn.ReLU())
        net.append(nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding='same'))
        net.append(nn.ReLU())
        net.append(nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding='same'))
        net.append(nn.ReLU())
        net.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        
        # add net into class property
        self.extract_feature1 = nn.Sequential(*net)
        
        # block 4 7
        net2 = []
        net2.append(nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, padding='same'))
        net2.append(nn.ReLU())
        net2.append(nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding='same'))
        net2.append(nn.ReLU())
        net2.append(nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding='same'))
        net2.append(nn.ReLU())
        net2.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2))

        # add net into class property
        self.extract_feature2 = nn.Sequential(*net2)

        # block 5 7
        net3 = []
        net3.append(nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding='same'))
        net3.append(nn.ReLU())
        net3.append(nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding='same'))
        net3.append(nn.ReLU())
        net3.append(nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, padding='same'))
        net3.append(nn.ReLU())
        net3.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2))

        # add net into class property
        self.extract_feature3 = nn.Sequential(*net3)

        # 用1層conv和1層maxpool弄成輸出256 maxpool 4層
        net256_4 = []
        net256_4.append(nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding='same'))
        net256_4.append(nn.BatchNorm3d(num_features=256))
        net256_4.append(nn.ReLU())
        net256_4.append(nn.MaxPool3d(kernel_size=(1, 16, 16), stride=16))     
        self.conv_pool = nn.Sequential(*net256_4)        

        # 用1層conv和1層maxpool弄成輸出256 maxpool 3層
        net256_3 = []
        net256_3.append(nn.Conv3d(in_channels=512, out_channels=256, kernel_size=3, padding='same'))
        net256_3.append(nn.BatchNorm3d(num_features=256))
        net256_3.append(nn.ReLU())
        net256_3.append(nn.MaxPool3d(kernel_size=(1, 8, 8), stride=8))     
        self.conv_pool2 = nn.Sequential(*net256_3)      
   
        # 用1層conv和1層maxpool弄成輸出256
        net512 = []
        net512.append(nn.Conv3d(in_channels=512, out_channels=256, kernel_size=3, padding='same'))
        net512.append(nn.BatchNorm3d(num_features=256))
        net512.append(nn.ReLU())
        net512.append(nn.MaxPool3d(kernel_size=(1, 4, 4), stride=4))  
        self.conv_pool3 = nn.Sequential(*net512)          

        # define an empty container for Linear operations
        classifier = []
        classifier.append(nn.Flatten(start_dim=1, end_dim=-1))
        classifier.append(nn.Linear(in_features=1*1*768, out_features=768))
        #classifier.append(nn.BatchNorm1d(num_features=768))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=768, out_features=768))
        #classifier.append(nn.BatchNorm1d(num_features=768))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        #對最後一個全連接層作修改，最多可以輸出1000，這裡改成2類
        classifier.append(nn.Linear(in_features=768, out_features=self.num_classes))
        classifier.append(nn.Softmax(dim=-1))

        # add classifier into class property 提供最後的三個全連接操作
        self.classifier = nn.Sequential(*classifier)

    #負責實現前向傳播
    def forward(self, x):
        feature1 = self.extract_feature1(x)
        #print(feature1.shape) #256 56 56
        feature2 = self.extract_feature2(feature1)
        #print(feature2.shape) #512 28 28
        feature3 = self.extract_feature3(feature2)
        #print(feature3.shape)  #512 14 14
        re_feature1 = self.conv_pool(feature1)
        #print(re_feature1.shape) #256 1 1
        re_feature2 = self.conv_pool2(feature2)
        #print(re_feature2.shape) #256 1 1
        re_feature3 = self.conv_pool3(feature3)
        #print(re_feature3.shape) #256 1 1
        feature = torch.cat((re_feature1, re_feature2),1)
        feature = torch.cat((feature, re_feature3),1)
        #print(feature.shape)
        #feature應該是[batch, 768]
        classify_result = self.classifier(feature)
        return classify_result

class ConvLstm(nn.Module):
    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional, n_class):
        super(ConvLstm, self).__init__()
        self.conv_model = Pretrained_conv(latent_dim, n_class)

    def forward(self, x):
        batch_size, channel_x, time, h_x, w_x = x.shape
        output = self.conv_model(x)
        return output

#管理vgg16
class Pretrained_conv(nn.Module):
    def __init__(self, latent_dim, n_class):
        super(Pretrained_conv, self).__init__()
        #self.conv_model = models.resnet152(pretrained=True)
        #for param in self.conv_model.parameters():
            #param.requires_grad = False
        self.conv_model = SE_VGG(n_class)
        #self.conv_model.fc = nn.Linear(self.conv_model.fc.in_features, latent_dim)
        #self.conv_model = models.vgg16(pretrained=False)
        #self.conv_model.classifier[6] = nn.Linear(in_features=4096, out_features=latent_dim)
        # ====== freezing all of the layers ======
        #for param in self.conv_model.parameters():
        #    param.requires_grad = False
        # ====== changing the last FC layer to an output with the size we need. this layer is un freezed ======
        #self.conv_model.fc = nn.Linear(self.conv_model.fc.in_features, 2)
        #self.conv_model = SE_VGG(2) #num_class
        #載入預訓練模型
        #vgg = models.vgg16(pretrained=False)
        """
        self.conv_model = SE_VGG(2)
        #block1
        #self.model.extract_feature1[0].weight.data.copy_(vgg.features[0].weight.data)
        self.model.extract_feature1[0].bias.data.copy_(vgg.features[0].bias.data)
        #self.model.extract_feature1[1].weight.data = torch.randn(64) * sqrt(2.0/64)
        self.model.extract_feature1[2].weight.data.copy_(vgg.features[2].weight.data)
        self.model.extract_feature1[2].bias.data.copy_(vgg.features[2].bias.data)
        #self.model.extract_feature1[3].weight.data = torch.randn(n) * sqrt(2.0/n)
        #block2
        self.model.extract_feature1[5].weight.data.copy_(vgg.features[5].weight.data)
        self.model.extract_feature1[5].bias.data.copy_(vgg.features[5].bias.data)
        #self.model.extract_feature1[6].weight.data = torch.randn(n) * sqrt(2.0/n)
        self.model.extract_feature1[7].weight.data.copy_(vgg.features[7].weight.data)
        self.model.extract_feature1[7].bias.data.copy_(vgg.features[7].bias.data)
        #self.model.extract_feature1[8].weight.data = torch.randn(n) * sqrt(2.0/n)
        #block3
        self.model.extract_feature1[10].weight.data.copy_(vgg.features[10].weight.data)
        self.model.extract_feature1[10].bias.data.copy_(vgg.features[10].bias.data)
        #self.model.extract_feature1[11].weight.data = torch.randn(n) * sqrt(2.0/n)
        self.model.extract_feature1[12].weight.data.copy_(vgg.features[12].weight.data)
        self.model.extract_feature1[12].bias.data.copy_(vgg.features[12].bias.data)
        #self.model.extract_feature1[13].weight.data = torch.randn(n) * sqrt(2.0/n)
        self.model.extract_feature1[14].weight.data.copy_(vgg.features[14].weight.data)
        self.model.extract_feature1[14].bias.data.copy_(vgg.features[14].bias.data)
        #self.model.extract_feature1[15].weight.data = torch.randn(n) * sqrt(2.0/n)
        #block4
        self.model.extract_feature2[0].weight.data.copy_(vgg.features[17].weight.data)
        self.model.extract_feature2[0].bias.data.copy_(vgg.features[17].bias.data)
        #self.model.extract_feature2[1].weight.data = torch.randn(n) * sqrt(2.0/n)
        self.model.extract_feature2[2].weight.data.copy_(vgg.features[19].weight.data)
        self.model.extract_feature2[2].bias.data.copy_(vgg.features[19].bias.data)
        #self.model.extract_feature2[3].weight.data = torch.randn(n) * sqrt(2.0/n)
        self.model.extract_feature2[4].weight.data.copy_(vgg.features[21].weight.data)
        self.model.extract_feature2[4].bias.data.copy_(vgg.features[21].bias.data)
        #self.model.extract_feature2[5].weight.data = torch.randn(n) * sqrt(2.0/n)
        #block5
        self.model.extract_feature3[0].weight.data.copy_(vgg.features[24].weight.data)
        self.model.extract_feature3[0].bias.data.copy_(vgg.features[24].bias.data)
        #self.model.extract_feature3[1].weight.data = torch.randn(n) * sqrt(2.0/n)
        self.model.extract_feature3[2].weight.data.copy_(vgg.features[26].weight.data)
        self.model.extract_feature3[2].bias.data.copy_(vgg.features[26].bias.data)
        #self.model.extract_feature3[3].weight.data = torch.randn(n) * sqrt(2.0/n)
        self.model.extract_feature3[4].weight.data.copy_(vgg.features[28].weight.data)
        self.model.extract_feature3[4].bias.data.copy_(vgg.features[28].bias.data)
        """
        #self.model.extract_feature3[5].weight.data = torch.randn(n) * sqrt(2.0/n)
        #剩下的全部高斯隨機初始化
        #self.model.conv_pool[0].weight.data = 0.01* torch.randn(D,H)
        #self.model.conv_pool[1].weight.data = torch.randn(n) * sqrt(2.0/n)
        #freezed conv1-2
        #for i, param in enumerate(self.model.parameters()):
            #print(i, param)
            #if i <= 9:
             #   param.requires_grad = False
        #self.conv_model.load_state_dict()
        #self.conv_model = models.vgg16_bn()

        #VGG16-BN
        #self.conv_model = models.vgg16_bn(pretrained=True)
        #for param in self.conv_model.parameters():
        #    param.requires_grad = False
        #num_ftrs = self.conv_model.classifier.in_features
        #self.conv_model.classifier = nn.Linear(num_ftrs, 2)
        #new_classifier = torch.nn.Sequential(*list(self.conv_model.children())[-1][:6])
        #self.conv_model.classifier = new_classifier
        #self.vgg.classifier.append(nn.Linear(in_features=4096, out_features=2))
        """
        #========freezing all
        for param in self.conv_model.parameters():
            param.requires_grad = False
        
        #======= unfreezing 0到第9層
        
        for i, param in enumerate(self.conv_model.parameters()):
            if i <= 9:
                param.requires_grad = True
            print(param)
        """
        #self.model = SE_VGG( 2)
        #load 預訓練模型參數
        """
        pretrained_dict = self.conv_model().state_dict()
        print(pretrained_dict)
        #load 自己模型參數
        model_dict = self.model().state_dict()
        #丟掉不用的
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
        #更新參數
        model_dict.update(pretrained_dict)
        new_model_dict.load_state_dict(model_dict)
        """

    def forward(self, x):
        #pretrain_output = self.conv_model(x)
        output = self.conv_model(x)
        return output

class Lstm(nn.Module):
    def __init__(self, latent_dim, hidden_size, lstm_layers, bidirectional):
        super(Lstm, self).__init__()
        # lstm默認是batch在第二維
        #lstm input shape(seq_len, batch, input_size)
        self.Lstm = nn.LSTM(latent_dim, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        #lstm output shape(seq_len, batch, num_direction*hidden_size)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self,x):
        output, self.hidden_state = self.Lstm(x, self.hidden_state)
        return output

