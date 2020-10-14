import torch
import numpy as np
import torchvision
import torch.nn.functional as F
from torch import nn
# from torchsummary import summary
from PIL import Image
# from abc import ABC, abstractmethod
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from loss.metrics import ArcFace, CosFace, SphereFace, Am_softmax

head_dict = {'ArcFace': ArcFace(in_features=1000, out_features=2, device_id=[0]),
             'CosFace': CosFace(in_features=1000, out_features=2, device_id=[0]),
             'SphereFace': SphereFace(in_features=1000, out_features=2, device_id=[0]),
             'Am_softmax': Am_softmax(in_features=1000, out_features=2, device_id=[0])}


class CelebASpoofDetector():
    def __init__(self):
        """
        Participants should define their own initialization process.
        During this process you can set up your network. The time cost for this step will
        not be counted in runtime evaluation
        """
        self.num_class = 2
        self.net = EfficientNet.from_pretrained('efficientnet-b4')
        self.head = head_dict['ArcFace']
        # self.net._fc.out_features = self.num_class, this method is incorect
        feature = self.net._fc.in_features
        # print(feature)
        # self.net._fc = nn.Linear(in_features=feature,out_features=2,bias=True)
        # self.net = self.net.to("cuda:0")
        # summary(self.net,(3,224,224))
        checkpoint = torch.load(
            './model/efficientModel/net_001.pth', map_location=torch.device('cpu'))
        checkpoint_head = torch.load(
            './model/efficientModel/net_head_001.pth', map_location=torch.device('cpu'))
        # print(checkpoint.items())
        self.net.load_state_dict(checkpoint, strict=False)

        # state_dict = torch.load('./model/efficientModel/net_014.pth')
        self.net.state_dict().update(checkpoint.items())

        self.head.load_state_dict(checkpoint_head)
        '''
        for name, param in checkpoint.items():
            if 1==1:
                print('t')
            elif name not in self.net.state_dict:
                print('not in')
                continue
            else:
                # backwards compatibility for serialized parameters
                param = param.data
            self.net.state_dict[name].copy_(param)
        '''
        '''=if not re define fc layer output by nn.Linear
        with torch.no_grad():
            self.net._fc.weight.copy_(checkpoint['module._fc.weight'])
            self.net._fc.bias.copy_(checkpoint['module._fc.bias'])
        '''
        # print(self.net)
        self.new_width = self.new_height = 224
        self.transform = transforms.Compose([
            transforms.Resize((self.new_width, self.new_height)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        # if torch.cuda.is_available():
        #     self.net.cuda()
        self.net.eval()
        self.head.eval()

    def preprocess_data(self, image):
        processed_data = Image.fromarray(image)
        processed_data = self.transform(processed_data)
        return processed_data

    def eval_image(self, image):
        data = torch.stack(image, dim=0)
        channel = 3
        # if torch.cuda.is_available():
        #     input_var = data.view(-1, channel, data.size(2), data.size(3)).cuda()
        # else:
        #     input_var = data.view(-1, channel, data.size(2), data.size(3))
        input_var = data.view(-1, channel, data.size(2), data.size(3))
        with torch.no_grad():
            # used in efficientnet with softmax method
            # rst = self.net(input_var).detach()[:,:2]
            # used in efficientnet with arcface method
            rst = self.net(input_var).detach()[:, :]
        # used in efficientnet with softmax method
        # return rst.reshape(-1, self.num_class)
        return rst

    def forwardWoM(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        '''
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1) 
        '''
        cosine = F.linear(F.normalize(input), F.normalize(self.head.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        '''
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        '''
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        # if self.device_id != None:
        #     one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output = (one_hot * cosine) + ((1.0 - one_hot) * cosine)
        output *= 64

        return output

    def predict(self, images):
        real_data = []
        for image in images:
            data = self.preprocess_data(image)
            real_data.append(data)
        rst = self.eval_image(real_data)
        theta = self.forwardWoM(rst,torch.zeros(len(rst)))
        y = F.softmax(theta, dim=1)


        '''
        y = F.linear(F.normalize(rst), F.normalize(self.head.weight))
        y = np.abs(y.detach().numpy())
        for index in range(len(y)):
            if y[index][0] > y[index][1]:
                # y[index][1] = y[index][0]
                y[index][1] = 1
                y[index][0] = 0
            else:
                # y[index][0] = y[index][1]
                y[index][0] = 1
                y[index][1] = 0
        '''
        '''
        thetasA = self.head(rst,torch.zeros(len(rst),device=torch.device("cuda:0")))        
        thetasB = self.head(rst,torch.ones(len(rst),device=torch.device("cuda:0")))

        for idx in range(14):
            print(y[idx])
            print(thetasA[idx])
            print(thetasB[idx])
        '''
        return y
