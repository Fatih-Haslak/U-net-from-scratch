import torch.nn as nn
import cv2
import os,os.path
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch.optim import Adam
import torchvision
from torch.nn import functional as F
from torchvision import transforms
import argparse


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
print("--model_name 'your_model_name' ile modelinizi seçebilirsiniz")
print("Output dosya içeriği ve .pth uzanttılı kullanilabilir Modeller {}".format(os.listdir("output")))
parser = argparse.ArgumentParser(description='--model_name ile kullanıcagınız modeli seçin')

parser.add_argument('--model_name', type=str, default="256_model.pth",
                        help='Model name (varsayılan:128_model.pth)')
args = parser.parse_args()

model_name=args.model_name
print("Aktif çalışan model ---> ",model_name)
size_x=int(model_name.split("_")[0])
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.residual(x)+self.shortcut(x)

class encoder_block(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([ConvBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool = nn.MaxPool2d((2,2))
    
    def forward(self, x):

        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class decoder_block(nn.Module):
    
    def __init__(self, chs=(1024,512, 256, 128, 64)):
        super().__init__()
        self.chs  = chs
        self.upconvs  = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([ConvBlock(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)    
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size,hidden2,hidden3, output_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, output_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x= self.sigmoid(x)
        return x
    
class build_unet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False):
        super().__init__()
        
        self.encoder    = encoder_block(enc_chs)
        self.decoder    = decoder_block(dec_chs)
        self.regression = RegressionModel(1024,512,256,64,4)
        self.head       = nn.Conv2d(dec_chs[-1], num_class, 1)
        #self.sigmoid    = nn.Sigmoid()
        #######################################################
        self.retain_dim  = retain_dim

    def forward(self, x):
    
        enc_ftrs = self.encoder(x)
        _x=enc_ftrs[::-1][0]
        enc_features_flat = torch.mean(_x.view(_x.size(0), _x.size(1), -1), dim=2)
        out_reg  = self.regression(enc_features_flat)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)

        return out,out_reg


unet=(torch.load(("output"+"/"+model_name),map_location=torch.device(DEVICE)))

unet=unet.to(DEVICE)
unet.eval()
torch.no_grad()

images_dataset=r"C:\Users\90546\Desktop\my_everything"

transforms = transforms.Compose([transforms.ToPILImage(),
transforms.Resize((size_x,
    size_x)),
transforms.ToTensor()])

#datalar

for file in os.listdir(images_dataset):
        
        file=images_dataset+"/"+file
        
        img = cv2.imread(file)
        
        x =transforms(img).unsqueeze(0)
      
        pred,label = unet(x.to(DEVICE))
        
        pred=pred[0].cpu().detach().numpy().transpose(1, 2, 0)
    
        imgal=x[0].cpu().detach().numpy().transpose(1, 2, 0)
        
        imG=cv2.cvtColor(imgal, cv2.COLOR_BGRA2BGR)
        pred=cv2.cvtColor(pred, cv2.COLOR_GRAY2BGRA)
       
        
        label=label.cpu().detach().numpy()

        a,b,c,d=(abs(label[0][0]*size_x)),abs(label[0][1]*size_x),abs((label[0][2]*size_x)),abs(label[0][3]*size_x)
        a=int(a)
        b=int(b)
        c=int(c)
        d=int(d)

        cv2.rectangle(pred,(a,b),(c,d),(205,128,155),2)
        cv2.imshow("Real image",imG)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("Mask image",pred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
