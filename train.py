import cv2
import os,os.path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import argparse
import torch.nn as nn
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torchvision
import imutils
from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from pathlib import Path
from PIL import Image
import PIL
import itertools

mask_dataset="masks"
images_dataset="images"

count=0
c = 4
r=len([name for name in os.listdir(mask_dataset) if os.path.isfile(os.path.join(mask_dataset, name))])
path_list=[]
arr = np.ones((r,c)) # bboxlar için kullanılacak olan arrayın tanımlanması
flag=1 
x_file=os.listdir(images_dataset)[0]
x_file=mask_dataset+"/"+x_file
img_size = cv2.imread(x_file)    
a,b,_=img_size.shape
original_image_size = [a,b]

#argümanların ataması
def args_init(args):
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    TEST_SPLIT = args.test_split
    INIT_LR = args.lr
    BASE_OUTPUT = args.BASE_OUTPUT
    INPUT_IMAGE_WIDTH = args.INPUT_IMAGE_WIDTH
    INPUT_IMAGE_HEIGHT = args.INPUT_IMAGE_HEIGHT
    TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
    MODEL_PATH_BEST = os.path.join(BASE_OUTPUT, str(INPUT_IMAGE_HEIGHT) +"_bestmodel.pth")
    MODEL_PATH_LAST = os.path.join(BASE_OUTPUT, str(INPUT_IMAGE_HEIGHT) +"_lastmodel.pth")
    PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
    print("BATCH_SIZE: {}, NUM_EPOCHS: {}, LR: {}".format(BATCH_SIZE, NUM_EPOCHS, INIT_LR))

    return BATCH_SIZE,NUM_EPOCHS,TEST_SPLIT,INIT_LR,INPUT_IMAGE_WIDTH,INPUT_IMAGE_HEIGHT,TEST_PATHS,MODEL_PATH_BEST,MODEL_PATH_LAST,PLOT_PATH

## 2.BÖLÜM
class CustomDataset(Dataset):
    def __init__(self,imagePaths,maskPaths,transforms,bbox):
        #mask ve image'leri depolar + transform
        self.imagePaths=imagePaths
        self.maskPaths=maskPaths
        self.transforms=transforms
        self.bbox=bbox

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self,idx):
        
        imagePath = self.imagePaths[idx]
        image = cv2.imread(imagePath) 
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) 
        bbox=self.bbox[idx]
        bbox=torch.Tensor(([(bbox[0]/original_image_size[1]),(bbox[1]/original_image_size[0]),(bbox[2]/original_image_size[1]),(bbox[3]/original_image_size[0])])) # 0, 1
 
        mask = cv2.imread(self.maskPaths[idx], 0)
        

        
        if self.transforms is not None:
            image = self.transforms(image)
            mask = self.transforms(mask)
        
        return (image, mask, bbox)
## 3.BÖLÜM
#U-Net mimarisi + Conv Residual block olarak kullanıldı
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
#U netin encoder mimarisi
class encoder_block(nn.Module):
    #U-net encoder orjinal paper'den.
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

#U netin decoder mimarisi
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


##Bbox tahmını ıcın encoderin cıktısına regressıon model baglanıldı.
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
#modelin tamamlanması
class build_unet(nn.Module):
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False):
        super().__init__()
        
        self.encoder    = encoder_block(enc_chs)

        self.decoder    = decoder_block(dec_chs)
        
        self.regression = RegressionModel(1024,512,256,64,4)
        self.head       = nn.Conv2d(dec_chs[-1], num_class, 1)
        #self.sigmoid    = nn.Sigmoid()# Kullanılan loss'dan dolayı kullanılmasına gerek yok.
        self.retain_dim  = retain_dim

    def forward(self, x):
        
        enc_ftrs = self.encoder(x)
        _x=enc_ftrs[::-1][0]
        enc_features_flat = torch.mean(_x.view(_x.size(0), _x.size(1), -1), dim=2)
        out_reg  = self.regression(enc_features_flat)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
    
        return out,out_reg

#4 VE 5.BÖLÜM
def train(BATCH_SIZE,NUM_EPOCHS,TEST_SPLIT,INIT_LR,INPUT_IMAGE_WIDTH,INPUT_IMAGE_HEIGHT,TEST_PATHS,MODEL_PATH_BEST,MODEL_PATH_LAST,PLOT_PATH,DEVICE,df1):
    temp=0
    global flag
    global transforms
    imagePaths =images_dataset
    maskPaths = mask_dataset

    #verileri %85 train %15 test olarak ayır.
    X_train, X_test,y_train, y_test, bbox_train, bbox_test = train_test_split(os.listdir(imagePaths),os.listdir(maskPaths),df1.to_numpy(),
                                   random_state=34, 
                                   test_size=TEST_SPLIT)
   

    (trainImages, testImages)=X_train,X_test
    (trainMasks, testMasks) =y_train,y_test
    
    #verilerin full pathini veriyoruz
    trainImages = list(map(lambda orig_string: imagePaths+"/"+orig_string , trainImages))
    testImages = list(map(lambda orig_string: imagePaths+"/"+orig_string , testImages))
    trainMasks = list(map(lambda orig_string: maskPaths+"/"+orig_string , trainMasks))
    testMasks = list(map(lambda orig_string: maskPaths+"/"+orig_string , testMasks))

    bbox_train=torch.tensor(bbox_train) # bboxları tensora çeviriyoruz
    bbox_test=torch.tensor(bbox_test)
    #
    print("[INFO] saving testing image paths...")
    f = open(TEST_PATHS, "w")
    f.write("\n".join(testImages))
    f.close()
    #

    # pytorch ıcın gereklı olan transformlar belırlenmıstır.
    transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((INPUT_IMAGE_HEIGHT,
		INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])

    # train ve test datasetlerın ayarlanması
    trainDS = CustomDataset(imagePaths=trainImages, maskPaths=trainMasks,
        transforms=transforms,bbox=bbox_train)
    testDS = CustomDataset(imagePaths=testImages, maskPaths=testMasks,
        transforms=transforms,bbox=bbox_test)

    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")
    
    # Training ve test veri setlerinin düzenlenmesi
    
    trainLoader = DataLoader(trainDS, shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=os.cpu_count())
    
    testLoader = DataLoader(testDS, shuffle=False,
        batch_size=BATCH_SIZE, 
        num_workers=os.cpu_count())
    
    unet = build_unet().to(DEVICE)   

    lossFunc = nn.BCEWithLogitsLoss() #for mask
    opt = Adam(unet.parameters(), lr=INIT_LR)
    lossFunc2= nn.L1Loss() #for bbox
   
    trainSteps = len(trainDS) // BATCH_SIZE
    testSteps = len(testDS) // BATCH_SIZE

    H = {"train_loss": [], "test_loss": []}
    print("[INFO] training the network...")
    startTime = time.time()
    
    #pytoch train 
    for e in tqdm(range(NUM_EPOCHS)):
        #Eğitim için modeli train modunda başlatalım
        unet.train()
        totalTrainLoss = 0
        totalTestLoss = 0
        
        for (i, (x, y, labels)) in enumerate(trainLoader):
            
            (x, y, labels) = (x.to(DEVICE), y.to(DEVICE),labels.to(DEVICE))
            
            #images, mask, bounding boxes
    
            pred,pred_bbox= unet(x)
            
            loss = lossFunc(pred, y) #mask loss BCEWithLogitsLoss
            loss2= lossFunc2(pred_bbox,labels)#bbox loss L1Loss
            totalTrainLoss = (loss2 * 10) + loss #loss optimizasyonu için testler yapılmıştır. Bbox losu 10 ile çarpılmıştır.

            opt.zero_grad()
            totalTrainLoss.backward()
            opt.step()

        with torch.no_grad():

            unet.eval() #test modunda modeli çağırıyoruz
            for (i, (x, y, labels)) in enumerate(testLoader):

                (x, y,labels) = (x.to(DEVICE), y.to(DEVICE),labels.to(DEVICE))

                pred,pred_bbox= unet(x)
                loss = lossFunc(pred, y) # mask loss
                loss2= lossFunc2(pred_bbox,labels) #bbox loss
                totalTestLoss = loss + (loss2*10)

        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps

        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            avgTrainLoss, avgTestLoss))
        
        #En iyi modelin kaydedilmesi
        if(temp>avgTestLoss.cpu().detach().numpy() or flag==1):
            torch.save(unet,MODEL_PATH_BEST)
            print("EN İYİ MODEL KAYDEDİLDİ")
            flag=0
        
        temp=avgTestLoss.cpu().detach().numpy()
        ###
   
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))
    print("ALL LOSS",H)

    ### Eğitim loss değerlerini görselleştirme 
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(PLOT_PATH)
    ###

    #Modelin son halini kaydeder
    torch.save(unet,MODEL_PATH_LAST) 
    print("SON MODEL KAYDEDİLDİ")

#### 1.B, 1.C
def create_dataframe(file,x,y,w,h):
    global count
    global df1
    global r
    path_list.append(mask_dataset+"/"+file)
    arr[count][0]=float(x) # x_min
    arr[count][1]=float(y) # y_min
    arr[count][2]=float(x+w) # x_max
    arr[count][3]=float(y+h) # y_max
     
    count+=1
   
    #pathlerden ve box lardan dataframe olusturur. df1'i kullanarak işlemlerimize devam edicez.

    if(count==r):
        df1=pd.DataFrame(arr, columns = ['x_min','y_min','x_max','y_max']) # bbox etiketi için kullanıcağız
        df2=pd.DataFrame(path_list, columns = ['path'])
        df=pd.concat([df1,df2],axis=1)
        #print(df1.head())

#### 1.A     
def mask_to_bbox(mask_dataset):

    #Max Contour yöntemini kullanarak verilen maskelerin bounding boxeslarını çıkarır ve x_min, x_max, y_min, y_max olarak depolar.
    for file in os.listdir(mask_dataset):
        file=mask_dataset+"/"+file
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) != 0:   
            for c in contours:
                c = max(contours, key = cv2.contourArea)
                x,y,w,h = cv2.boundingRect(c)
            create_dataframe(file,x,y,w,h)

if __name__ == "__main__":
    #args
    ### 1.D
    parser = argparse.ArgumentParser(description='-')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size değeri (varsayılan: 8)')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='num_epochs size değeri (varsayılan: 200)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learnig_rate değeri (varsayılan: 0.001)')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='Test split değeri değeri (varsayılan: 0.15)')
    parser.add_argument('--BASE_OUTPUT', type=str, default="output",
                        help='Outputların kaydedileceği konum(varsayılan: ./output)')
    parser.add_argument('--INPUT_IMAGE_WIDTH', type=int, default=128,
                        help='INPUT_IMAGE_WIDTH(varsayılan: 128)')
    parser.add_argument('--INPUT_IMAGE_HEIGHT', type=int, default=128,
                        help='INPUT_IMAGE_HEIGHT(varsayılan: 128)')
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device",DEVICE)
    BATCH_SIZE,NUM_EPOCHS,TEST_SPLIT,INIT_LR,INPUT_IMAGE_WIDTH,INPUT_IMAGE_HEIGHT,TEST_PATHS,MODEL_PATH_BEST,MODEL_PATH_LAST,PLOT_PATH=args_init(args)
    
    mask_to_bbox(mask_dataset)

    train(BATCH_SIZE,NUM_EPOCHS,TEST_SPLIT,INIT_LR,INPUT_IMAGE_WIDTH,INPUT_IMAGE_HEIGHT,TEST_PATHS,MODEL_PATH_BEST,MODEL_PATH_LAST,PLOT_PATH,DEVICE,df1)
  
