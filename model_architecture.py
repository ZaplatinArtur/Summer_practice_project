import torch.nn.functional as F
import torch.nn as nn

class ModelNatoTanks(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.MaxPool_1 = nn.MaxPool2d(kernel_size=(2,2))
        self.BatchNorn_1 = nn.BatchNorm2d(num_features=16)

        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.MaxPool_2 = nn.MaxPool2d(kernel_size=(2,2))
        self.BatchNorn_2 = nn.BatchNorm2d(num_features=32)

        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.MaxPool_3 = nn.MaxPool2d(kernel_size=(2,2))
        self.BatchNorn_3 = nn.BatchNorm2d(num_features=64)

        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.MaxPool_4 = nn.MaxPool2d(kernel_size=(2,2))
        self.BatchNorn_4 = nn.BatchNorm2d(num_features=128)

        self.conv_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.MaxPool_5 = nn.MaxPool2d(kernel_size=(2,2))
        self.BatchNorn_5 = nn.BatchNorm2d(num_features=256)

        self.conv_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=(1,1), stride=(1,1))
        self.MaxPool_6 = nn.MaxPool2d(kernel_size=(2,2))
        self.BatchNorn_6 = nn.BatchNorm2d(num_features=256)

        self.layer0 = nn.Linear(16384, 8192)  
        self.batch_0_fc = nn.BatchNorm1d(8192)

        self.layer1 = nn.Linear(8192, 4096)  
        self.batch_1_fc = nn.BatchNorm1d(4096)
        
        self.layer2 = nn.Linear(4096, 2048)
        self.batch_2_fc = nn.BatchNorm1d(2048)
        
        self.layer3 = nn.Linear(2048, 1024)
        self.batch_3_fc = nn.BatchNorm1d(1024)
        
        self.layer4 = nn.Linear(1024, 512)
        self.batch_4_fc = nn.BatchNorm1d(512)

        self.layer5 = nn.Linear(512, 256)
        self.batch_5_fc = nn.BatchNorm1d(256)
        
        self.layer6 = nn.Linear(256, 64)
        self.batch_6_fc = nn.BatchNorm1d(64)
        
        self.layer7 = nn.Linear(64, 3)
    
    def forward(self, x):
        # 3*512*512
        x = self.conv_1(x)
        x = self.MaxPool_1(x)
        x = self.BatchNorn_1(x)
        x = F.relu(x)
        # 16*256*256

        x = self.conv_2(x)
        x = self.MaxPool_2(x)
        x = self.BatchNorn_2(x)
        x = F.relu(x)
        # 32*128*128

        x = self.conv_3(x)
        x = self.MaxPool_3(x)
        x = self.BatchNorn_3(x)
        x = F.relu(x)
        # 64*64*64

        x = self.conv_4(x)
        x = self.MaxPool_4(x)
        x = self.BatchNorn_4(x)
        x = F.relu(x)
        # 128*32*32

        x = self.conv_5(x)
        x = self.MaxPool_5(x)
        x = self.BatchNorn_5(x)
        x = F.relu(x)
        # 256*16*16

        x = self.conv_6(x)
        x = self.MaxPool_6(x)
        x = self.BatchNorn_6(x)
        x = F.relu(x)
        # 256*8*8

        x = x.flatten(start_dim=1)  # 256*8*8 = 16384

        x = self.layer0(x) 
        x = self.batch_0_fc(x)
        x = F.relu(x)
        
        x = self.layer1(x)  
        x = self.batch_1_fc(x)
        x = F.relu(x)
        
        x = self.layer2(x)  
        x = self.batch_2_fc(x)
        x = F.relu(x)

        x = self.layer3(x)  
        x = self.batch_3_fc(x)
        x = F.relu(x)

        x = self.layer4(x) 
        x = self.batch_4_fc(x)
        x = F.relu(x)

        x = self.layer5(x) 
        x = self.batch_5_fc(x)
        x = F.relu(x)

        x = self.layer6(x) 
        x = self.batch_6_fc(x)
        x = F.relu(x)
        
        x = self.layer7(x)  

        return x