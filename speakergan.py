import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import sys
import random
import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
import time
import struct
from torchaudio.compliance.kaldi import fbank
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.init import xavier_uniform_


class MyDataset(Dataset):
    # input: librispeech:train-clean-100 vad.
    # output: fbank(1,160,64)
    def __init__(self, mode = 'train', use_redis = True):
        self.mode = mode
        self.use_redis = use_redis
        if self.mode == "train":
            self.datafile = 'data/train.csv'
        else:
            self.datafile = 'data/test.csv'
        self.file_ids = open(self.datafile,'r').readlines()
        self.sample_rate = 16000
        self.width = 160
        self.height = 64
    
    def _toRedis(self,r,arr,key):
       """Store given Numpy array 'arr' in Redis under key """
       h, w = arr.shape
       shape = struct.pack('>II',h,w)
       encoded = shape + arr.tobytes()
    
       # Store encoded data in Redis
       r.set(key,encoded)
       return
    
    def _fromRedis(self,r,key):
       """Retrieve Numpy array from Redis key 'arr'"""
       encoded = r.get(key)
       h, w = struct.unpack('>II',encoded[:8])
       arr = np.frombuffer(encoded, dtype=np.float32, offset=8).reshape(h,w)
       return arr

    def __getitem__(self, idx):
        f,label = self.file_ids[idx].strip().split(" ")
        if self.use_redis : #use redis
            import redis
            r = redis.Redis(host='localhost', port=6379,  db=1)
            if r.exists(f): #try to read from redis
                feature = self._fromRedis(r,f) #read from redis
                feature = torch.from_numpy(feature)
            else: #get from cpu and save to redis.
                wav,sr = torchaudio.load(f,normalize=False)
                assert sr == self.sample_rate
                wav = wav / 1.0
                feature = fbank(wav, dither=1,high_freq=-200, low_freq=64, htk_compat=True,  num_mel_bins=self.height, sample_frequency=self.sample_rate, use_energy=False, window_type='hamming')
                self._toRedis(r,feature.numpy(),f) #set redis
        else: #not ues redis
            wav,sr = torchaudio.load(f,normalize=False)
            assert sr == self.sample_rate
            wav = wav / 1.0
            feature = fbank(wav, dither=1,high_freq=-200, low_freq=64, htk_compat=True,  num_mel_bins=self.height, sample_frequency=self.sample_rate, use_energy=False, window_type='hamming')

        feature_len = len(feature)

        if feature_len < self.width:# for too short utterance.
            for _ in range(self.width // feature_len):
                feature = torch.cat((feature,feature),0)
            feature_len = len(feature)

        if self.mode == "train": #random start pieces
            rand_start = random.randint(0,feature_len - self.width)
            feature = feature[rand_start : rand_start + self.width]
        else: #fixed feature for test
            feature = feature[0 : self.width]
            #rand_start = random.randint(0,feature_len - self.width)
            #feature = feature[rand_start : rand_start + self.width]

        #normalize
        std,mu = torch.std_mean(feature,dim=0)
        feature = (feature - mu) / (std + 1e-5)

        feature = torch.unsqueeze(feature, dim=0)
        label = torch.LongTensor([int(label)])

        return feature,label 

    def __len__(self):
        return len(self.file_ids)
    
class Generator(nn.Module):
    #input x: h,c
    #output G(x): h,c
    def __init__(self):
        super(Generator,self).__init__()
        self.conv1 = nn.Conv2d(1,256, (15,1), stride = (1,1),padding='same')
        self.gate1 = nn.Sequential(nn.Conv2d(1,256, (15,1), stride = (1,1),padding='same'), nn.Sigmoid())
        self.conv2 = nn.Conv2d(256,512, (5,1), stride = (2,1), padding=(2,0))
        self.gate2 = nn.Sequential(nn.Conv2d(256,512, (5,1), stride = (2,1), padding=(2,0)),nn.Sigmoid())
        self.conv3 = nn.Conv2d(512,1024, (5,1), stride = (2,1), padding=(2,0))
        self.gate3 = nn.Sequential(nn.Conv2d(512,1024, (5,1), stride = (2,1), padding=(2,0)),nn.Sigmoid())
        self.conv4 = nn.Conv2d(1024,1024, (5,1), stride = (1,1),padding='same')
        self.conv5 = nn.Conv2d(512,512, (5,1), stride = (1,1),padding='same')
        self.gate5 = nn.Sequential(nn.Conv2d(256,256, (5,1), stride = (2,1)),nn.Sigmoid())
        self.conv6 = nn.Conv2d(256,1, (15,1), stride = (1,1),padding='same')
        self._initialize_weights()

    def __pixel_shuffle(self,input, upscale_factor_h, upscale_factor_w):
        batch_size, channels, in_height, in_width = input.size()
        channels //= upscale_factor_h * upscale_factor_w

        out_height = in_height * upscale_factor_h
        out_width = in_width * upscale_factor_w

        input_view = input.contiguous().view(batch_size, channels, upscale_factor_h, upscale_factor_w,in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
        return shuffle_out.view(batch_size, channels, out_height, out_width)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def _instance_norm(self,inputs): #never use.
        return nn.InstanceNorm2d(inputs.shape[1])(inputs)
                
    def forward(self,x):
        # gate = A * B
        A_1 = self.conv1(x)
        B_1 = self.gate1(x)
        x = A_1 * B_1

        #downsample 1
        A_2 = self.conv2(x)
        B_2 = self.gate2(x)
        x = A_2 * B_2

        #downsample 2
        A_3 = self.conv3(x)
        B_3 = self.gate3(x)
        x = A_3 * B_3

        #upsample 1
        x = self.conv4(x)
        A_4 = self.__pixel_shuffle(x,2,1)
        B_4 = nn.Sigmoid()(self.__pixel_shuffle(x,2,1))
        x = A_4 * B_4

        #upsample 2
        x = self.conv5(x)
        A_5 = self.__pixel_shuffle(x,2,1)
        B_5 = nn.Sigmoid()(self.__pixel_shuffle(x,2,1))
        x = A_5 * B_5

        x = self.conv6(x)
     
        return x

class RestBlock1(nn.Module):
    #RestBlock1 architecture:
    def __init__(self):
        super(RestBlock1, self).__init__()
        self.conv1 = nn.Conv2d(64,32, (1,1), stride = (1,1), padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32,32, (3,3), stride = (1,1), padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32,64, (1,1), stride = (1,1), padding='same')
        self.bn3 = nn.BatchNorm2d(64)
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        #add x into out
        out = self.relu(torch.add(x, out))
        return out

class RestBlock2(nn.Module):
    #RestBlock2 architecture:
    def __init__(self):
        super(RestBlock2, self).__init__()
        self.conv1 = nn.Conv2d(128,64, (1,1), stride = (1,1), padding='same')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64,64, (3,3), stride = (1,1), padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64,128, (1,1), stride = (1,1), padding='same')
        self.bn3 = nn.BatchNorm2d(128)
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        #add x into out 
        out = self.relu(torch.add(x, out))
        return out

class RestBlock3(nn.Module):
    #RestBlock3 architecture:
    def __init__(self):
        super(RestBlock3, self).__init__()
        self.conv1 = nn.Conv2d(256,128, (1,1), stride = (1,1), padding='same')
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128,128, (3,3), stride = (1,1), padding='same')
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128,256, (1,1), stride = (1,1), padding='same')
        self.bn3 = nn.BatchNorm2d(256)
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        #add x into out
        out = self.relu(torch.add(x, out))
        return out

class RestBlock4(nn.Module):
    #RestBlock4 architecture:
    def __init__(self):
        super(RestBlock4, self).__init__()
        self.conv1 = nn.Conv2d(512,256, (1,1), stride = (1,1), padding='same')
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256,256, (3,3), stride = (1,1), padding='same')
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256,512, (1,1), stride = (1,1), padding='same')
        self.bn3 = nn.BatchNorm2d(512)
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        #add x into out
        out = self.relu(torch.add(x, out))
        return out

class Discriminator(nn.Module):
    #input x,G(x)
    #output 251 + 2
    def __init__(self, num_classes):
        super(Discriminator,self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1,64, (5,5), stride = (2,2), padding=2)
        self.rest1 = nn.Sequential(RestBlock1(), RestBlock1(), RestBlock1())

        self.conv2 = nn.Conv2d(64,128, (5,5), stride = (2,2), padding=2)
        self.rest2 = nn.Sequential(RestBlock2(), RestBlock2(), RestBlock2())

        self.conv3 = nn.Conv2d(128,256, (5,5), stride = (2,2), padding=2)
        self.rest3 = nn.Sequential(RestBlock3(), RestBlock3(), RestBlock3())

        self.conv4 = nn.Conv2d(256,512, (5,5), stride = (2,2), padding=2)
        self.rest4 = nn.Sequential(RestBlock4(), RestBlock4(), RestBlock4())

        self.avgpool = nn.AvgPool1d(10)
        self.fc = nn.Linear(2048,512)

        self.fc_nc = nn.Linear(512,self.num_classes)
        self.fc_rf = nn.Linear(512,2)

        self.softmax_nc = nn.Softmax()
        self.softmax_rf = nn.Softmax(dim=-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                xavier_uniform_(m.weight)
                m.bias.data.fill_(0)

    def forward(self,x):
        #conv1 and rest1 (out dim: 80,32,64)
        out = self.conv1(x)
        out = self.rest1(out)

        #conv2 and rest2 (out dim: 40,16,128)
        out = self.conv2(out)
        out = self.rest2(out)

        #conv3 and rest3 (out dim: 20,8,256)
        out = self.conv3(out)
        out = self.rest3(out)

        #conv4 and rest4 (out dim: 10,4,512)
        out = self.conv4(out)
        out = self.rest4(out)

        #Reshape(10,2048)
        out = out.reshape(-1,2048,10)
        #averge temporal pooling layer
        out = self.avgpool(out)
        #remove third dim
        out = torch.squeeze(out,-1)
        #fully connected (out dim:512)
        out = self.fc(out)

        #fully connected layer output N class (out dim:251)
        out_nc = self.fc_nc(out)
        #fully connected layer output real or fake (out dim:2)
        out_rf = self.fc_rf(out)

        #softmax layer output N class
        #out_nc = self.softmax_nc(out_nc) #CrossEntropy Loss has employed logsoftmax

        #softmax layer output real or fake
        out_rf = self.softmax_rf(out_rf) #use MSE and label smoothing

        return out_nc, out_rf    

        

class SpeakerGAN():
    def __init__(self,):
        #base info
        self.num_classes = 251
        self.batch_size = 64
        self.lr_init10 = 5e-4 
        self.lr_after10 = 2e-4
        self.epochs = 50
        self.device = 'cuda:0' #'cuda:0' or 'cpu
        self.d_train_times = 4
        self.model_path = './model'

        #network
        self.D = Discriminator(self.num_classes).to(self.device)
        self.G = Generator().to(self.device)

        #loss
        self.HuberLoss = nn.HuberLoss().to(self.device) #the HuberLoss formula in pytorch is a little different with paper
        self.Class_CrossEntropyLoss = nn.CrossEntropyLoss().to(self.device) #cross entropy loss for speaker id classification
        self.AdversarialLoss = nn.MSELoss().to(self.device) #from LSGAN

        #load data
        self.train_dataset = MyDataset(mode='train')
        self.train_dataloader = DataLoader(self.train_dataset,batch_size = self.batch_size, shuffle=True, drop_last=True, num_workers=5 )
        self.test_dataset = MyDataset(mode='test')
        self.test_dataloader = DataLoader(self.test_dataset,batch_size = 1, shuffle=False, drop_last=False, num_workers=2 )

        #optimizer
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr = self.lr_init10, betas=(0.5, 0.999) )
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr = self.lr_init10, betas=(0.5, 0.999) )

        #adjust lr
        self.lr_scheduler_D = MultiStepLR(self.optimizer_D, milestones=[10,], gamma = 0.4)
        self.lr_scheduler_G = MultiStepLR(self.optimizer_G, milestones=[10,], gamma = 0.4)
        self.writer = SummaryWriter(log_dir='log')

    def save(self,epoch):
        torch.save(self.D,os.path.join(self.model_path,str(epoch)+"_"+"D.pkl"))
        torch.save(self.G,os.path.join(self.model_path,str(epoch)+"_"+"G.pkl"))

    def train(self):
        idx = 0
        for epoch in range(self.epochs):
            id_in_epoch = 0
            for batch_id,(x,y) in enumerate(self.train_dataloader):

                idx = idx + 1
                id_in_epoch = id_in_epoch + 1

                self.writer.add_scalar('lr_D',self.optimizer_D.param_groups[0]["lr"] ,idx)
                self.writer.add_scalar('lr_G',self.optimizer_G.param_groups[0]["lr"] ,idx)
                
                #load data to device
                x = x.to(self.device)
                y = y.to(self.device)

                y = torch.squeeze(y)

                print ('id_in_epoch/total: ' + str(id_in_epoch) + '/'+str(self.train_dataset.__len__() // self.batch_size) + ' epoch:' + str(epoch))
                
                smooth_label_fake = torch.rand(self.batch_size, 2).to(self.device).squeeze()*0.3 + 0.7
                smooth_label_real = torch.rand(self.batch_size, 2).to(self.device).squeeze()*0.3 + 0.7
                smooth_label_fake.T[1] = 1 - smooth_label_fake.T[0]
                smooth_label_real.T[0] = 1 - smooth_label_real.T[1]
                label_fake_smooth = smooth_label_fake #smooth for fake_x 0, label: [>0.7, <0.3]
                label_real_smooth = smooth_label_real #smooth for x 1, label: [<0.3, >0.7]

                #update D
                self.G.eval()
                self.D.train()

                self.optimizer_D.zero_grad()

                pred_real_y, pred_real_flag = self.D(x)
                real_loss_d = self.AdversarialLoss(pred_real_flag,label_real_smooth)

                fake_x = self.G(x)
                pred_fake_y, pred_fake_flag = self.D(fake_x) #this line is important.
                fake_loss_d = self.AdversarialLoss(pred_fake_flag,label_fake_smooth) #this line is important.

                adv_loss_d = real_loss_d + fake_loss_d
                
                class_acc_real = torch.eq(torch.argmax(pred_real_y, dim = 1), y).sum().float().item() / len(y)
                class_acc_fake = torch.eq(torch.argmax(pred_fake_y, dim = 1), y).sum().float().item() / len(y)

                self.writer.add_scalar('acc/class_acc_real',class_acc_real , idx)
                self.writer.add_scalar('acc/class_acc_fake',class_acc_fake , idx)

                classification_loss_real = self.Class_CrossEntropyLoss(pred_real_y,y) 
                classification_loss_fake = self.Class_CrossEntropyLoss(pred_fake_y,y) 
                classification_loss = classification_loss_real + classification_loss_fake
                loss_d = adv_loss_d + classification_loss

                loss_d.backward()
                self.optimizer_D.step()

                self.writer.add_scalar('loss_d/adv_loss_d', adv_loss_d.item(), idx)
                self.writer.add_scalar('loss_d/class_loss', classification_loss.item(),idx)
                self.writer.add_scalar('loss_d', loss_d.item(),idx)
                
                
                if idx % self.d_train_times == 0:
                    self.G.train()
                    #update G
                    self.optimizer_G.zero_grad()
                    fake_x = self.G(x)
                    pred_fake_y, pred_fake_flag = self.D(fake_x)

                    huber_loss = self.HuberLoss(x,fake_x)
                    adv_loss_g = self.AdversarialLoss(pred_fake_flag, label_real_smooth)

                    loss_g = adv_loss_g + huber_loss

                    loss_g.backward()
                    self.optimizer_G.step()

                    self.writer.add_scalar('loss_g/adv_loss_g', adv_loss_g.item(), idx)
                    self.writer.add_scalar('loss_g/huber_loss', huber_loss.item(),idx)
                    self.writer.add_scalar('loss_g', loss_g.item(),idx)

            #save model
            self.save(epoch)

            #adjust lr
            self.lr_scheduler_D.step()
            self.lr_scheduler_G.step()
                
        self.writer.close()        
            
    def test(self,D_model):
        d = torch.load(D_model)
        d.eval()
        correct = 0.0
        for batch_id,(x,y) in enumerate(self.test_dataloader):
            if batch_id % 5000 == 0:
                print ('testing : ' + str(batch_id) + '/' + str(self.test_dataset.__len__()))
            x = x.to(self.device)
            y = y.to(self.device)
            pred_y, _ = d(x)
            test_acc = torch.eq(torch.argmax(pred_y, dim = 1), y).sum().float().item() / len(y)
            correct = correct + test_acc
        print ('test accuracy:')
        print (correct / self.test_dataset.__len__())

    def generate_sample(self, G_model):
        g = torch.load(G_model)
        g.eval()
        g.to('cpu')

        gen_file = "gen.png"

        fig = plt.figure()

        totalimg = 20
        for batch_id,(x,y) in enumerate(self.test_dataloader):
            fake_x = g(x)            

            x = torch.squeeze(x)
            fake_x = torch.squeeze(fake_x).detach().numpy()
             
            if batch_id > (totalimg -1 ):
                break
            posid = batch_id * 2  + 1
            ax = fig.add_subplot(totalimg,2,posid)
            ax.imshow(x.T,cmap='plasma')
            ax = fig.add_subplot(totalimg,2,posid + 1)
            ax.imshow(fake_x.T,cmap='plasma')

        plt.savefig(gen_file)
        print ("save genrated samples to: " + gen_file)
            
if __name__ == "__main__":
    model = SpeakerGAN()
    model.train()
    model.test('model/49_D.pkl')
    model.generate_sample('model/49_G.pkl')
