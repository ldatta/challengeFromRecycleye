from __future__ import print_function
import argparse
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Netconv(nn.Module):
    def __init__(self):
        super(Netconv, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 5, 1 )
        self.conv2 = nn.Conv2d(4, 8, 5, 1 )
        self.conv3 = nn.Conv2d(8, 16, 5, 1 )
        self.conv4 = nn.Conv2d(16, 32, 5, 1)
        self.conv5 = nn.Conv2d(32, 64, 5, 1 )
        self.conv6 = nn.Conv2d(64, 3, 3, 1 )
        self.GAP=nn.AvgPool2d((2,2), stride=1, padding=0)
        
    def forward(self, x):
        
        x=x.float()
        x=self.conv1(x) 
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x=self.conv2(x) 
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x=self.conv3(x) 
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x=self.conv4(x) 
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x=self.conv5(x) 
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x=self.conv6(x) 
        x = F.relu(x)
        x = self.GAP(x)
        x = x.view(-1, 3) 
        x=F.log_softmax(x, dim=1)
        return x    
    
def train(args, model, device, train_loader, optimizer, epoch):
    running_loss = 0
    total_train = 0
    correct_train = 0
    model.train() 
    
    for batch_idx, (data, target) in enumerate(train_loader):
        model.train()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total_train += target.size(0)
        correct_train += predicted.eq(target.data).sum().item()
        train_accuracy = 100 * correct_train / total_train
        
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \taccuracy: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),train_accuracy))
    return train_accuracy

def test(args, model, device, test_loader):
    model.train(mode=False)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() 
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc=int(100. * correct / len(test_loader.dataset))
    test_loss=float(test_loss)
    return acc


def main():    
    parser = argparse.ArgumentParser(description='Recyleeye challenge')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N')
    parser.add_argument('--epochs', type=int, default=50, metavar='N')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}    
    data_0=np.zeros((50,256,256,3))
    data_1=np.zeros((43,256,256,3))
    data_2=np.zeros((50,256,256,3))
    
    for i in range(data_0.shape[0]):
        image=Image.open('bin_bag_test/general_waste/{}.jpg'.format(i))
        image=np.asarray(image)
        data_0[i]=cv2.resize(image, (256, 256))
        data_0[i]=data_0[i]/data_0[i].max()
        
    for i in range(data_1.shape[0]):
        image=Image.open('bin_bag_test/green_sack/{}.jpg'.format(i))
        image=np.asarray(image)
        data_1[i]=cv2.resize(image, (256, 256))
        data_1[i]=data_1[i]/data_1[i].max()
    
    for i in range(data_2.shape[0]):
        image=Image.open('bin_bag_test/mixed_recycling/{}.jpg'.format(i))
        image=np.asarray(image)
        data_2[i]=cv2.resize(image, (256, 256))
        data_2[i]=data_2[i]/data_2[i].max()
    
    for i in range(2):
        plt.imshow(data_0[i])
        plt.show()
        
        
    for i in range(2):
        plt.imshow(data_1[i])
        plt.show()
        
        
    for i in range(2):
        plt.imshow(data_2[i])
        plt.show()
    #Normalizing data
    data_0=(data_0-np.mean(data_0))/np.std(data_0)   
    data_1=(data_1-np.mean(data_1))/np.std(data_1)   
    data_2=(data_2-np.mean(data_2))/np.std(data_2) 
    
    data_0_label=np.zeros((data_0.shape[0]))
    data_1_label=np.zeros((data_1.shape[0]))+1
    data_2_label=np.zeros((data_2.shape[0]))+2
    
    
    
    data_train=np.concatenate([data_0[0:45],data_1[0:38],data_2[0:45]])
    data_train_label=np.concatenate([data_0_label[0:45],data_1_label[0:38],data_2_label[0:45]])
    
    data_test=np.concatenate([data_0[45:50],data_1[38:43],data_2[45:50]])
    data_test_label=np.concatenate([data_0_label[38:43],data_1_label[38:43],data_2_label[45:50]])
    
    data_train=np.transpose(data_train, (0,3, 1, 2))
    data_test=np.transpose(data_test, (0,3, 1, 2))
    

    
    traindata=torch.from_numpy(data_train)
    traintarget=torch.from_numpy(data_train_label)
    traintarget = torch.tensor(traintarget, dtype=torch.long)
    my_train_dataset = utils.TensorDataset(traindata,traintarget)
    
    testdata=torch.from_numpy(data_test)
    testtarget=torch.from_numpy(data_test_label)
    testtarget = torch.tensor(testtarget, dtype=torch.long)
    my_test_dataset = utils.TensorDataset(testdata,testtarget)

    torch.manual_seed(1) 
    train_loader = torch.utils.data.DataLoader(my_train_dataset,batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(my_test_dataset,batch_size=args.test_batch_size, shuffle=True, **kwargs)
    model = Netconv().to(device)    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)    
    testacc=(np.arange(1,(args.epochs+1),1 ))
    trnacc=(np.arange(1,(args.epochs+1),1 ))    
    for epoch in range(1, args.epochs + 1):    
        trnacc[epoch-1]=train(args, model, device, train_loader, optimizer, epoch)
        testacc[epoch-1]=test(args, model, device, test_loader)
    print("Train Accuracy",repr(trnacc))
    print("Test Accuracy",repr(testacc))
        
if __name__ == '__main__':
    main()
    


