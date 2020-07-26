from __future__ import print_function
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.model_selection import cross_val_score


def main():
    data_0=np.empty((50,480,640,3))
    data_1=np.empty((43,480,640,3))
    data_2=np.zeros((50,480,640,3))
    b_0=np.empty((3,480,640,3))
    b_1=np.empty((2,480,640,3))
    b_2=np.empty((2,480,640,3))
    #Loading Genreal Waste images as numpy array, class labeled 0
    #all samples have been scaled between 0 and 1   

    for i in range(data_0.shape[0]):
        image=Image.open('bin_bag_test/general_waste/{}.jpg'.format(i))
        image=np.asarray(image)
        data_0[i]=image
        data_0[i]=((data_0[i]/data_0[i].max()))
        
    #Loading Green Sack images as numpy array, class labeled 1
    #all samples have been scaled between 0 and 1 
    for i in range(data_1.shape[0]):
        image=Image.open('bin_bag_test/green_sack/{}.jpg'.format(i))
        image=np.asarray(image)
        data_1[i]=image
        data_1[i]=(data_1[i]/data_1[i].max())
        
    #Loading Mixed Recycling images as numpy array, class labeled 2
    #all samples have been scaled between 0 and 1 
    for i in range(data_2.shape[0]):
        image=Image.open('bin_bag_test/mixed_recycling/{}.jpg'.format(i))
        image=np.asarray(image)
        data_2[i]=image
        data_2[i]=(data_2[i]/data_2[i].max())
    
    #Loading the 7 background image given in 3 folders, 
    for i in range(b_0.shape[0]):
        image=Image.open('bin_bag_test/general_waste/b0{}.jpg'.format(i+1))
        image=np.asarray(image)
        b_0[i]=image
        b_0[i]=(b_0[i]/b_0[i].max())
    for i in range(b_1.shape[0]):
        image=Image.open('bin_bag_test/green_sack/b1{}.jpg'.format(i+1))
        image=np.asarray(image)
        b_1[i]=image
        b_1[i]=(b_1[i]/b_1[i].max())
    for i in range(b_2.shape[0]):
        image=Image.open('bin_bag_test/mixed_recycling/b2{}.jpg'.format(i+1))
        image=np.asarray(image)
        b_2[i]=image
        b_2[i]=(b_2[i]/b_2[i].max())
    #Back contains the 7 background images under different lighting conditions 
    back=np.concatenate((b_0,b_1,b_2))     
    #mean_back is the mean background image created from the 7 background 
    mean_back=np.mean(back, axis=0)
    # Uncomment line 59 to 76 for visualising samples
    #Print two samples of each class and the mean background class 
    for i in range(2):
        plt.figure(figsize=(3,3))
        plt.suptitle("data_0 general waste bag, image number {}".format(i))
        plt.imshow(data_0[i])
        plt.show()
    
    for i in range(2):
        plt.figure(figsize=(3,3))
        plt.suptitle("data_1 green sack bag, image number {}".format(i))
        plt.imshow(data_1[i])
        plt.show()
        
    for i in range(2):
        plt.figure(figsize=(3,3))
        plt.suptitle("data_2 mixed recycling bag, image number {}".format(i))
        plt.imshow(data_2[i])
        plt.show()
        
    plt.figure(figsize=(3,3))
    plt.suptitle("mean background class")
    plt.imshow(mean_back)
    plt.show()
    
    
            
    #Removing the background 
    #Sunbtract the background from the Images
    #Scaling data samples between 0 and 1 again
    
    for i in range(data_0.shape[0]):
        data_0[i]=data_0[i]-mean_back
    data_0=(data_0-data_0.min())/(data_0.max()-data_0.min())  
        
    for i in range(data_1.shape[0]):
        data_1[i]=data_1[i]-mean_back
    data_1=(data_1-data_1.min())/(data_1.max()-data_1.min())  
        
    for i in range(data_2.shape[0]):
        data_2[i]=data_2[i]-mean_back
    data_2=(data_2-data_2.min())/(data_2.max()-data_2.min())  
    
    #Visualise data samples after removing background
    #for i in range(2):
    #    plt.figure(figsize=(3,3))
    #    plt.suptitle("data_0 general waste bag, image number {}".format(i))
    #    plt.imshow(data_0[i])
    #    plt.show()
    #
    #for i in range(2):
    #    plt.figure(figsize=(3,3))
    #    plt.suptitle("data_1 green sack bag, image number {}".format(i))
    #    plt.imshow(data_1[i])
    #    plt.show()
    #    
    #for i in range(2):
    #    plt.figure(figsize=(3,3))
    #    plt.suptitle("data_2 mixed recycling bag, image number {}".format(i))
    #    plt.imshow(data_2[i])
    #    plt.show()
    #Now, Create a mask that Separates background from the image
    threshold=0.6    
    for i in range(data_0.shape[0]):
        for j in range(3):
            mask=data_0[i,:,:,j]
            mask=1*(mask>threshold) #here RGB are given same threshold values but can be made different if needed
            data_0[i,:,:,j]=data_0[i,:,:,j]*mask
            
    for i in range(data_1.shape[0]):
        for j in range(3):
            mask=data_1[i,:,:,j]
            mask=1*(mask>threshold) #here RGB are given same threshold values but can be made different if needed
            data_1[i,:,:,j]=data_1[i,:,:,j]*mask
        
    for i in range(data_2.shape[0]):
        for j in range(3):
            mask=data_2[i,:,:,j]
            mask=1*(mask>threshold) #here RGB are given same threshold values but can be made different if needed
            data_2[i,:,:,j]=data_2[i,:,:,j]*mask
            
    for j in range(3):
        mask=mean_back[:,:,j]
        mask=1*(mask>threshold) #here RGB are given same threshold values but can be made different if needed
        mean_back[:,:,j]=mean_back[:,:,j]*mask
        
    #Binarizing images for better data       
    threshold_binary_red=0.5
    threshold_binary_green=0.3
    threshold_binary_blue=0.5
    for i in range(data_0.shape[0]):
        data_0[i,:,:,0]=1*(data_0[i,:,:,0]>threshold_binary_red)
        data_0[i,:,:,1]=1*(data_0[i,:,:,1]>threshold_binary_green)
        data_0[i,:,:,2]=1*(data_0[i,:,:,2]>threshold_binary_blue)
    for i in range(data_1.shape[0]):
        data_1[i,:,:,0]=1*(data_1[i,:,:,0]>threshold_binary_red)
        data_1[i,:,:,1]=1*(data_1[i,:,:,1]>threshold_binary_green)
        data_1[i,:,:,2]=1*(data_1[i,:,:,2]>threshold_binary_blue)
    for i in range(data_2.shape[0]):
        data_2[i,:,:,0]=1*(data_2[i,:,:,0]>threshold_binary_red)
        data_2[i,:,:,1]=1*(data_2[i,:,:,1]>threshold_binary_green)
        data_2[i,:,:,2]=1*(data_2[i,:,:,2]>threshold_binary_blue)
    
#    mean_back[:,:,0]=1*(mean_back[:,:,0]>threshold_binary_red)
#    mean_back[:,:,1]=1*(mean_back[:,:,1]>threshold_binary_green)
#    mean_back[:,:,2]=1*(mean_back[:,:,2]>threshold_binary_blue)
    
    for i in range(2):
        plt.figure(figsize=(3,3))
        plt.suptitle("data_0 general waste bag, image number {}".format(i))
        plt.imshow(data_0[i])
        plt.show()
    
    for i in range(4):
        plt.figure(figsize=(3,3))
        plt.suptitle("data_1 green sack bag, image number {}".format(i))
        plt.imshow(data_1[i])
        plt.show()
        
    for i in range(2):
        plt.figure(figsize=(3,3))
        plt.suptitle("data_2 mixed recycling bag, image number {}".format(i))
        plt.imshow(data_2[i])
        plt.show()        
    
    
    data_0_feat=np.zeros((data_0.shape[0],3))
    data_0_feat[:,0]=np.sum(data_0[:,:,:,0],axis=(1,2))
    data_0_feat[:,1]=np.sum(data_0[:,:,:,1],axis=(1,2))
    data_0_feat[:,2]=np.sum(data_0[:,:,:,2],axis=(1,2))
    
    data_1_feat=np.zeros((data_1.shape[0],3))
    data_1_feat[:,0]=np.sum(data_1[:,:,:,0],axis=(1,2))
    data_1_feat[:,1]=np.sum(data_1[:,:,:,1],axis=(1,2))
    data_1_feat[:,2]=np.sum(data_1[:,:,:,2],axis=(1,2))
    
    data_2_feat=np.zeros((data_2.shape[0],3))
    data_2_feat[:,0]=np.sum(data_2[:,:,:,0],axis=(1,2))    
    data_2_feat[:,1]=np.sum(data_2[:,:,:,1],axis=(1,2))
    data_2_feat[:,2]=np.sum(data_2[:,:,:,2],axis=(1,2))
    
    # mean_back_feat=np.zeros((3))
    # mean_back_feat[0]=np.sum(mean_back[:,:,0],axis=(0,1))  
    # mean_back_feat[1]=np.sum(mean_back[:,:,1],axis=(0,1))
    # mean_back_feat[2]=np.sum(mean_back[:,:,2],axis=(0,1))
    
    
    data0label=np.zeros((data_0.shape[0]))
    data1label=np.zeros((data_1.shape[0]))+1
    data2label=np.zeros((data_2.shape[0]))+2
    
    X=np.concatenate((data_0_feat,data_1_feat,data_2_feat))
    y=np.concatenate((data0label,data1label,data2label))
    #Shuffle data samples 
    c = list(zip(X, y))
    random.shuffle(c)
    X, y = zip(*c)
    knn_cv = KNeighborsClassifier(n_neighbors=2)
    cv_scores = cross_val_score(knn_cv, X, y, cv=5) 
    print("After 5 fold cross validation, the mean score is")
    print(np.mean(cv_scores))

        
if __name__ == '__main__':
    main()

