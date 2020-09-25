import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

# MLP


import CMINE_lib as CMINE



#-----------------------------------------------------------------#    
#--------------- Create the dataset ------------------------------#
#-----------------------------------------------------------------#    
dim=5
n=int(8e4)

sigma_x=10
sigma_1=1
sigma_2=5
arrng=[[0],[1],[2]]


K=2


params=(sigma_x,sigma_1,sigma_2)

True_CMI=-dim*0.5*np.log(sigma_1**2 * (sigma_x**2+sigma_1**2 + sigma_2**2)/((sigma_x**2 + sigma_1**2)*(sigma_1**2 + sigma_2**2)))





#----------------------------------------------------------------------#
#------------------------Train the network-----------------------------#
#----------------------------------------------------------------------#

# Set up neural network paramters
LR = 2e-3
EPOCH=300
input_size=3*dim
hidden_size = 64
num_classes = 2
tau=0.0001

NN_params=(input_size,hidden_size,num_classes,tau)
EVAL=False

#Monte Carlo param
T=10
S=1

CMI_LDR=[]
CMI_DV=[]
CMI_NWJ=[]
loss=[]


for s in range(S):
    CMI_LDR_t=[]
    CMI_DV_t=[]
    CMI_NWJ_t=[]
    loss_t=[]
        
    #Create dataset
    dataset=CMINE.create_dataset(GenModel='Gaussian_nonZero',Params=params, Dim=dim, N=n)
    
    
    
    
    
    for t in range(T): 
        
        
        start_time = time.time()
        
        batch_train, target_train, joint_test, prod_test=CMINE.batch_construction(data=dataset, arrange=arrng, set_size=n//2, K_neighbor=K)    
        print('Duration of data preparation: ',time.time()-start_time,' seconds')

        
        if EVAL:
            CMI_LDR_Eval=[]
            CMI_DV_Eval=[]
            CMI_NWJ_Eval=[]
        
    
        start_time = time.time()
        
    
        #Train
        if EVAL:
            model, loss_e, CMI_LDR_e, CMI_DV_e, CMI_NWJ_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Eval=True, JointEval=joint_test, ProdEval=prod_test)        
            CMI_LDR_Eval.append(CMI_LDR_e)
            CMI_DV_Eval.append(CMI_DV_e)    
            CMI_NWJ_Eval.append(CMI_NWJ_e)
        else:   
            model, loss_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR)
        
        #Compute I(X;Y|Z)
        CMI_est = CMINE.estimate_CMI(model, joint_test, prod_test)
        print(CMI_est)
    
        print('Duration: ', time.time()-start_time, ' seconds')       
        
        print('LDR=',CMI_est[0])   
        print('DV=',CMI_est[1])   
        print('NWJ=',CMI_est[2]) 
        print('True=',True_CMI)
        
        CMI_LDR_t.append(CMI_est[0])
        CMI_DV_t.append(CMI_est[1])
        CMI_NWJ_t.append(CMI_est[2])
        
    CMI_LDR.append(np.mean(CMI_LDR_t))
    CMI_DV.append(np.mean(CMI_DV_t))
    CMI_NWJ.append(np.mean(CMI_NWJ_t))    
    
# open a file, where you ant to store the data
file = open('Data/CMI_k2_8e4', 'wb')
# dump information to that file
pickle.dump((True_CMI,CMI_LDR,CMI_DV,CMI_NWJ,CMI_LDR_Eval,CMI_DV_Eval,CMI_NWJ_Eval,n,dim,K,LR,EPOCH,loss_e), file)

file.close()    