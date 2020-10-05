import time
import numpy as np
import pickle as pk
from scipy.io import loadmat

import CMINE_lib as CMINE




#-----------------------------------------------------------------#    
#--------------- Create the dataset ------------------------------#
#-----------------------------------------------------------------#    
dim=1
depth=5

data = loadmat('Data/Traffic/Road_5S_3s_3.mat')

n=data['X'].shape[0]
dataset=[col.reshape(n,1) for col in data['X'].T]
dataset=CMINE.prepare_dataset(dataset,Mode='norm')
n=dataset[0].shape[0]

K=2

train_size=n//2
#test_size=n//2
T=10

M=3
s1=[0,1,2]
for i in s1:
    s2=s1[:]
    s2.remove(i)
    for j in s2:
        link=str(i)+str(j)
        print('link=',link)
        s3=s2[:]
        s3.remove(j)        
        arrng=[[i],[j],s3]

                
        start_time = time.time()
        batch_train, target_train, joint_test, prod_test= CMINE.batch_construction(data=dataset, arrange=arrng, set_size=train_size, K_neighbor=K, mode='with_memory',depth=depth)    
        #print('Duration: ',time.time()-start_time,' seconds')
        
        
        
        #----------------------------------------------------------------------#
        #------------------------Train the network-----------------------------#
        #----------------------------------------------------------------------#
        
        # Set up neural network paramters
        LR = 1e-3
        EPOCH=1000
        input_size=M*dim*(depth+1)
        hidden_size = 64
        num_classes = 2
        tau=1e-3
        EPS=1e-6
        
        NN_params=(input_size,hidden_size,num_classes,tau)
        CMI_LDR_t=[]
        CMI_DV_t=[]
        CMI_NWJ_t=[]
        loss_t=[] 
            
        EVAL=False
        if EVAL:
            CMI_LDR_Eval=[]
            CMI_DV_Eval=[]
            CMI_NWJ_Eval=[]
        
        
        for t in range(T):
            print('t: ',t)
            start_time = time.time()
            
        
            #Train
            if EVAL:
                model, loss_e, CMI_LDR_e, CMI_DV_e, CMI_NWJ_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Epsilon=EPS, Eval=True, JointEval=joint_test, ProdEval=prod_test)        
                CMI_LDR_Eval.append(CMI_LDR_e)
                CMI_DV_Eval.append(CMI_DV_e)    
                CMI_NWJ_Eval.append(CMI_NWJ_e)
            else:   
                model,loss_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Epsilon=EPS)
            loss_t.append(loss_e)    
            
            #Compute I(X->Y||Z)
            CMI_est = CMINE.estimate_CMI(model,joint_test,prod_test)
        
            print('Duration: ',time.time()-start_time,' seconds')
            CMI_LDR_t.append(CMI_est[0])
            CMI_DV_t.append(CMI_est[1])
            CMI_NWJ_t.append(CMI_est[2])        
            
            print('LDR=',CMI_LDR_t[-1])   
            print('DV=',CMI_DV_t[-1])   
            print('NWJ=',CMI_NWJ_t[-1]) 
        
            
        
        
        CMI_LDR=np.mean(CMI_LDR_t)
        CMI_DV=np.mean(CMI_DV_t)
        CMI_NWJ=np.mean(CMI_NWJ_t)
        print('DV final:',CMI_DV)
        print('\n\n')
        
        # open a file, where you ant to store the data
        file = open('Data/Traffic/CMI_Traffic_d5_'+str(link), 'wb')
        # dump information to that file
        pk.dump((CMI_LDR_t,CMI_DV_t,CMI_NWJ_t,CMI_LDR_Eval,CMI_DV_Eval,CMI_NWJ_Eval,n,dim,K,LR,EPOCH,loss_t), file)
        
        file.close()    