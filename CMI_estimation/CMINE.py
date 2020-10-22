import time
import numpy as np
import pickle
from numpy.linalg import det

import CMINE_lib as CMINE



def estimate_CMI(config):
    #-----------------------------------------------------------------#    
    #--------------- Create the dataset ------------------------------#
    #-----------------------------------------------------------------#    
    dim = config.d
    n = config.n
    
    sigma_x = config.sigma_x
    sigma_1 = config.sigma_y
    sigma_2 = config.sigma_z
    arrng = config.arrng
    
    params=(sigma_x,sigma_1,sigma_2)
    
    if config.scenario==0: #Estimate I(X;Y|Z)
        True_CMI=-dim*0.5*np.log(sigma_1**2 * (sigma_x**2+sigma_1**2 + sigma_2**2)/((sigma_x**2 + sigma_1**2)*(sigma_1**2 + sigma_2**2)))
    elif config.scenario==1: #Estimate I(X;Z|Y)    
        True_CMI=0
    
    K=config.k
    
    #----------------------------------------------------------------------#
    #------------------------Train the network-----------------------------#
    #----------------------------------------------------------------------#
    
    # Set up neural network paramters
    LR = config.lr
    EPOCH = config.e
    SEED = config.seed
    input_size = 3*dim
    hidden_size = 64
    num_classes = 2
    tau = config.tau
    
    NN_params = (input_size,hidden_size,num_classes,tau)
    EVAL=False
    
    #Monte Carlo param
    T=config.t
    S=config.s
    
    CMI_LDR=[]
    CMI_DV=[]
    CMI_NWJ=[]
    
    for s in range(S):
        CMI_LDR_t=[]
        CMI_DV_t=[]
        CMI_NWJ_t=[]
            
        #Create dataset
        dataset=CMINE.create_dataset(GenModel='Gaussian_nonZero',Params=params, Dim=dim, N=n)

        for t in range(T): 
            start_time = time.time()
            
            batch_train, target_train, joint_test, prod_test=CMINE.batch_construction(data=dataset, arrange=arrng, set_size=n//2, K_neighbor=K)    
            print('Duration of data preparation: ',time.time()-start_time,' seconds')
            
            CMI_LDR_Eval=[]
            CMI_DV_Eval=[]
            CMI_NWJ_Eval=[]

            start_time = time.time()
            #Train
            if EVAL:
                model, loss_e, CMI_LDR_e, CMI_DV_e, CMI_NWJ_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED, Eval=True, JointEval=joint_test, ProdEval=prod_test)        
                CMI_LDR_Eval.append(CMI_LDR_e)
                CMI_DV_Eval.append(CMI_DV_e)    
                CMI_NWJ_Eval.append(CMI_NWJ_e)
            else:   
                model, loss_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED)
            
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
    file = open(config.directory+'/result_'+str(config.seed), 'wb')
    # dump information to that file
    pickle.dump((True_CMI,CMI_LDR,CMI_DV,CMI_NWJ,CMI_LDR_Eval,CMI_DV_Eval,CMI_NWJ_Eval,n,dim,K,LR,EPOCH,loss_e), file)
    
    file.close()    
    


def estimate_CMI_DPI(config):
    #-----------------------------------------------------------------#    
    #--------------- Create the dataset ------------------------------#
    #-----------------------------------------------------------------#    
    dim = config.d
    dim_split=config.dim_split
    n = config.n
    

    sigma_x = config.sigma_x
    sigma_1 = config.sigma_y
    sigma_2 = config.sigma_z
    
    q = config.q
    
    Sigma_x=np.asarray([[sigma_x**2, q*sigma_x**2, 0, 0, 0],
                        [q*sigma_x**2, sigma_x**2, q*sigma_x**2, 0, 0],
                        [0, q*sigma_x**2, sigma_x**2, q*sigma_x**2,0],
                        [0, 0, q*sigma_x**2, sigma_x**2, q*sigma_x**2],
                        [0, 0, 0, q*sigma_x**2, sigma_x**2]])
    
    Sigma_1=np.asarray([[sigma_1**2, q*sigma_1**2, 0, 0, 0],
                        [q*sigma_1**2, sigma_1**2, q*sigma_1**2, 0, 0],
                        [0, q*sigma_1**2, sigma_1**2, q*sigma_1**2,0],
                        [0, 0, q*sigma_1**2, sigma_1**2, q*sigma_1**2],
                        [0, 0, 0, q*sigma_1**2, sigma_1**2]])
    
    Sigma_2=np.asarray([[sigma_2**2, q*sigma_2**2, 0, 0, 0],
                        [q*sigma_2**2, sigma_2**2, q*sigma_2**2, 0, 0],
                        [0, q*sigma_2**2, sigma_2**2, q*sigma_2**2,0],
                        [0, 0, q*sigma_2**2, sigma_2**2, q*sigma_2**2],
                        [0, 0, 0, q*sigma_2**2, sigma_2**2]])
    
    params=(Sigma_x, Sigma_1, Sigma_2, dim_split)

    arrng = config.arrng
    
    K=config.k
    
    #----------------------------------------------------------------------#
    #------------------------Train the network-----------------------------#
    #----------------------------------------------------------------------#
    
    # Set up neural network paramters
    LR = config.lr
    EPOCH = config.e
    SEED = config.seed
    input_size = 3*dim
    hidden_size = 64
    num_classes = 2
    tau = config.tau
    
    NN_params = (input_size,hidden_size,num_classes,tau)
    EVAL=False
    
    #Monte Carlo param
    T=config.t
    S=config.s
    
    CMI_LDR=[]
    CMI_DV=[]
    CMI_NWJ=[]
    
    CMI_LDR_0=[]
    CMI_DV_0=[]
    CMI_NWJ_0=[]
    
    
    CMI_LDR_1=[]
    CMI_DV_1=[]
    CMI_NWJ_1=[]
    
    CMI_LDR_2=[]
    CMI_DV_2=[]
    CMI_NWJ_2=[]
    
    
    True_CMI=0.5*np.log(det(Sigma_x+Sigma_1)/det(Sigma_1)) - 0.5*np.log(det(Sigma_x+Sigma_1+Sigma_2)/det(Sigma_1+Sigma_2))
    for s in range(S):
        ########################################################
        #   Compute I(X;Y|Z)=I(X;Y_1|Z)+ I(X;Y_2|Z,Y_1)
        #   I1=I(X;Y_1|Z)
        #   I2=I(X;Y_2|Z,Y_1)
        ########################################################
    
        hidden_size = 64
        num_classes = 2
        tau=1e-3
    
        for mode in range(3):
            if mode==0:  # I(X;Y|Z)   
                arrng=[[0],[1],[2]]
                input_size = 3*dim
                NN_params=(input_size,hidden_size,num_classes,tau)    
            elif mode==1: # I1=I(X;Y_1|Z)
                arrng=[[0],[3],[2]]
                input_size = 2*dim + dim_split
                NN_params=(input_size,hidden_size,num_classes,tau)
            elif mode==2: # I1=I(X;Y_2|Z Y_1)
                arrng=[[0],[4],[2,3]]
                input_size = 3*dim
                NN_params=(input_size,hidden_size,num_classes,tau)
            
        
        
            CMI_LDR_t=[]
            CMI_DV_t=[]
            CMI_NWJ_t=[]
            loss_t=[]
                
            #Create dataset
            dataset=CMINE.create_dataset(GenModel='Gaussian_Correlated_split',Params=params, Dim=dim, N=n)
            #dataset=CMINE.create_dataset(GenModel='Gaussian_Split',Params=params, Dim=dim, N=n)
                
            for t in range(T): 
                
                print('s,t= ',s,t)
                start_time = time.time()
                
                batch_train, target_train, joint_test, prod_test=CMINE.batch_construction(data=dataset, arrange=arrng, set_size=n//2, K_neighbor=K)    
                print('Duration of data preparation: ',time.time()-start_time,' seconds')
        
                
                
                CMI_LDR_Eval=[]
                CMI_DV_Eval=[]
                CMI_NWJ_Eval=[]
                
            
                start_time = time.time()
                
            
                #Train
                if EVAL:
                    model, loss_e, CMI_LDR_e, CMI_DV_e, CMI_NWJ_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED, Eval=True, JointEval=joint_test, ProdEval=prod_test)        
                    CMI_LDR_Eval.append(CMI_LDR_e)
                    CMI_DV_Eval.append(CMI_DV_e)    
                    CMI_NWJ_Eval.append(CMI_NWJ_e)
                else:   
                    model, loss_e = CMINE.train_classifier(BatchTrain=batch_train, TargetTrain=target_train, Params=NN_params, Epoch=EPOCH, Lr=LR, Seed=SEED)
                
                #Compute I(X;Y|Z)
                CMI_est = CMINE.estimate_CMI(model, joint_test, prod_test)
            
                print('Duration: ', time.time()-start_time, ' seconds')       
                print('Mode= ',mode)
                #print('LDR=',CMI_est[0])   
                print('DV=',CMI_est[1])   
                #print('NWJ=',CMI_est[2]) 
                print('True=',True_CMI)
                
                CMI_LDR_t.append(CMI_est[0])
                CMI_DV_t.append(CMI_est[1])
                CMI_NWJ_t.append(CMI_est[2])
            
            if mode==0:
                CMI_LDR_0.append(np.mean(CMI_LDR_t))
                CMI_DV_0.append(np.mean(CMI_DV_t))
                CMI_NWJ_0.append(np.mean(CMI_NWJ_t))  
            elif mode==1:
                CMI_LDR_1.append(np.mean(CMI_LDR_t))
                CMI_DV_1.append(np.mean(CMI_DV_t))
                CMI_NWJ_1.append(np.mean(CMI_NWJ_t))  
            elif mode==2:
                CMI_LDR_2.append(np.mean(CMI_LDR_t))
                CMI_DV_2.append(np.mean(CMI_DV_t))
                CMI_NWJ_2.append(np.mean(CMI_NWJ_t))  
        
        CMI_LDR.append(CMI_LDR_0[-1],CMI_LDR_1[-1],CMI_LDR_2[-1])
        CMI_DV.append(CMI_DV_0[-1],CMI_DV_1[-1],CMI_DV_2[-1])
        CMI_NWJ.append(CMI_NWJ_0[-1],CMI_NWJ_1[-1],CMI_NWJ_2[-1])
        
        print('DV:   ',CMI_DV_0[-1],' - ',CMI_DV_1[-1],' - ',CMI_DV_2[-1])
        print('True=',True_CMI)
        print('\n\n')   
        
    # open a file, where you ant to store the data
    file = open(config.directory+'/result_'+str(config.seed), 'wb')
    # dump information to that file
    pickle.dump((True_CMI,CMI_LDR,CMI_DV,CMI_NWJ,CMI_LDR_Eval,CMI_DV_Eval,CMI_NWJ_Eval,n,dim,K,LR,EPOCH,loss_e), file)
    
    file.close()        


