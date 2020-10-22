import numpy as np
import time
import pickle

import torch
from torch import autograd, nn, optim
import torch.nn.functional as F

def sample_batch(data, batch_size=100, sample_mode='joint'):
    if len(data) == 3:
        #data is represented as [x,y,z]
        (X,Y,Z) = data
        N = X.shape[0]
    elif len(data) == 2:
        #data is represented as [x,y]
        (X,Y) = data
        N = X.shape[0]

    if sample_mode == 'joint':
        #sample according to p(x,y,z)
        index = np.random.choice(range(N), size=batch_size, replace=False)
        if len(data) == 3:
            batch = np.concatenate((X[index],Y[index],Z[index]),axis=1)        
        elif len(data) == 2:
            batch = np.concatenate((X[index],Y[index]),axis=1)        

        
    elif sample_mode == 'prod':
        
        if len(data) == 3:
            #p(yz)p(x)
            #It is allowed to pick more samples by allowing repeation
            index_1 = np.random.choice(range(N), size=batch_size, replace=True)
            index_2 = np.random.choice(range(N), size=batch_size, replace=True)                
            batch = np.concatenate((X[index_1], Y[index_2],Z[index_2]),axis=1)                      
        if len(data) == 2:
            #p(x)p(y)
            #It is allowed to pick more samples by allowing repeation
            index_1 = np.random.choice(range(N), size=batch_size, replace=True)
            index_2 = np.random.choice(range(N), size=batch_size, replace=True)                
            batch = np.concatenate((X[index_1], Y[index_2]),axis=1)          

    return batch


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, tau):
        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.h3 = nn.Linear(hidden_size, num_classes)
        self.Tau = tau

    def forward(self, x):
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        x = self.h3(x)
        x = F.softmax(x,dim=1)
        hardT = nn.Hardtanh(self.Tau, 1-self.Tau)
        x = hardT(x)
        return x  



def estimate_CMI(config):
    
    n = config.n
    d = config.d
    
    
    hidden_size = 64
    num_classes = 2
    
    tau = config.tau
    
    sigma_x = config.sigma_x
    sigma_1 = config.sigma_y
    sigma_2 = config.sigma_z
    
    b_size = config.batch_size
    
    LR = config.lr
    EPOCH = config.e  
     
    LR2 = config.lr
    EPOCH2 = config.e      
    
    T = config.t
    S = config.s
    
    #I(X;YZ)
    MI_DV_1 = []
    MI_NWJ_1 = []
    part1_1 = []
    part2_1 = []
    
    #I(X;Z)
    MI_DV_2 = []
    MI_NWJ_2 = []
    part1_2 = []
    part2_2 = []
        
    Estimated_CMI_DV = []
    Estimated_CMI_NWJ = []
        
    for s in range(S):
        if config.scenario == 0:
            x = np.random.multivariate_normal(mean=[0]*d,
                                             cov=sigma_x**2*np.eye(d),
                                             size=n)         
            y = x + np.random.multivariate_normal(mean=[0]*d,
                                             cov=sigma_1**2*np.eye(d),
                                             size=n)             
            z = y + np.random.multivariate_normal(mean=[0]*d,
                                             cov=sigma_2**2*np.eye(d),
                                             size=n)     
        elif config.scenario == 1:
            x = np.random.multivariate_normal(mean=[0]*d,
                                             cov=sigma_x**2*np.eye(d),
                                             size=n)         
            z = x + np.random.multivariate_normal(mean=[0]*d,
                                             cov=sigma_1**2*np.eye(d),
                                             size=n)             
            y = z + np.random.multivariate_normal(mean=[0]*d,
                                             cov=sigma_2**2*np.eye(d),
                                             size=n)     

            
        print('I(X;YZ)')
        ##-----------------------------------------------------------
        ##             I(X;YZ)       
        ##-----------------------------------------------------------
        input_size = 3*d       
        MI_DV_1_t = []
        MI_NWJ_1_t = []
        part1_1_t = []
        part2_1_t = []
        
        for t in range(T):
            print('t=',t)
            start_time = time.time()
    
            train_index = np.random.choice(range(n), size=b_size, replace=False) 
            test_index = [j for j in range(n) if j not in train_index]        
            
            Train_set = (x[train_index],y[train_index],z[train_index])
            Test_set = (x[test_index],y[test_index],z[test_index])
                                    
            Train_target = np.concatenate((np.repeat([[1,0]],b_size,axis=0),np.repeat([[0,1]],b_size,axis=0) ),axis=0)
            Train_joint = sample_batch(data=Train_set,batch_size=b_size,sample_mode='joint')
            Train_marginal = sample_batch(data=Train_set,batch_size=b_size,sample_mode='prod')
            Train_batch = np.concatenate((Train_joint,Train_marginal))
                            
            Test_joint = sample_batch(data=Test_set,batch_size=b_size,sample_mode='joint')
            Test_marginal = sample_batch(data=Test_set,batch_size=b_size,sample_mode='prod')
                        
            torch.manual_seed(config.seed)
                       
            Train_target_tensor = autograd.Variable(torch.tensor(Train_target).float())
            Train_batch_tensor = autograd.Variable(torch.tensor(Train_batch).float())
            
            Last_loss = 1000
            model = Model(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes,tau=tau)
            opt = optim.Adam(params=model.parameters(), lr=LR)

            for epoch in range(EPOCH):
                out = model(Train_batch_tensor)
                _, pred = out.max(1)        
        
                loss = F.binary_cross_entropy(out, Train_target_tensor) 
                #print(abs(loss-Last_loss))    
                if(abs(loss-Last_loss)<1e-7):
                    print('epoch=',epoch)
                    break
        
                Last_loss = loss
        
                model.zero_grad()
                loss.backward()
                opt.step()
                
            #Compute I(X;YZ)
            Test_joint_tensor = autograd.Variable(torch.tensor(Test_joint).float())
            Test_marginal_tensor = autograd.Variable(torch.tensor(Test_marginal).float())
            
            Test_joint_out = model(Test_joint_tensor)
            gamma_joint=Test_joint_out.detach().numpy()[:,0]
            
            
            Test_marginal_out = model(Test_marginal_tensor)
            gamma_marginal = Test_marginal_out.detach().numpy()[:,0]
            
            sum1 = 0
            for k in range(b_size):  
                sum1 += np.log(gamma_joint[k]/(1-gamma_joint[k]))
            
            sum2=0
            for k in range(b_size):
                sum2 += gamma_marginal[k]/(1-gamma_marginal[k])
            
            print('Duration: ',time.time()-start_time,' seconds')
            MI_DV_1_t.append((1/b_size)*sum1 - np.log((1/b_size)*sum2))
            MI_NWJ_1_t.append(1+(1/b_size)*sum1 - (1/b_size)*sum2)
            part1_1_t.append((1/b_size)*sum1)
            part2_1_t.append(1- (1/b_size)*sum2)
            print('DV_t=',MI_DV_1_t[-1])   
            print('NWJ_t=',MI_NWJ_1_t[-1])
            if config.scenario == 0:
                #I(X;YZ)=I(X;Y) in this case
                True_MI_1 = -d*0.5*np.log(sigma_1**2 /(sigma_1**2 + sigma_x**2)) 
                print('True I(X;YZ)',True_MI_1)
        print('*****')    
        
        MI_DV_1.append(np.mean(MI_DV_1_t))
        MI_NWJ_1.append(np.mean(MI_NWJ_1_t))
        part1_1.append(np.mean(part1_1_t))
        part2_1.append(np.mean(part2_1_t))
            
        print('Averaged Estimated DV=', MI_DV_1[-1])   
        print('Averaged Estimated NWJ=',MI_NWJ_1[-1])
        if config.scenario == 0:
            print('True I(X;YZ)=',True_MI_1,'\n')
        print('-----------\n')

        print('Now I(X;Z)')
        ##-----------------------------------------------------------
        ##             I(X;Z)       
        ##-----------------------------------------------------------
        input_size = 2*d
        
        MI_DV_2_t = []
        MI_NWJ_2_t = []
        part1_2_t = []
        part2_2_t = [] 
           
        for t in range(T):
            print('t=',t)
            start_time = time.time()
    
            train_index = np.random.choice(range(n), size=b_size, replace=False) 
            test_index = [j for j in range(n) if j not in train_index]        
            
            Train_set = (x[train_index],z[train_index])
            Test_set = (x[test_index],z[test_index])
                         
            Train_target = np.concatenate((np.repeat([[1,0]],b_size,axis=0),np.repeat([[0,1]],b_size,axis=0) ),axis=0)
            Train_joint = sample_batch(data=Train_set,batch_size=b_size,sample_mode='joint')
            Train_marginal = sample_batch(data=Train_set,batch_size=b_size,sample_mode='prod')
            Train_batch = np.concatenate((Train_joint,Train_marginal))
                            
            Test_joint = sample_batch(data=Test_set,batch_size=b_size,sample_mode='joint')
            Test_marginal = sample_batch(data=Test_set,batch_size=b_size,sample_mode='prod')
                        
            torch.manual_seed(config.seed)
            
            
            Train_target_tensor = autograd.Variable(torch.tensor(Train_target).float())
            Train_batch_tensor = autograd.Variable(torch.tensor(Train_batch).float())
            
            Last_loss = 1000
            model = Model(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes,tau=tau)
            opt = optim.Adam(params=model.parameters(), lr=LR2)
            for epoch in range(EPOCH2):
                out = model(Train_batch_tensor)
                _, pred = out.max(1)        
        
                loss = F.binary_cross_entropy(out, Train_target_tensor) 
                #print(abs(loss-Last_loss))    
                if(abs(loss-Last_loss)<1e-7):
                    print('epoch=',epoch)
                    break
        
                Last_loss = loss
        
                model.zero_grad()
                loss.backward()
                opt.step()
                
            #Compute I(X;Z)            
            Test_joint_tensor = autograd.Variable(torch.tensor(Test_joint).float())
            Test_marginal_tensor = autograd.Variable(torch.tensor(Test_marginal).float())
            
            Test_joint_out = model(Test_joint_tensor)
            gamma_joint = Test_joint_out.detach().numpy()[:,0]
            
            
            Test_marginal_out = model(Test_marginal_tensor)
            gamma_marginal = Test_marginal_out.detach().numpy()[:,0]
            
            sum1=0
            for k in range(b_size):  
                sum1 += np.log(gamma_joint[k]/(1-gamma_joint[k]))
            
            sum2=0
            for k in range(b_size):
                sum2 += gamma_marginal[k]/(1-gamma_marginal[k])
            
            print('Duration: ',time.time()-start_time,' seconds')
            MI_DV_2_t.append((1/b_size)*sum1 - np.log((1/b_size)*sum2))
            MI_NWJ_2_t.append(1+(1/b_size)*sum1 - (1/b_size)*sum2)
            part1_2_t.append((1/b_size)*sum1)
            part2_2_t.append(1- (1/b_size)*sum2)
            print('DV_t=',MI_DV_2_t[-1])   
            print('NWJ_t=',MI_NWJ_2_t[-1])
            if config.scenario == 0:
                True_MI_2=d*0.5*np.log((sigma_x**2+sigma_1**2 + sigma_2**2)/(sigma_1**2 + sigma_2**2))
                print('True I(X;Z)',True_MI_2)
        print('*****')    
    
        MI_DV_2.append(np.mean(MI_DV_2_t))
        MI_NWJ_2.append(np.mean(MI_NWJ_2_t))
        part1_2.append(np.mean(part1_2_t))
        part2_2.append(np.mean(part2_2_t))    
            
        print('Averaged Estimated DV=', MI_DV_2[-1])   
        print('Averaged Estimated NWJ=',MI_NWJ_2[-1]) 
        if config.scenario == 0:
            print('True I(X;Z)=',True_MI_2,'\n')
        print('*****') 
        print('trial=',s)
        Estimated_CMI_DV.append(MI_DV_1[-1] - MI_DV_2[-1])
        Estimated_CMI_NWJ.append(MI_NWJ_1[-1] - MI_NWJ_2[-1])
        
        print('CMI_Averaged Estimated DV=',Estimated_CMI_DV[-1])
        print('CMI_Averaged Estimated NWJ=',Estimated_CMI_NWJ[-1])
        
        if config.scenario == 0:
            True_CMI = True_MI_1 - True_MI_2
            print('True I(X;Y|Z)',True_MI_1-True_MI_2)
        elif config.scenario == 1:
            True_CMI = 0
            print('True I(X;Y|Z)',True_CMI)
            

    file = open(config.directory+'/result_'+str(config.seed), 'wb')
    pickle.dump((True_CMI,Estimated_CMI_DV,Estimated_CMI_NWJ,MI_DV_1,MI_NWJ_1,part1_1,part2_1,MI_DV_2,MI_NWJ_2,part1_2,part2_2,T,n,b_size,LR,EPOCH), file)
    
    file.close()    
    