import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# MLP
import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


def sample_batch(data, arrange=[[0],[1],[2]], batch_size=100, sample_mode='joint', K_neighbor=10, radius=1000):
    #data is represented as a tuple (x,y,u,w)
    #arrange is a triple that each determines what random variables are placed in what positions
    # I(X;Y|Z) where Z=(U,W)
    #(X,Y,Z)=data
    
    #X=data[arrange[0][0]]
    #Y=data[arrange[1][0]]
    X=np.concatenate([data[i] for i in arrange[0]],axis=1)
    Y=np.concatenate([data[i] for i in arrange[1]],axis=1)
    Z=np.concatenate([data[i] for i in arrange[2]],axis=1)
    
    N=X.shape[0]
    
    if sample_mode == 'joint':
        #sample according to p(x,y,z)
        index = np.random.choice(range(N), size=batch_size, replace=False)
        batch = np.concatenate((X[index],Y[index],Z[index]),axis=1)        
                
    elif sample_mode == 'prod_iso_kNN':
        #In this case we first pick m=batch_size/K_neighbor x. Then we look for neighbors among the rest of samples
        #Note that in nearest neighbor we should not consider the point itself as neighbor
        m=batch_size//K_neighbor
        index_yz = np.random.choice(range(N), size=m, replace=False)
        neigh = NearestNeighbors(n_neighbors=K_neighbor, radius=radius,metric='euclidean')
        X2=np.asarray([element for i, element in enumerate(X) if i not in index_yz])
        Y2=np.asarray([element for i, element in enumerate(Y) if i not in index_yz])
        Z2=np.asarray([element for i, element in enumerate(Z) if i not in index_yz])
        neigh.fit(Z2)
        Neighbor_indices=neigh.kneighbors(Z[index_yz],return_distance=False)
        index_x=[]
        index_y=[]
        index_z=[]
        for n_i in Neighbor_indices:
            index_x=np.append(index_x,n_i).astype(int)
        for ind in index_yz:
            index_y=np.append(index_y,[ind]*K_neighbor).astype(int)
            index_z=np.append(index_z,[ind]*K_neighbor).astype(int)
                    
        batch = np.column_stack((X2[index_x],Y[index_y],Z[index_z]))
    

    return batch

def sample_batch_memory(data, arrange=[[0],[1],[2]], depth=0, batch_size=100, sample_mode='joint', K_neighbor=10, radius=1000):
    #data is represented as a tuple (x,y,u,w)
    #arrange is a triple that each determines what random variables are placed in what positions
    # I(X->Y||Z) where Z=(U,W)
    #(X,Y,Z)=data
    
    X=data[arrange[0][0]]
    Y=data[arrange[1][0]]
    Z=np.concatenate([data[i] for i in arrange[2]],axis=1)
    
    n=X.shape[0]
    r=X.shape[1]
    # Create data table with time shifts
    # [X(t-2) X(t-1) X(t), Y(t-2) Y(t-1) Y(t), Z(t-2) Z(t-1) Z(t)]

    X_G=X[0:n-depth]
    Y_G=Y[0:n-depth]
    Z_G=Z[0:n-depth]
    
        
    for l in range(depth):       
        X_G=np.column_stack((X_G, X[1+l:n-depth+l+1]))
        Y_G=np.column_stack((Y_G, Y[1+l:n-depth+l+1]))
        Z_G=np.column_stack((Z_G, Z[1+l:n-depth+l+1]))         
    

    if sample_mode == 'joint':
        #sample according to p(x,y,z)
        index = np.random.choice(range(n-depth), size=batch_size//K_neighbor*K_neighbor, replace=False)
        batch=np.column_stack((X_G[index],Y_G[index],Z_G[index]))        
    elif sample_mode == 'prod_iso_kNN':
        #sample according to p(y^{d+1},z^{d+1})p(x^{d+1}|y^{d},z^{d+1})    
        m=batch_size//K_neighbor
        index_yz = np.random.choice(range(n-depth), size=m, replace=False) 
        neigh = NearestNeighbors(n_neighbors=K_neighbor, radius=radius,metric='euclidean')        
        
        # Create the condition 
        if depth>0:
            cond = np.column_stack((np.delete(Y_G,depth*r+np.arange(r),1),Z_G)) # Y_1^d Z_1^{d+1} 
        else:
            cond = np.reshape(Z_G,(n-depth,1))    
        #Exclude the m indices from the dataset
        X_G_ex = np.asarray([element for i, element in enumerate(X_G) if i not in index_yz])
        #Y_G_ex = np.asarray([element for i, element in enumerate(Y_G) if i not in index_yz])
        #Z_G_ex = np.asarray([element for i, element in enumerate(Z_G) if i not in index_yz])    
        cond_ex = np.asarray([element for i, element in enumerate(cond) if i not in index_yz])  
        
        neigh.fit(cond_ex)                
        index_x = []    
        Neighbor_indices=neigh.kneighbors(cond[index_yz],return_distance=False)

        index_x=[]
        index_y=[]
        index_z=[]
        for n_i in Neighbor_indices:
            index_x=np.append(index_x,n_i).astype(int)
        for ind in index_yz:
            index_y=np.append(index_y,[ind]*K_neighbor).astype(int)
            index_z=np.append(index_z,[ind]*K_neighbor).astype(int)
        

        batch = np.column_stack((X_G_ex[index_x],Y_G[index_y],Z_G[index_z]))
    return batch

def batch_construction(data,arrange,set_size=100,K_neighbor=2, mode='memoryless', depth=0):   
    n = data[0].shape[0]
                     
    
    
    if mode=='memoryless':
        train_index = np.random.choice(range(n), size=set_size, replace=False) 
        test_index = [j for j in range(n) if j not in train_index]        
    
            
        Train_set = [data[i][train_index] for i in range(len(data))]
        Test_set = [data[i][test_index] for i in range(len(data))]

        joint_target = np.repeat([[1,0]],set_size,axis=0)
        prod_target = np.repeat([[0,1]],set_size,axis=0)
        target_train = np.concatenate((joint_target,prod_target),axis=0)
        target_train = autograd.Variable(torch.tensor(target_train).float())
        
        joint_train = sample_batch(Train_set, arrange, batch_size=set_size,sample_mode='joint',K_neighbor=K_neighbor)
        prod_train = sample_batch(Train_set, arrange, batch_size=set_size,sample_mode='prod_iso_kNN',K_neighbor=K_neighbor)
        batch_train = autograd.Variable(torch.tensor(np.concatenate((joint_train, prod_train))).float())
        
        joint_test = sample_batch(Test_set, arrange, batch_size=set_size,sample_mode='joint',K_neighbor=K_neighbor)
        joint_test = autograd.Variable(torch.tensor(joint_test).float())
        prod_test = sample_batch(Test_set, arrange, batch_size=set_size,sample_mode='prod_iso_kNN',K_neighbor=K_neighbor)
        prod_test = autograd.Variable(torch.tensor(prod_test).float())
    elif mode=='with_memory':
        joint_target = np.repeat([[1,0]],set_size,axis=0)
        prod_target = np.repeat([[0,1]],set_size,axis=0)
        joint_data = sample_batch_memory(data, arrange, depth, batch_size=(n-depth),sample_mode='joint',K_neighbor=K_neighbor)
        prod_data = sample_batch_memory(data, arrange, depth, batch_size=(n-depth),sample_mode='prod_iso_kNN',K_neighbor=K_neighbor)        
        
        #np.random.shuffle(indj)
        #train with (Folds-1)/Folds of the data

        input_data=np.concatenate((joint_data[0:set_size],prod_data[0:set_size]))
        input_target=np.concatenate((joint_target[0:set_size],prod_target[0:set_size]))
        #Prepare tensors
        batch_train=autograd.Variable(torch.tensor(input_data).float())
        target_train=autograd.Variable(torch.tensor(input_target).float())
        
        joint_test=autograd.Variable(torch.tensor(joint_data[set_size:n]).float())
        prod_test=autograd.Variable(torch.tensor(prod_data[set_size:n]).float())

    return batch_train, target_train, joint_test, prod_test
    

# Classifier
class ClassifierModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,tau):
        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.h3 = nn.Linear(hidden_size, num_classes)
        self.Tau=tau

    def forward(self, x):
        x = self.h1(x)
        x = F.relu(x)
        x = self.h2(x)
        x = F.relu(x)
        x = self.h3(x)
        x = F.softmax(x, dim=1)
        hardT = nn.Hardtanh(self.Tau, 1-self.Tau)
        x=hardT(x)
        return x  


def estimate_CMI(Model,JointBatch,ProdBatch):
    gamma_joint = Model(JointBatch).detach().numpy()[:,0]
    gamma_prod = Model(ProdBatch).detach().numpy()[:,0]
    
    b =JointBatch.shape[0]
    b_=ProdBatch.shape[0]
    sum1=0
    for j in range(b):  
        sum1+=np.log(gamma_joint[j]/(1-gamma_joint[j]))

    sum2=0
    for j in range(b_):
        sum2=sum2+gamma_prod[j]/(1-gamma_prod[j])
            
    CMI_LDR=(1/b)*sum1
    CMI_DV=(1/b)*sum1 - np.log((1/b_)*sum2)
    CMI_NWJ=1+(1/b)*sum1 - (1/b_)*sum2        
    return CMI_LDR, CMI_DV, CMI_NWJ
    
def train_classifier(BatchTrain, TargetTrain, Params, Epoch, Lr, Seed, Epsilon=1e-7, Eval=False, JointEval=[], ProdEval=[]):
    loss_e=[]
    last_loss=1000
    CMI_LDR_e=[]
    CMI_DV_e=[]
    CMI_NWJ_e=[]

    #Set up the model
    torch.manual_seed(Seed)
    (input_size, hidden_size, num_classes, tau)=Params
    model = ClassifierModel(input_size, hidden_size, num_classes, tau)    
    opt = optim.Adam(params=model.parameters(), lr=Lr)

    for epoch in range(int(Epoch)):
        out = model(BatchTrain)        
        _, pred = out.max(1)        

        loss = F.binary_cross_entropy(out, TargetTrain) 
        loss_e.append(loss.detach().numpy())
        
        if Eval:
            CMI_eval = estimate_CMI(model,JointEval,ProdEval)
            print('epoch: ',epoch,'  ,', CMI_eval[1], ' loss: ',loss_e[-1])        
            CMI_LDR_e.append(CMI_eval[0])
            CMI_DV_e.append(CMI_eval[1])
            CMI_NWJ_e.append(CMI_eval[2])        
        
        if abs(loss-last_loss)<Epsilon and epoch>50:
            print('epoch=',epoch)
            break

        last_loss=loss
        
        model.zero_grad()
        loss.backward()
        opt.step()
    if Eval:    
        return model, loss_e, CMI_LDR_e, CMI_DV_e, CMI_NWJ_e
    else:
        return model, loss_e



def create_dataset(GenModel, Params, Dim, N):
    # If the dataset=(x,y,z) we compute I(X;Y|Z)    
    if GenModel=='Gaussian_nonZero':
        (sigma_x,sigma_1,sigma_2)= Params
        x = np.random.multivariate_normal(mean=[0]*Dim,
                                     cov=sigma_x**2*np.eye(Dim),
                                     size=N) 

        y = x + np.random.multivariate_normal(mean=[0]*Dim,
                                     cov=sigma_1**2*np.eye(Dim),
                                     size=N)     

    
        z = y + np.random.multivariate_normal(mean=[0]*Dim,
                                     cov=sigma_2**2*np.eye(Dim),
                                     size=N)     
        dataset=[x,y,z]

    elif GenModel=='Gaussian_Correlated':
        (Sigma_x,Sigma_1,Sigma_2)= Params
        x = np.random.multivariate_normal(mean=[0]*Dim,
                                     cov=Sigma_x,
                                     size=N) 

        y = x + np.random.multivariate_normal(mean=[0]*Dim,
                                     cov=Sigma_1,
                                     size=N)     

    
        z = y + np.random.multivariate_normal(mean=[0]*Dim,
                                     cov=Sigma_2,
                                     size=N)     
        dataset=[x,y,z]


    elif GenModel=='Gaussian_Split':
        (sigma_x, sigma_1, sigma_2, dim_split)= Params
        x = np.random.multivariate_normal(mean=[0]*Dim,
                                     cov=sigma_x**2*np.eye(Dim),
                                     size=N) 

        y = x + np.random.multivariate_normal(mean=[0]*Dim,
                                     cov=sigma_1**2*np.eye(Dim),
                                     size=N)     
        
        y1= y[:,0:dim_split]
        y2= y[:,dim_split:Dim]
    
        z = y + np.random.multivariate_normal(mean=[0]*Dim,
                                     cov=sigma_2**2*np.eye(Dim),
                                     size=N)     
        dataset=[x,y,z,y1,y2]

    elif GenModel=='Gaussian_Correlated_split':
        (Sigma_x,Sigma_1,Sigma_2, dim_split)= Params
        x = np.random.multivariate_normal(mean=[0]*Dim,
                                     cov=Sigma_x,
                                     size=N) 

        y = x + np.random.multivariate_normal(mean=[0]*Dim,
                                     cov=Sigma_1,
                                     size=N)     

    
        z = y + np.random.multivariate_normal(mean=[0]*Dim,
                                     cov=Sigma_2,
                                     size=N)    
        
        y1= y[:,0:dim_split]
        y2= y[:,dim_split:Dim]
        
        dataset=[x,y,z,y1,y2]


    return dataset


def prepare_dataset(dataset, Mode='max',S=4):
    if Mode=='max':
        #map the data between [-S,S]
        #scaling the whole dataset with one factor to be able to compute the true CMI   
        
        for i in range(len(dataset)):  
            MAX=np.max(dataset[i])
            MIN=np.min(dataset[i])
            
            dataset[i]=2*S*(dataset[i]-MIN)/(MAX-MIN)-S
        return dataset
    elif Mode=='norm':
        #Make each column zero mean and devided to SD
        for i in range(len(dataset)):  
            MEAN=dataset[i].mean()
            STD=dataset[i].std()
            
            dataset[i]=2*(dataset[i]-MEAN)/STD
        return dataset        
    