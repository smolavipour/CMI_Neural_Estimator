import os
import sys
import time

class Config(object):
    ########## general ##########
    name = 'TSP Config'
    simulation_name= '0'
    directory = ''

    ######## randomness #########
    seed = 123

    ######## Model params #########
    n = int(8e4)                # number of samples
    d = 1                       # dimension of X,Y, and Z
    sigma_x = 10
    sigma_y = 1
    sigma_z = 5    
    arrng=[[0],[1],[2]]         # I(arg0 ; arg1 | arg2)
    
    ######## kNN params #########
    k = 2
    

    ######## training #########
    opt = "adam"                # optimizer name
    lr = 2e-3                   # Learning rate
    e = 300                     # num of epochs for training
    tau = 1e-4                  # clip the NN output [tau,1-tau]
    batch_size = n//2           # batch size for training
    t=20                        # number of trials   
    s=10                        # number of repretitions  
    
    
    
    
class Config_Additivity(object):
    ########## general ##########
    name = 'TSP Config'
    simulation_name= '0'
    directory = ''
    
    ######## randomness #########
    seed = 123

    ######## Model params #########
    n = int(8e4)                # number of samples
    d = 1                       # dimension of X,Y, and Z
    dim_split = 1                 # Y=Y1,Y2  split_d is the dimension of Y1   
    sigma_x = 10
    sigma_y = 1
    sigma_z = 5    
    q = 0.5
    arrng=[[0],[1],[2]]         # I(arg0 ; arg1 | arg2)

    
    ######## kNN params #########
    k = 2
    

    ######## training #########
    opt = "adam"                # optimizer name
    lr = 2e-3                   # Learning rate
    e = 300                     # num of epochs for training
    tau = 1e-4                  # clip the NN output [tau,1-tau]
    batch_size = n//2           # batch size for training
    t=20                        # number of trials   
    s=10                        # number of repretitions  
        
def define_configs(args):
    
    if args.scenario == 0: #Estimate I(X;Y|Z)
        config = Config()
    elif args.scenario == 1: #Estimate I(X;Z|Y)
        config = Config()
        config.arrng=[[0],[2],[1]]
    elif args.scenario == 2: # Test Additivity and DPI
        config = Config_Additivity()
    else:
        raise ValueError("Invalid choice of configuration")

    config = read_flags(config, args)

    #seed_tmp = time.time()
    #config.seed = int((seed_tmp - int(seed_tmp))*1e6) if args.seed is None else args.seed
    #print(config.seed)

    simulation_name = "CMI_kNN_d" + str(config.d) + '_Scenario_' + str(args.scenario)
    config.simulation_name = simulation_name
    config.directory = directory = "{}/results/{}".format(os.getcwd(),
                                                             simulation_name)
    if not os.path.exists(directory):
        os.makedirs(directory) 

    return config    

def read_flags(config, args):
    # assign flags into config
    for arg in sorted(vars(args)):
        key = arg
        val = getattr(args, arg)
        if val is not None:
            setattr(config, key, val)

    return config
