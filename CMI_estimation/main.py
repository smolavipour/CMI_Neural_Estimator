import os
import sys
#import logging
import argparse

import time
from time import sleep
import numpy as np
import pickle

import config
import CMINE




################## Parsing simulation arguments ##################

parser = argparse.ArgumentParser(description='provide arguments for DI estimation')
parser.add_argument('--n',              type=int,       default=None,     help='Number of samples')
parser.add_argument('--d',              type=int,       default=None,     help='Dimension of data')
parser.add_argument('--k',              type=int,       default=None,     help='k, the parameter of the kNN method')
parser.add_argument('--model',          type=str,       default=None,     help='Uncorrelated or Correlated Gaussian')
parser.add_argument('--sigma_x',        type=float,     default=None,     help='Sigma_x in the model')
parser.add_argument('--sigma_y',        type=float,     default=None,     help='Sigma_y in the model')
parser.add_argument('--sigma_z',        type=float,     default=None,     help='Sigma_z in the model')
parser.add_argument('--q',        type=float,     default=None,     help='Correlated coefficient')
parser.add_argument('--scenario',       type=int,       default=None,     help='0--> Estimate I(X;Y|Z)'
                                                                               '1--> Estimate I(X;Z|Y)'
                                                                               '2--> Test additivity and DPI')
parser.add_argument('--lr',             type=float,     default=None,     help='Learning Rate')
parser.add_argument('--e',              type=int,       default=None,     help='Epochs')
parser.add_argument('--tau',            type=float,     default=None,     help='Tau')
parser.add_argument('--t',              type=int,       default=None,     help='# of trials')
parser.add_argument('--s',              type=int,       default=None,     help='# of times to repeat')
parser.add_argument('--seed',           type=int,       default=None,     help='Seed')
                    
parser.add_argument('--verbose',        dest='verbose', action='store_true')
parser.set_defaults(verbose=False)

args = parser.parse_args()


#----  Setup the configureation  --------
conf = config.define_configs(args)

if conf.scenario==0 or conf.scenario==1:
    #----     Estimate CMI      -------
    CMINE.estimate_CMI(conf)
elif conf.scenario==2:
    #---- Estimate DPI Additivity  -------
    CMINE.estimate_CMI_DPI(conf)
