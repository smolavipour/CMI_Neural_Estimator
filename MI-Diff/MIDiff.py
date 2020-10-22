import os
import sys
#import logging
import argparse

import config
import MIDiff_Lib




################## Parsing simulation arguments ##################

parser = argparse.ArgumentParser(description='provide arguments for DI estimation')
parser.add_argument('--n',              type=int,       default=None,     help='Number of samples')
parser.add_argument('--d',              type=int,       default=None,     help='Dimension of data')
parser.add_argument('--sigma_x',        type=float,     default=None,     help='Sigma_x in the model')
parser.add_argument('--sigma_y',        type=float,     default=None,     help='Sigma_y in the model')
parser.add_argument('--sigma_z',        type=float,     default=None,     help='Sigma_z in the model')
parser.add_argument('--scenario',       type=int,       default=None,     help='0--> Estimate I(X;Y|Z)'
                                                                               '1--> Estimate I(X;Z|Y)')
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


#----     Estimate CMI      -------
MIDiff_Lib.estimate_CMI(conf)
