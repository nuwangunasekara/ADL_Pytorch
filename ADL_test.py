from ADLmainloop import ADLmain, ADLmainId
from ADLbasic import ADL
from utilsADL import dataLoader, plotPerformance
import random
import torch
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/Users/ng98/Desktop/datasets/NEW/mat/',
                    help='Save Directory')
parser.add_argument('--save_dir', type=str, default='/Users/ng98/Desktop/datasets/NEW/mat/',
                    help='Save Directory')
parser.add_argument('--random_seed', type=int, default=1, help='Random seed')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for test and train')
args = parser.parse_args()

# random seed control
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
random.seed(args.random_seed)

batch_size = args.batch_size
trainingBatchSize = args.batch_size

datasets = [
            # "elecNormNew",
            # "nomao",
            # "WISDM_ar_v1.1_transformed",
            # "covtypeNorm",
            #
            # "airlines",
            # "RBF_f",
            # "RBF_m",
            # "AGR_a",
            # "AGR_g",
            # "LED_a",
            # "LED_g",
            # #
            "kdd99",
            # #
            # "gisette_scale_class_Nominal",
            # "epsilon_normalized.t_class_Nominal",
            # "SVHN.scale.t.libsvm.sparse_class_Nominal",
            # "spam_corpus",
            # "sector.scale.libsvm.class_Nominal_sparse"
            ]

for d in datasets:
    f = os.path.join(args.data_dir, d + '.mat')
    p_dump_f = os.path.join(args.save_dir, d + '_predictions.csv')
    # load data
    dataStreams = dataLoader(f, batchSize=batch_size)

    print('All labeled')

    # initialization
    ADLnet = ADL(dataStreams.nInput,dataStreams.nOutput, predictions_dump_file=p_dump_f)

    ADLnet0, performanceHistory0, allPerformance0 = ADLmain(ADLnet,dataStreams, trainingBatchSize=trainingBatchSize)

    plotPerformance(performanceHistory0[0],performanceHistory0[1],performanceHistory0[2],
                    performanceHistory0[3],performanceHistory0[4],performanceHistory0[5])

    print("Dataset,{},{}".format(d, allPerformance0[0]))
