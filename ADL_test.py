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
            "elecNormNew",
            "nomao",
            "WISDM_ar_v1.1_transformed",
            "covtypeNorm",
            #
            "airlines",
            "RBF_f",
            "RBF_m",
            "AGR_a",
            "AGR_g",
            "LED_a",
            "LED_g",
            # #
            "kdd99",
            # #
            "gisette_scale_class_Nominal",
            "epsilon_normalized.t_class_Nominal",
            "SVHN.scale.t.libsvm.sparse_class_Nominal",
            "spam_corpus",
            "sector.scale.libsvm.class_Nominal_sparse"
            ]

# for f in AGR_a.arff AGR_g.arff WISDM_ar_v1.1_transformed.arff airlines.arff elecNormNew.arff kdd99.arff nomao.arff; do printf "'%s': [" $f; grep '@attribute' $f |grep -n '{'|awk -F ':' '{printf "%d, ", $1-1}'; printf "],\n"; done
# 'AGR_a.arff': [3, 4, 5, 9, ],
# 'AGR_g.arff': [3, 4, 5, 9, ],
# 'WISDM_ar_v1.1_transformed.arff': [1, 45, ],
# 'airlines.arff': [0, 2, 3, 4, 7, ],
# 'elecNormNew.arff': [1, 8, ],
# 'kdd99.arff': [1, 2, 3, 6, 11, 20, 21, 41, ],
# 'nomao.arff': [6, 7, 14, 15, 22, 23, 30, 31, 38, 39, 46, 47, 54, 55, 62, 63, 70, 71, 78, 79, 86, 87, 91, 95, 99, 103, 107, 111, 115, 118, ],
onehot_columns = {
    'AGR_a': [3, 4, 5],
    'AGR_g': [3, 4, 5],
    'WISDM_ar_v1.1_transformed': [1],
    'airlines': [0, 2, 3, 4],
    'elecNormNew': [1],
    'kdd99': [1, 2, 3, 6, 11, 20, 21],
    'nomao': [6, 7, 14, 15, 22, 23, 30, 31, 38, 39, 46, 47, 54, 55, 62, 63, 70, 71, 78, 79, 86, 87, 91, 95, 99, 103, 107, 111, 115],
}


for d in datasets:
    f = os.path.join(args.data_dir, d + '.mat')
    p_dump_f = os.path.join(args.save_dir, d + '_predictions.csv')
    # load data
    dataStreams = dataLoader(f,
                             batchSize=batch_size,
                             onehot_columns = onehot_columns[d] if d in onehot_columns.keys() else None)

    print('All labeled')

    # initialization
    ADLnet = ADL(dataStreams.nInput,dataStreams.nOutput, predictions_dump_file=p_dump_f)

    ADLnet0, performanceHistory0, allPerformance0 = ADLmain(ADLnet,dataStreams, trainingBatchSize=trainingBatchSize, normalize=True)

    plotPerformance(performanceHistory0[0],performanceHistory0[1],performanceHistory0[2],
                    performanceHistory0[3],performanceHistory0[4],performanceHistory0[5])

    print("Dataset,{},{}".format(d, allPerformance0[0]))
