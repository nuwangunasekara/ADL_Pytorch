addpath("./matlab2weka/")


javaaddpath("/Volumes/weka-3-8-4/weka-3-8-4/weka.jar")

if wekaPathCheck()
    disp('Weka.jar is loaded!')
end

dataset_dir = "/Users/ng98/Desktop/datasets/NEW/unzipped/"
save_dir = "/Users/ng98/Desktop/datasets/NEW/mat/"
% datasets = ["WISDM_ar_v1.1_transformed covtypeNorm" "epsilon_normalized.t_class_Nominal" "AGR_g" "real-sim.libsvm.class_Nominal_sparse" "elecNormNew" "SVHN.scale.t.libsvm.sparse_class_Nominal" "sector.scale.libsvm.class_Nominal_sparse" "RBF_m" ]
% datasets = ["kdd99"]
% datasets = ["gisette_scale_class_Nominal"]
% datasets = ["sector.scale.libsvm.class_Nominal_sparse"]
% datasets = ["spam_corpus"]
datasets = ["WISDM_ar_v1.1_transformed"]
for datasetName = datasets
    in_f = strcat(dataset_dir, strcat(datasetName,'.arff'))
    out_f = strcat(save_dir, strcat(datasetName,'.mat'))

    wekaOBJ = loadARFF(in_f)
    
    [mdata,featureNames,targetNDX,stringVals,relationName] = weka2matlab(wekaOBJ)
    
    data = mdata
    save (sprintf('%s', out_f), 'data', '-v7.3')
    clear data
    clear mdata featureNames targetNDX stringVals relationName

end