from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.abod import ABOD
from pyod.models.hbos import HBOS
from joblib import dump, load
import visualizations
import pandas as pd
import numpy as np
import os

# Performs cross-validation with one-class models based on pre-extracted .csv files with linguistic features
# Models are saved for external use

folds = 5

feature_settings = ["text", "repetitiveness", "semantic", "readability", "pos"]
feat_attr = "essay"
class_attr = "class"
outdir = "results/one_class_models/" + str(folds) + "_CV_stratified_true/"

# ___________________________________________________________________________
# SHORT ESSAYS (BALANCED) - ES
# ___________________________________________________________________________
language = 'es'
filename = "spanish/short-essays-balanced-merged-cv.csv"
filename_ling_feat = "spanish/essays-linguistic-features-all.csv"
encoding = 'latin'
# ___________________________________________________________________________
# SHORT ESSAYS (BALANCED) - EN
# ___________________________________________________________________________
# language = 'en'
# filename = "english/short-essays-en-balanced-merged-cv.csv"
# filename_ling_feat = "english/essays-linguistic-features-all.csv"
# encoding = 'iso-8859-1'
# ___________________________________________________________________________

comments = pd.read_csv(filename, encoding=encoding)
features = comments[feat_attr]
labels = comments[class_attr]

feat_dict = {   # Start and end positions for each feature subset
    "text": [0, 5],
    "repetitiveness": [6, 10],
    "semantic": [11, 15],
    "readability": [16, 28],
    "pos": [29, 48]
}

# k-Fold CV using linguistic features
models = [OneClassSVM(gamma='auto', kernel='rbf'),
          OneClassSVM(gamma='auto', kernel='poly'),
          OneClassSVM(gamma='auto', kernel='sigmoid'),
          OneClassSVM(gamma='auto', kernel='linear'),
          LocalOutlierFactor(n_neighbors=3, novelty=True),
          LocalOutlierFactor(n_neighbors=5, novelty=True),
          LocalOutlierFactor(n_neighbors=10, novelty=True),
          IsolationForest(random_state=0),
          ABOD(contamination=0.01),
          HBOS(contamination=0.01)]

skf = StratifiedKFold(n_splits=folds)
skf.get_n_splits(features, labels)

for feature_setting in feature_settings:

    print(f'\n\nFeature setting: {feature_setting}')

    ling_features_all = np.loadtxt(filename_ling_feat, delimiter=",")
    ling_features_all = ling_features_all[:, feat_dict[feature_setting][0]:feat_dict[feature_setting][1] + 1]

    for m in models:
        print(f'\n\n{m}')

        preds_from_folds = []
        labels_from_folds = []

        for i, (train_index, test_index) in enumerate(skf.split(features, labels)):

            ling_feat_train = ling_features_all[train_index, :]
            labels_train = labels[train_index]

            # Consider human essays only for training (class 0) by removing AI-generated essays from training
            ling_feat_train = ling_feat_train[labels[train_index] == 0]
            labels_train = labels_train[labels[train_index] == 0]
            ling_feat_test = ling_features_all[test_index, :]
            labels_test = labels[test_index]
            clf = m.fit(ling_feat_train)

            # Predict and evaluate on balanced folds (class 0/1)
            preds = clf.predict(ling_feat_test)

            for l in labels[test_index]:
                labels_from_folds.append(l)

            for p in preds:
                preds_from_folds.append(p)

        labels_folds_np = np.array(labels_from_folds)  # After using the full set of labels to slice from as folds are generated, we don't need it anymore
        preds_folds_np = np.array(preds_from_folds)

        # Converting prediction format to ground truth (0: human, 1: AI) - Only needed for SkLearn models, not Pyod
        if (str(m).__contains__("contamination") == False):
            #preds = np.where(preds == 1, 0, preds)
            np.place(preds_folds_np, preds_folds_np == 1, 0)
            #preds = np.where(preds == -1, 1, preds)
            np.place(preds_folds_np, preds_folds_np == -1, 1)

        dump(m, "MODELS/" + language + "/" + str(m) + ".joblib")

        # Compute metrics
        cf = confusion_matrix(labels_folds_np, preds_folds_np, labels=[0, 1])
        print(cf)

        [precision_weighted, recall_weighted, fscore_weighted, support_RF_weighted] = precision_recall_fscore_support(labels_folds_np, preds_folds_np, average='weighted')
        [precision_micro, recall_micro, fscore_micro, support_RF_micro] = precision_recall_fscore_support(labels_folds_np, preds_folds_np, average='micro')
        [precision_macro, recall_macro, fscore_macro, support_RF_macro] = precision_recall_fscore_support(labels_folds_np, preds_folds_np, average='macro')

        print(str(precision_weighted) + "," + str(recall_weighted) + "," + str(fscore_weighted))
        print(str(precision_micro) + "," + str(recall_micro) + "," + str(fscore_micro))
        print(str(precision_macro) + "," + str(recall_macro) + "," + str(fscore_macro))

        if not os.path.exists(outdir):
            os.mkdir(outdir)

        # Save Predictions:
        np.savetxt(outdir + "preds_" + str(m) + ".csv", preds, delimiter=",", fmt="%.0f")

        # Generate Visualizations
        predNames = [str(m) + "_" + str(feature_setting)]
        predData_all = [preds_folds_np]
        finalMetrics_all = visualizations.pRFSbyPredictor(labels_folds_np, predNames, predData_all)

        aMTest = finalMetrics_all
        aMTestMMW = aMTest[['micro', 'macro', 'weighted']]
        aMTestCFM = aMTest['ConfMatrix']

        visualizations.genCFMPlotfromPred(labels_folds_np, predNames, predData_all, 'testCFM', outdir)
        visualizations.genBars(aMTestMMW, 'aMTestMMW', outdir)
