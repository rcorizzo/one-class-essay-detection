import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# Combining all metrics in the right format for visualizations

def pRFSbyPredictor(groundTruth, predictorNames, predictorData):
    measures = ['macro', 'micro', 'weighted']

    pRFS_precision = []
    pRFS_recall = []
    pRFS_f1 = []

    measureArr = [[], [], []]

    cMatrixArr = []

    for i in range(len(predictorNames)):
        #print(predictorNames[i])
        for m in range(len(measures)):
            pRFS_precision = []
            pRFS_recall = []
            pRFS_f1 = []
            tempVar = precision_recall_fscore_support(groundTruth, predictorData[i], average=measures[m])
            #print(tempVar)
            pRFS_precision.append(round(tempVar[0], 3))
            pRFS_recall.append(round(tempVar[1], 3))
            pRFS_f1.append(round(tempVar[2], 3))

            measureArr[m].append([pRFS_precision[0], pRFS_recall[0], pRFS_f1[0]])
        cMatrixArr.append(confusion_matrix(groundTruth, predictorData[i]))

    # Precision, Recall, F1, (support)
    pRFS_Df = pd.DataFrame(index=predictorNames)
    pRFS_Df['macro'] = measureArr[0]
    pRFS_Df['micro'] = measureArr[1]
    pRFS_Df['weighted'] = measureArr[2]
    pRFS_Df['ConfMatrix'] = cMatrixArr

    return pRFS_Df



# Confusion Matrix 

def genCFMPlotfromPred(groundTruth,predLabels,predData,fileLabel, path):
    # print(type(groundTruth))
    # print(groundTruth)
    # print(type(predLabels))
    # print(predLabels)
    # print(type(predData))
    # print(predData)
    for i in range(len(predLabels)):
        ConfusionMatrixDisplay.from_predictions(groundTruth, predData[i], cmap='plasma')

        if '(' in predLabels[i]:
            cleanLabel = predLabels[i].split("(")[0]
        elif '_' in predLabels[i]:
            cleanLabel = predLabels[i].split("_")[0]
        else:
            cleanLabel = predLabels[i]

        plt.title(cleanLabel)
        plt.savefig(path + fileLabel + "_" + str(predLabels[i]) + '.pdf', format='pdf')

#genCFMPlotfromPred(real_values,predNames,predData_all,'testCFM')

# Bar Plot

def genBars(data, fileLabel, path):
    labels = data.index

    for i in range(len(labels)):
        y1= data.loc[labels[i]][0]
        y2= data.loc[labels[i]][1]
        y3= data.loc[labels[i]][2]
        width = 0.2
        x = np.arange(3)
        plt.clf()
        plt.bar(x-0.2, y1, width, color='slategrey')
        plt.bar(x, y2, width, color='lightsteelblue')
        plt.bar(x+0.2, y3, width, color='lavender')

        if '(' in labels[i]:
            cleanLabel = labels[i].split("(")[0]
        elif '_' in labels[i]:
            cleanLabel = labels[i].split("_")[0]
        else:
            cleanLabel = labels[i]

        plt.xlabel(cleanLabel)

        plt.xticks(x, ['Micro', 'Macro', 'Weighted'])
        plt.legend(["Precision", "Recall", "F1"],loc='lower left')
        plt.ylabel("Scores")
        plt.ylim(0.0, 1.0)
        plt.savefig(path + fileLabel + "_" + str(labels[i]) + '.pdf', format='pdf')

