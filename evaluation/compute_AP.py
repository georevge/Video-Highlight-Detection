# -*- coding: utf-8 -*-
from os import listdir
import json
import numpy as np
import h5py
from evaluation_metrics import evaluate_summary
from generate_summary import generate_summary
import argparse
from statistics import mean


# arguments to run the script
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str,
                    default='/home/gevge/Downloads/Highlight_detection/Summaries/exp4/TVSum/results/split0',
                    help="Path to the json files with the scores of the frames for each epoch")
parser.add_argument("--dataset", type=str, default='TVSum', help="Dataset to be used")
parser.add_argument("--eval", type=str, default="avg", help="Eval method to be used for f_score reduction (max or avg)")

args = vars(parser.parse_args())
path = args["path"]
dataset = args["dataset"]
eval_method = args["eval"]

results = [f for f in listdir(path) if f.endswith(".json")]
results.sort(key=lambda video: int(video[6:-5]))
dataset_path = '/home/gevge/Downloads/Highlight_detection/data/' + dataset + '/eccv16_dataset_' + dataset.lower() + '_google_pool5.h5'
gtscores_path = '/home/gevge/Downloads/Highlight_detection/data/TVSum/tvsum.h5'


# Function returns N largest elements
def Nmaxelements(list1, N):
    final_list = []

    for i in range(0, N):
        max1 = 0

        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j]

        list1.remove(max1)
        final_list.append(max1)

    return final_list


def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in pred_scores]
        max_len = max(len(y_true), len(y_pred))
        S = np.zeros(max_len, dtype=int)
        G = np.zeros(max_len, dtype=int)
        S[:len(y_pred)] = y_pred
        G[:len(y_true)] = y_true
        overlapped = S & G

        # Compute precision, recall
        precision = sum(overlapped) / sum(S)
        recall = sum(overlapped) / sum(G)

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls

def get_ap_at_k(y_true, y_predict, k=5):
    '''
    Average precision in different formats: (non-) interpolated and/or 11-point approximated
    point_11=True and interpolate=True corresponds to the 11-point interpolated AP used in
    the PASCAL VOC challenge up to the 2008 edition and has been verfied against the vlfeat implementation
    The exact average precision (interpolate=False, point_11=False) corresponds to the one of vl_feat
    :param y_true: list/ numpy vector of true labels in {0,1} for each element
    :param y_predict: predicted score for each element
    :param interpolate: Use interpolation?
    :param point_11: Use 11-point approximation to average precision?
    :return: average precision
    '''

    # Check inputs
    assert len(y_true)==len(y_predict), "Prediction and ground truth need to be of the same length"
    if len(set(y_true))==1:
        if y_true[0]==0:
            raise ValueError('True labels cannot all be zero')
        else:
            return 1
    else:
        assert sorted(set(y_true))==[0,1], "Ground truth can only contain elements {0,1}"


    sortIdx = np.argsort(-y_predict)
    y_predict = y_predict[sortIdx]
    y_true = y_true[sortIdx]
    # pdb.set_trace()

    # 选择topk

    y_true = y_true[:k]
    n_good = y_true.sum()
    if n_good == 0:
        return 0
    ap = 0.0
    intersect_size = 0.0
    old_recall = 0.0
    old_precision = 1.0
    for j in range(k):
        if y_true[j] == 1:
            intersect_size += 1
        recall = intersect_size / n_good
        precision = intersect_size / (j + 1)
        ap = ap + (recall - old_recall) * ((old_precision + precision) / 2)
        old_recall = recall
        old_precision = precision

    return ap

def get_ap_at_k_correct(y_true, y_predict, k=5):
    # Check inputs
    assert len(y_true)==len(y_predict), "Prediction and ground truth need to be of the same length"
    if len(set(y_true))==1:
        if y_true[0]==0:
            raise ValueError('True labels cannot all be zero')
        else:
            return 1
    else:
        assert sorted(set(y_true))==[0,1], "Ground truth can only contain elements {0,1}"


    sortIdx = np.argsort(-y_predict)
    y_predict = y_predict[sortIdx]
    y_true = y_true[sortIdx]
    # pdb.set_trace()

    # 选择topk

    y_true = y_true[:k]
    n_good = y_true.sum()
    if n_good == 0:
        return 0
    ap = 0.0
    intersect_size = 0.0
    for j in range(k):
        if y_true[j] == 1:
            intersect_size += 1
        precision = intersect_size / (j + 1)
        precision = precision * y_true[j]
        ap = ap + (1/k) * precision

    return ap

mAP_epochs = []
for epoch in results:                       # for each epoch ...
    all_scores = []
    with open(path + '/' + epoch) as f:     # read the json file ...
        data = json.loads(f.read())
        keys = list(data.keys())

        for video_name in keys:             # for each video inside that json file ...
            video_name = video_name[-11:]
            scores = np.asarray(data[video_name]).squeeze(0)  # read the importance scores from frames
            all_scores.append(scores)
    all_gtscores = []
    with h5py.File(gtscores_path, 'r') as hdf:
        for video_name in keys:
            video_name = video_name[-11:]
            eval_gtscore = np.array(hdf.get(video_name + '/eval_gtscore'))
            all_gtscores.append(eval_gtscore)
    list_AP = []
    for video in range(len(all_gtscores)):
        y_true = all_gtscores[video]  # .tolist()
        pred_scores = all_scores[video]  # .tolist()
        # thresholds = np.arange(start=0.05, stop=1.05, step=0.01)
        # precisions, recalls = precision_recall_curve(y_true=y_true, pred_scores=pred_scores, thresholds=thresholds)
        # precisions.append(1)
        # recalls.append(0)

        # precisions = np.array(precisions)
        # recalls = np.array(recalls)

        # AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
        AP = np.mean([get_ap_at_k(gt, pred_scores, k=5) for gt in y_true])
        list_AP.append(AP)
    meanAP = mean(list_AP)
    mAP_epochs.append(meanAP)
    print("mAP: ", meanAP)

with open(path + '/mAP_scores.txt', 'w') as outfile:
    for mAP_score in mAP_epochs:
        outfile.write('%s\n' % mAP_score)
