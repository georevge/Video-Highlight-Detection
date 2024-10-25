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
                    default='/home/gevge/Downloads/Highlight_detection/Summaries/exp1/TVSum/results/split3',
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
gtscores_path = '/home/gevge/Downloads/Highlight_detection/data/TVSum/vivit_tvsum_highlight_gtscores.h5'

f_score_epochs = []
for epoch in results:                       # for each epoch ...
    all_scores = []
    with open(path + '/' + epoch) as f:     # read the json file ...
        data = json.loads(f.read())
        keys = list(data.keys())

        for video_name in keys:             # for each video inside that json file ...
            scores = np.asarray(data[video_name])  # read the importance scores from frames
            all_scores.append(scores)


    all_user_summary, all_shot_bound, all_nframes, all_positions, all_number_of_shots = [], [], [], [], []
    all_gtscores = []
    with h5py.File(gtscores_path, 'r') as hdf:
        for video_name in keys:
            video_index = video_name[6:]
            all_gtscores_per_video = []
            gt_paths = np.array(hdf.get('video_' + video_index + '/gtscores'))
            n = len(gt_paths)
            for num_of_clip in range(1, n + 1):
                gt_per_clip = np.array(hdf[video_name + '/gtscores' + '/clip_' + str(num_of_clip)])
                all_gtscores_per_video.append(gt_per_clip)
            all_gtscores_per_video = np.array(all_gtscores_per_video)
            all_gtscores.append(all_gtscores_per_video)



    all_f_scores = []
    # compare the resulting summary with the ground truth one, for each video
    for video_index in range(len(all_summaries)):
        summary = all_summaries[video_index]
        user_summary = all_user_summary[video_index]
        f_score = evaluate_summary(summary, user_summary, eval_method)
        all_f_scores.append(f_score)

    f_score_epochs.append(np.mean(all_f_scores))
    print("f_score: ", np.mean(all_f_scores))

# Save the importance scores in txt format.
with open(path + '/f_scores.txt', 'w') as outfile:
    for f_score in f_score_epochs:
        outfile.write('%s\n' % f_score)
