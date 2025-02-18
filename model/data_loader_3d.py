# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json


class VideoData(Dataset):
    def __init__(self, mode, video_type, split_index):
        """ Custom Dataset class wrapper for loading the frame features and ground truth importance scores.

        :param str mode: The mode of the model, train or test.
        :param str video_type: The Dataset being used, SumMe or TVSum.
        :param int split_index: The index of the Dataset split being used.
        """
        self.mode = mode
        self.name = video_type.lower()
        self.datasets = ['.../vivit_summe_all.h5',
                         '.../data/TVSum/tvsum.h5',
                         '.../data/SumMe/eccv16_dataset_summe_google_pool5.h5',
                         '.../data/TVSum/eccv16_dataset_tvsum_google_pool5.h5']
        self.datasets_cap = ['.../data/SumMe/summe_cap_roberta_amt.h5',
                             '.../TVSum/tvsum_cap_roberta_amt.h5']
        self.splits_filename = ['.../Highlight_detection/data/splits/' + self.name + '_splits.json']
        self.split_index = split_index  # it represents the current split (varies from 0 to 4)

        if 'summe' in self.splits_filename[0]:
            self.filename = self.datasets[0]
            self.filename_cap = self.datasets_cap[0]
            self.filename_gt = self.datasets[2]
        elif 'tvsum' in self.splits_filename[0]:
            self.filename = self.datasets[1]
            self.filename_cap = self.datasets_cap[1]
            self.filename_gt = self.datasets[3]
        hdf = h5py.File(self.filename, 'r')
        hdf_cap = h5py.File(self.filename_cap, 'r')
        hdf_gt = h5py.File(self.filename_gt, 'r')
        self.list_frame_features, self.list_frame_features_cap, self.list_gtscores = [], [], []

        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split
                    break

        for video_name in self.split[self.mode + '_keys']:
            video_name = video_name[-11:]
            frame_features = torch.Tensor(np.array(hdf[video_name + '/features']))
            self.list_frame_features.append(frame_features)

        hdf.close()
        hdf_cap.close()
        hdf_gt.close()

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        self.len = len(self.split[self.mode+'_keys'])
        return self.len

    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset.
        train mode returns: frame_features and gtscores
        test mode returns: frame_features and video name

        :param int index: The above-mentioned id of the data.
        """

        frame_features = self.list_frame_features[index]
        # frame_features_cap = self.list_frame_features_cap[index]

        if self.mode == 'test':
            video_name = self.split[self.mode + '_keys'][index]
            video_name = video_name[-11:]
            # return frame_features, frame_features_cap, video_name
            return frame_features, video_name
        else:
            # gtscore = self.list_gtscores[index]
            return frame_features  # , frame_features_cap, gtscore


def get_loader(mode, video_type, split_index):
    """ Loads the `data.Dataset` of the `split_index` split for the `video_type` Dataset.
    Wrapped by a Dataloader, shuffled and `batch_size` = 1 in train `mode`.

    :param str mode: The mode of the model, train or test.
    :param str video_type: The Dataset being used, SumMe or TVSum.
    :param int split_index: The index of the Dataset split being used.
    :return: The Dataset used in each mode.
    """
    if mode.lower() == 'train':
        vd = VideoData(mode, video_type, split_index)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return VideoData(mode, video_type, split_index)


if __name__ == '__main__':
    pass
