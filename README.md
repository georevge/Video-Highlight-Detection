# A code implementation for the paper "Contrastive Learning for Unsupervised Video Highlight Detection"

To train the model using one of the aforementioned datasets and for a number of randomly created splits of the dataset use the corresponding JSON file that is included in the data/splits directory. This file contains the 5 randomly-generated splits that were utilized in our experiments.

For training the model using a single split, run:
```
python model/main.py --split_index N --n_epochs E --batch_size B --video_type 'dataset_name'
```
where, N refers to the index of the used data split, E refers to the number of training epochs, B refers to the batch size, and dataset_name refers to the name of the used dataset.

To train and evaluate the model for all 5 splits, use the run_summe_splits.sh or run_tvsum_splits.sh script

## Data

Structured h5 files with the video features and annotations of the SumMe and TVSum datasets are available within the data folder. The GoogleNet features of the video frames were extracted by Ke Zhang and Wei-Lun Chao and the h5 files were obtained from Kaiyang Zhou.

The ViViT features have been extracted and, also, located in data folder (zip files) along with BERT, RoBERTa and XLNet text features for each video.
