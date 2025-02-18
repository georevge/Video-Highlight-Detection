LC_NUMERIC="en_US.UTF-8"
for i in $(seq 0 4); do
  python model/main.py --split_index "$i" --n_epochs 300 --batch_size 40 --video_type 'TVSum'
  python evaluation/compute_fscores.py --path ".../TVSum/results/split$i" --dataset TVSum --eval avg
done
