LC_NUMERIC="en_US.UTF-8"
for i in $(seq 0 4); do
  python model/main.py --split_index "$i" --n_epochs 300 --batch_size 20 --video_type 'SumMe'
  python evaluation/compute_fscores.py --path ".../SumMe/results/split$i" --dataset SumMe --eval max
done
