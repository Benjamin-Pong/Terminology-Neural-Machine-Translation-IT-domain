#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate awesome_align
LANG=ru
HOME=''path_to_your_directory''

python lm.py \
    --lm_train "$HOME/lm.$LANG.train.txt" \
    --indomaindatade "$HOME/synthetic_data/ende_syn.json" \
    --indomaindataes "$HOME/synthetic_data/enes_syn.json" \
    --indomaindataru "$HOME/synthetic_data/enru_syn.json" \
    --model_type Laplace \
    --n 4 \
    --outmodel "$HOME/outlm.train.$LANG.pkl" \
    --inmodel "$HOME/inlm.train.$LANG.pkl" \
