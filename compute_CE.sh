#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate awesome_align
HOME=''path_to_your_directory''
LANG=es
python compute_CE.py \
    --new_data "$HOME/ced.$LANG.data.txt" \
    --testdata "$HOME/lm.$LANG.test.txt" \
    --outmodel "$HOME/outlm.train.$LANG.pkl" \
    --inmodel "$HOME/inlm.train.$LANG.pkl" \