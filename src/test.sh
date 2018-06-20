#/bin/bash

python SentenceMatchDecoder.py --in_path /home/project/decomp-attn/data20180617/atec_3.0_test.csv --word_vec_path /home/project/decomp-attn/glove/vectors.txt --out_path result.json --model_prefix saved_model/SentenceMatch.atec
