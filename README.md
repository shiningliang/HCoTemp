# HCoTemp
Code for the paper Dynamic Item Feature Modelling for Rating Prediction in Recommender Systems

## Data Preprocess
Download the Amazon Review dataset from [this link](https://nijianmo.github.io/amazon/)
### Static
nohup python rec_main.py --model Static_ID --prepare --epochs 5 --task AM_Games >prep_games_si.txt 2>&1 &

The output path will be data/processed_data/AM_Pets_static
### Dynamic
nohup python rec_main.py --model Dynamic_ID --prepare --epochs 5 --task AM_Pets --dynamic >prep_pets_di.txt 2>&1 &

The output path will be data/processed_data/AM_Pets_dynamic
### Period
nohup python rec_main.py --model Dynamic_COTEMP_GRAPH --prepare --epochs 5 --task AM_CD --period --P 6 >prep_cd_dcg_p6.txt 2>&1 &

The output path will be data/processed_data/AM_Pets_period

### Parameters
--task dataset name, i.e., AM_**

--model processed data format (same as the model name)，including: Static_ID, Static_COTEMP, Dynamic_ID, and Dynamic_COTEMP

--dynamic use Dynamic_** models

--P when you add --dynamic, user/item embedding is 36 months, i.e., T=36，if you use --P it's parameter, X, indicated the length of period for empress embeddings, i.e., T=T/X

## Training&Testing
### Static
nohup python -m torch.distributed.launch --nproc_per_node=2 rec_run.py --model Static_COTEMP --train --test --epochs 3 --task AM_Pets --batch_train 8 --batch_eval 8 --lr 1e-4 --weight_decay 3e-3 >train_pets_static_b8lr1e-4wd3e-3.txt 2>&1 &

### Dynamic
nohup python -m torch.distributed.launch --nproc_per_node=1 rec_run.py --model Dynamic_COTEMP --train --test --epochs 3 --task AM_CD --batch_train 32 --batch_eval 32 --lr 1e-4 --weight_decay 3e-3 --period --P 6 >train_cd_p6_b32n1lr1e-4wd3e-3.txt 2>&1 &

### Init item nodem embeddings
nohup python -m torch.distributed.launch --nproc_per_node=2 rec_run.py --model Static_ID_GRAPH --train --test --epochs 3 --task AM_Pets --batch_train 8 --batch_eval 8 --lr 1e-4 --weight_decay 3e-3

### HCoTemp
(Below is for Period, others are similar)

nohup python rec_run_hg.py --train --test --period --P 6 --lr 0.0001 --epochs 3 --han_out_size 64 >train_games_hgdc_p6_b32n1lr1e-4wd3e-4ep3.txt 2>&1 &

## Requirements
pytorch-transformers=1.2

pytorch-pretrained-bert=0.6.2

pytorch>=1.2

nvidia amp
