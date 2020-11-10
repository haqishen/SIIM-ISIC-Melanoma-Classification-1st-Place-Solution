## SIIM-ISIC-Melanoma-Classification-1st-Place-Solution

Competition Leaderboard: https://www.kaggle.com/c/siim-isic-melanoma-classification/leaderboard 

### SOFTWARE (python packages are detailed separately in `requirements.txt`)

Python 3.6.9

CUDA Version 10.2.89

cuddn 7.6.5

nvidia Driver Version: 418.116.00


### DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
Download and unzip resized 2020 and 2019 data (including 2018) of 3 sizes by Chris Deotte.

```
mkdir ./data
cd ./data
for input_size in 512 768 1024
do
  kaggle datasets download -d cdeotte/jpeg-isic2019-${input_size}x${input_size}
  kaggle datasets download -d cdeotte/jpeg-melanoma-${input_size}x${input_size}
  unzip -q jpeg-melanoma-${input_size}x${input_size}.zip -d jpeg-melanoma-${input_size}x${input_size}
  unzip -q jpeg-isic2019-${input_size}x${input_size}.zip -d jpeg-isic2019-${input_size}x${input_size}
  rm jpeg-melanoma-${input_size}x${input_size}.zip jpeg-isic2019-${input_size}x${input_size}.zip
done
```

### Model Illustration

![](https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/blob/master/figure1.png)

More details can be found here: 

https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175412

http://arxiv.org/abs/2010.05351
 
### Training

Training commands of the 18 models. Training time for a single model ranges from 15 to 45 hours for all 5 folds in our setup.

After training, models will be saved in `./weights/` Tranning logs will be saved in `./logs/`

```
python train.py --kernel-type 9c_meta_b3_768_512_ext_18ep --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type efficientnet_b3 --use-meta --n-epochs 18 --use-amp --CUDA_VISIBLE_DEVICES 0,1

python train.py --kernel-type 9c_b4ns_2e_896_ext_15ep --data-dir ./data/ --data-folder 1024 --image-size 896 --enet-type tf_efficientnet_b4_ns --use-amp --init-lr 2e-5 --CUDA_VISIBLE_DEVICES 0,1,2,3,4,5

python train.py --kernel-type 9c_b4ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0

python train.py --kernel-type 9c_b4ns_768_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1,2

python train.py --kernel-type 9c_b4ns_768_768_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 768 --enet-type tf_efficientnet_b4_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1,2

python train.py --kernel-type 9c_meta_b4ns_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b4_ns --use-meta --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3

python train.py --kernel-type 4c_b5ns_1.5e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b5_ns --out-dim 4 --init-lr 1.5e-5 --use-amp --CUDA_VISIBLE_DEVICES 0,1,2

python train.py --kernel-type 9c_b5ns_1.5e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b5_ns --init-lr 1.5e-5 --use-amp --CUDA_VISIBLE_DEVICES 0,1,2

python train.py --kernel-type 9c_b5ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b5_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1

python train.py --kernel-type 9c_meta128_32_b5ns_384_ext_15ep --data-dir ./data/ --data-folder 512 --image-size 384 --enet-type tf_efficientnet_b5_ns --use-meta --n-meta-dim 128,32 --use-amp --CUDA_VISIBLE_DEVICES 0

python train.py --kernel-type 9c_b6ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b6_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1

python train.py --kernel-type 9c_b6ns_576_ext_15ep_oldfold --data-dir ./data/ --data-folder 768 --image-size 576 --enet-type tf_efficientnet_b6_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3

python train.py --kernel-type 9c_b6ns_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b6_ns --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3

python train.py --kernel-type 9c_b7ns_1e_576_ext_15ep_oldfold --data-dir ./data/ --data-folder 768 --image-size 576 --enet-type tf_efficientnet_b7_ns --init-lr 1e-5 --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3

python train.py --kernel-type 9c_b7ns_1e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b7_ns --init-lr 1e-5 --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3,4,5,6,7

python train.py --kernel-type 9c_meta_1.5e-5_b7ns_384_ext_15ep --data-dir ./data/ --data-folder 512 --image-size 384 --enet-type tf_efficientnet_b7_ns --use-meta --use-amp --CUDA_VISIBLE_DEVICES 0,1,2

python train.py --kernel-type 9c_nest101_2e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type resnest101 --init-lr 2e-5 --use-amp --CUDA_VISIBLE_DEVICES 0,1,2,3

python train.py --kernel-type 9c_se_x101_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type seresnext101 --use-amp --CUDA_VISIBLE_DEVICES 0,1
```

### (Optional) Evaluating

Optionally, you can evaluate each model on 5 fold cross valiation sets. You can either use the models trained in previous step, or use the trained models we provided and specify the directory in `--model-dir`.

Evaluation results will be printed out and saved to `./logs/` Out-of-folds prediction results will be saved to `./oofs/`


```
python evaluate.py --kernel-type 9c_meta_b3_768_512_ext_18ep --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type efficientnet_b3 --use-meta

python evaluate.py --kernel-type 9c_b4ns_2e_896_ext_15ep --data-dir ./data/ --data-folder 1024 --image-size 896 --enet-type tf_efficientnet_b4_ns

python evaluate.py --kernel-type 9c_b4ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b4_ns

python evaluate.py --kernel-type 9c_b4ns_768_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b4_ns

python evaluate.py --kernel-type 9c_b4ns_768_768_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 768 --enet-type tf_efficientnet_b4_ns

python evaluate.py --kernel-type 9c_meta_b4ns_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b4_ns --use-meta

python evaluate.py --kernel-type 4c_b5ns_1.5e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b5_ns --out-dim 4

python evaluate.py --kernel-type 9c_b5ns_1.5e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b5_ns

python evaluate.py --kernel-type 9c_b5ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b5_ns

python evaluate.py --kernel-type 9c_meta128_32_b5ns_384_ext_15ep --data-dir ./data/ --data-folder 512 --image-size 384 --enet-type tf_efficientnet_b5_ns --use-meta --n-meta-dim 128,32

python evaluate.py --kernel-type 9c_b6ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b6_ns

python evaluate.py --kernel-type 9c_b6ns_576_ext_15ep_oldfold --data-dir ./data/ --data-folder 768 --image-size 576 --enet-type tf_efficientnet_b6_ns

python evaluate.py --kernel-type 9c_b6ns_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b6_ns

python evaluate.py --kernel-type 9c_b7ns_1e_576_ext_15ep_oldfold --data-dir ./data/ --data-folder 768 --image-size 576 --enet-type tf_efficientnet_b7_ns

python evaluate.py --kernel-type 9c_b7ns_1e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b7_ns

python evaluate.py --kernel-type 9c_meta_1.5e-5_b7ns_384_ext_15ep --data-dir ./data/ --data-folder 512 --image-size 384 --enet-type tf_efficientnet_b7_ns --use-meta

python evaluate.py --kernel-type 9c_nest101_2e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type resnest101

python evaluate.py --kernel-type 9c_se_x101_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type seresnext101
```

### Predicting

Make predictions on test set. You can either use the models trained in the Training step, or use the trained models we provided and specify the directory in `--model-dir`

Each models submission file will be saved to `./subs/`

```
python predict.py --kernel-type 9c_meta_b3_768_512_ext_18ep --data-dir ./data/ --data-folder 768 --image-size 512 --enet-type efficientnet_b3 --use-meta

python predict.py --kernel-type 9c_b4ns_2e_896_ext_15ep --data-dir ./data/ --data-folder 1024 --image-size 896 --enet-type tf_efficientnet_b4_ns

python predict.py --kernel-type 9c_b4ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b4_ns

python predict.py --kernel-type 9c_b4ns_768_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b4_ns

python predict.py --kernel-type 9c_b4ns_768_768_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 768 --enet-type tf_efficientnet_b4_ns

python predict.py --kernel-type 9c_meta_b4ns_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b4_ns --use-meta

python predict.py --kernel-type 4c_b5ns_1.5e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b5_ns --out-dim 4

python predict.py --kernel-type 9c_b5ns_1.5e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b5_ns

python predict.py --kernel-type 9c_b5ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b5_ns

python predict.py --kernel-type 9c_meta128_32_b5ns_384_ext_15ep --data-dir ./data/ --data-folder 512 --image-size 384 --enet-type tf_efficientnet_b5_ns --use-meta --n-meta-dim 128,32

python predict.py --kernel-type 9c_b6ns_448_ext_15ep-newfold --data-dir ./data/ --data-folder 512 --image-size 448 --enet-type tf_efficientnet_b6_ns

python predict.py --kernel-type 9c_b6ns_576_ext_15ep_oldfold --data-dir ./data/ --data-folder 768 --image-size 576 --enet-type tf_efficientnet_b6_ns

python predict.py --kernel-type 9c_b6ns_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b6_ns

python predict.py --kernel-type 9c_b7ns_1e_576_ext_15ep_oldfold --data-dir ./data/ --data-folder 768 --image-size 576 --enet-type tf_efficientnet_b7_ns

python predict.py --kernel-type 9c_b7ns_1e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type tf_efficientnet_b7_ns

python predict.py --kernel-type 9c_meta_1.5e-5_b7ns_384_ext_15ep --data-dir ./data/ --data-folder 512 --image-size 384 --enet-type tf_efficientnet_b7_ns --use-meta

python predict.py --kernel-type 9c_nest101_2e_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type resnest101

python predict.py --kernel-type 9c_se_x101_640_ext_15ep --data-dir ./data/ --data-folder 768 --image-size 640 --enet-type seresnext101
```

### Ensembling

Ensemble the 18 single model's submission files (from previous step) into the final submission file.

The submission file `final_sub1.csv` will be saved in root directory.

```
python ensemble.py
```

### Trained Weights

We published our trained weigths of the model settings above (which we won this competition):

https://www.kaggle.com/boliu0/melanoma-winning-models

Download it into `./weights/` then you can run `evaluate.py` and `predict.py` directly.

