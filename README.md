# Hierarchical-Similarity-Learning-for-Language-based-Product-Image-Retrieval

## Network
HSL
![Image](https://github.com/liufh1/HSL/blob/main/img/pipelinebig.jpg)

## Requirements

### Environments
PyTorch 1.7.1  
CUDA 10.2.89  
Python 3.7  

We use anaconda to create our experimental environment. You can rebuild it by the following commands.


```
conda create -n {your_env_name} python=3.7  
conda activate {your_env_name}  
pip install -r requirements.txt  
...
conda deactivate 
```


### Download Data
#### Dataset  
We use the dataset of the [KDD Cup 2020 Challenges for Modern E-Commerce Platform: Multimodalities Recall](https://tianchi.aliyun.com/competition/entrance/231786/information). You can download it in this website. You need to download multimodal_train.zip and multimodal_valid.zip and then unzip them.  
You can also download pretrained word vectors model "Glove" (http://nlp.stanford.edu/data/glove.42B.300d.zip).
```
|-- data
    |-- glove_model.bin
    |-- valid
        |-- valid.tsv
        |-- valid_answer.json
    |-- train
        |-- train.tsv
|-- code
    |-- parse_dataset.py
```
Then run 
```
python parse_dataset.py
```
After finishing it, the list are as follows.  
```
|-- data
    |-- glove_model.bin
    |-- kddcup_vocab.pkl
    |-- valid
        |-- valid.tsv
        |-- valid.features.npy
        |-- valid.ids.txt
        |-- valid.products.pkl
        |-- valid.queries.pkl
        |-- valid_answer.json
    |-- train
        |-- train.tsv
        |-- train.features.npy
        |-- train.ids.txt
        |-- train.products.pkl
        |-- train.queries.pkl
|-- code
    |-- data.py
    |-- main.py
    ...
    |-- utils.py
    |-- HSL
        |-- api.py
        |-- data.py
        ...
        |-- model.py
```
## Getting Started
After the pretreatment of data, then run
```
python main.py --logger_name runs/KddCup2020 --use_dup --no_txtnorm --glove --score_agg mean
```
to run the full model.  
  
There are also optional arguments for dataset, initial learning rate, batch size and so on. Check them by
```
python main.py --help
```
## Testing
As training terminates, the highest performance on validation set is saved for testing. But in this dataset, the organizer doesn't release the ground truth of the test set, so we can only test it on the validation set. You can load it and test on the validation set.
```
python main.py --eval_only --resume runs/{your_exp_name}/model_best.pth.tar
```
## Citing
```
@inproceedings{ma2021hsl,
title = {Hierarchical-Similarity-Learning-for-Language-based-Product-Image-Retrieval},
author = {Zhe Ma and Fenghao Liu and Jianfeng Dong and Xiaoye Qu and Yuan He and Shouling Ji},
booktitle = {International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
year = {2021},
}
```
