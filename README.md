# Hierarchical-Similarity-Learning-for-Language-based-Product-Image-Retrieval

## Network
HSL
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
```
|-- data
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
    |-- valid
        |-- valid.tsv
        |-- valid_answer.json
    |-- train
        |-- train.tsv
|-- code
    |-- data.py
    |-- main.py
    ...
    |-- vocab.py
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

## Citing
