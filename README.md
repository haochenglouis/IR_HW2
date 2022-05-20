
# Recommendation Systems

This repository is the implementation of the recommendation system for Homework 2 in the Information Retrieval Class (CSE272 Spring UCSC). All the code are written in python.

## Prerequisites

Python 3

[pytorch](https://pytorch.org) (pip install torch)




## Guideline


### Run data parser to generate training data and test data

Run command below:

```
python data_processor.py
```

This command will split the dataset into training data and test data and store the processed data in **parsed_data/**. 

### Run each algorithm to get MAE and RMSE evaluation:

In the repository, there are "user_CF.py","MF_train_eval.py" and "MF_train_eval_cuda.py". "user_CF.py" indicates user-based collaborative filtering, "MF_train_eval.py" indicates matrix factorization. "MF_train_eval_cuda.py" indicates matrix factorization implementation based on CUDA. One can simply run each algorithm via python command. For example, for user-based collaborative filtering, just run the command below to see evaluation of user_CF:

```
python user_CF.py
```

### Plot Training dynamics

For matrix factorization method, one can also view the training dynamics of MAE and RMSE. Simply run: 

```
CUDA_VISIBLE_DEVICES=0 nohup python MF_train_eval_cuda.py > results/MF_cuda.out 2>&1 &.
```

Then run:

```
python plot_results.py
```

In the results/, one can see the training dynamics in "mae_rmse.pdf".
