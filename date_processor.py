import torch
import numpy as np
import json
import glob
import random
import time
random.seed(0)

start = time.time()

file=glob.glob('data/Musical_Instruments_5.json')
review=[]
with open(file[0]) as data_file:
    data=data_file.read()
    for i in data.split('\n'):
        review.append(i)
all_data = []
for i in range(len(review)-1):
    jd = json.loads(review[i])
    all_data.append([jd['reviewerID'],jd['asin'],float(jd['overall'])])


all_reviewer = set()
all_product = set()
for i in all_data:
    all_reviewer.add(i[0])
    all_product.add(i[1])
all_reviewer = list(all_reviewer)
all_product = list(all_product)
num_all_reviewer = len(all_reviewer)
num_all_product = len(all_product)

id2rev = {}
rev2id = {}
id2pro = {}
pro2id = {}
for i,j in enumerate(all_reviewer):
    id2rev[i]=j
    rev2id[j]=i
for i,j in enumerate(all_product):
    id2pro[i]=j
    pro2id[j]=i    
for i in range(len(all_reviewer)):
    all_reviewer[i] = rev2id[all_reviewer[i]]
for i in range(len(all_product)):
    all_product[i] = pro2id[all_product[i]]
for i in range(len(all_data)):
    old = all_data[i]
    new = [rev2id[old[0]],pro2id[old[1]],old[2]]
    all_data[i] = new



train_data = []
test_data = []
for i in all_reviewer:
    gather_i = []
    for j in all_data:
        if j[0]==i:
            gather_i.append(j)
    random.shuffle(gather_i)
    select_num = int(len(gather_i)*0.8)
    train_data.extend(gather_i[:select_num])
    test_data.extend(gather_i[select_num:])

train_data_matrix = np.zeros((num_all_reviewer,num_all_product))
test_data_matrix = np.zeros((num_all_reviewer,num_all_product))
for i in train_data:
    train_data_matrix[i[0],i[1]]=i[2]
for i in test_data:
    test_data_matrix[i[0],i[1]]=i[2]



np.save('parsed_data/train_data.npy',train_data_matrix)
np.save('parsed_data/test_data.npy',test_data_matrix)

print('processing data using',time.time()-start,'seconds')


