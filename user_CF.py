import torch 
import numpy as np 
import time 

start = time.time()


train_data = np.load('parsed_data/train_data.npy')
test_data = np.load('parsed_data/test_data.npy')
num_all_reviewer = train_data.shape[0]
num_all_product = train_data.shape[1]

user_share_items = {}
for i in range(num_all_reviewer):
    for j in range(num_all_reviewer):
        if i!=j:
            user_share_items[i,j]=None

user_correlations = {}
for i in range(num_all_reviewer):
    for j in range(num_all_reviewer):
        if i!=j:
            user_correlations[i,j]=None


for i in range(num_all_reviewer):
    for j in range(num_all_product):
        if i!=j:
            items_i = np.where(train_data[i]>0)[0]
            items_j = np.where(train_data[j]>0)[0]
            share_items = set(items_i).intersection(set(items_j))
            if len(share_items)>0:
                user_share_items[i,j] = list(share_items)

user_bias = np.zeros(num_all_reviewer)
normalized_train_data = np.zeros((num_all_reviewer,num_all_product))
for i in range(num_all_reviewer):
    valid = []
    for j in range(num_all_product):
        if train_data[i][j]>0:
            valid.append(train_data[i][j])
    mean_valid = np.mean(valid)
    user_bias[i]=mean_valid
    for j in range(num_all_product):
        if train_data[i][j]>0:
            normalized_train_data[i,j]=train_data[i][j]-mean_valid


def correlation(user1,user2):
    share_items = user_share_items[user1,user2]
    Term1 = sum(  normalized_train_data[user1,item]* normalized_train_data[user2,item] for item in share_items)
    Term2 = np.sqrt(sum([pow(normalized_train_data[user1,item], 2) for item in share_items])) * np.sqrt(sum([pow(normalized_train_data[user2,item], 2) for item in share_items]))
    if Term2 == 0:
        return 0
    else:
        return Term1/Term2


def validate():
    count = 0
    error = 0
    error_rmse = 0
    for i in range(num_all_reviewer):
        for j in range(num_all_product):
            if test_data[i,j]>0:
                count+=1
                users_for_j = np.where(train_data[:,j]>0)[0]
                users_for_j_has_i = []
                for jj in users_for_j:
                    if user_share_items[i,jj]!=None:
                        users_for_j_has_i.append(jj)
                if len(users_for_j_has_i)==0:
                    predict = user_bias[i]
                else:
                    correlation_i_users = []
                    for jj in users_for_j_has_i:
                        if user_correlations[i,jj]==None:
                            user_correlations[i,jj] = correlation(i,jj)
                            correlation_i_users.append(user_correlations[i,jj])
                        else:
                            correlation_i_users.append(user_correlations[i,jj])
                    score = 0
                    weights = 0
                    for perid in range(len(correlation_i_users)):
                        weight = correlation_i_users[perid]
                        weights += weight
                        score+= weight*normalized_train_data[users_for_j_has_i[perid],j]
                    if weights == 0:
                        predict = user_bias[i]
                    else:
                        predict = user_bias[i] + score/weights                        
                    if predict<1:
                        predict = 1
                    if predict>5:
                        predict = 5
                error+= abs(predict - test_data[i,j])
                error_rmse+=np.square(predict - test_data[i,j])
    print('mae is',error/count)
    print('rmse is',np.sqrt(error_rmse/count))

validate()


print('training data using',time.time()-start,'seconds')



