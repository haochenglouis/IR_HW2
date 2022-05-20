import torch 
import numpy as np 
import time 

start = time.time()


train_data = np.load('parsed_data/train_data.npy')
test_data = np.load('parsed_data/test_data.npy')
num_all_reviewer = train_data.shape[0]
num_all_product = train_data.shape[1]
hidden_dim = 3
steps = 200


class Model(torch.nn.Module):
    def __init__(self, num_all_reviewer, num_all_product, hidden_dim=3):
        super().__init__()
        self.reviewer = torch.nn.Embedding(num_all_reviewer, hidden_dim)
        self.prodcut = torch.nn.Embedding(num_all_product, hidden_dim)

    def forward(self, reviewer_id, product_id):
        return sum(self.reviewer(reviewer_id) * self.prodcut(product_id))

mae_best = [10000.0]

def validate(model,test_data):
    count = 0
    error = 0
    error_rmse = 0
    for i in range(num_all_reviewer):
        for j in range(num_all_product):
            if test_data[i,j]>0:
                count+=1
                predict = model(torch.tensor(i).cuda(), torch.tensor(j).cuda()).detach().cpu().numpy()
                if predict<1:
                    predict = 1
                if predict>5:
                   	predict = 5
                error+= abs(predict - test_data[i,j])
                error_rmse+=np.square(predict - test_data[i,j])
    if error/count < mae_best[0]:
        mae_best[0] = error/count
        torch.save({'state_dict': model.state_dict()},'checkpoints/model_best.pth')
     
    print('mae is',error/count)
    print('rmse is',np.sqrt(error_rmse/count))





model = Model(num_all_reviewer, num_all_product, hidden_dim=hidden_dim)
model.cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay = 1e-4) 


train_data = torch.tensor(train_data).float().cuda()
for step in range(steps):
    model.train()
    print(step)
    loss_all = 0
    count = 0
    for i in range(num_all_reviewer):
        for j in range(num_all_product):
            if train_data[i,j]>0:
                count+=1
                output = model(torch.tensor(i).cuda(), torch.tensor(j).cuda())
                loss = criterion(output, train_data[i][j])
                loss_all += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    print('avg_training_loss =',loss_all.detach().cpu().numpy()/count)
    if step%10 == 0:
    	print('validation on',step,'steps')
    	validate(model,test_data)


print('training data using',time.time()-start,'seconds')

