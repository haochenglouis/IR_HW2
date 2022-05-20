import numpy as np 
import os 
import collections
import matplotlib.pyplot as plt



output = open('results/MF_cuda.out','rb')
output = output.readlines()

mae = []
rmse = []


for i in output:
	i = i.decode('utf-8').strip().split(' ')
	if i[0].strip() =='mae':
		mae.append(float(i[2]))
	if i[0].strip() =='rmse':
		rmse.append(float(i[2]))

epochs = np.arange(len(mae))*10

ig = plt.figure()
ax = plt.axes()
ax.set_title('MAE and RMSE with training epochs')
plt.setp(ax, xlabel = 'epoch')
plt.setp(ax, ylabel = 'test error')

#ax.plot(boolean_P_measure, boolean_R_measure, color="red", linewidth=2.0, linestyle="-",label="boolean")
ax.plot(epochs, mae, color="blue", linewidth=2.0, linestyle="-",label="MAE")
ax.plot(epochs, rmse, color="black", linewidth=2.0, linestyle="-",label="RMSE")
ax.legend()
plt.savefig('results/mae_rmse.pdf')











