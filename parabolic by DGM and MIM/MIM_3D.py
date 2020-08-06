#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
# import matplotlib.pyplot as plt
from math import *
import time
torch.cuda.set_device(2)
torch.set_default_tensor_type('torch.DoubleTensor')


# In[2]:


# activation function
def activation(x):
    return x * torch.sigmoid(x) 


# In[3]:


# exact solution
def u_ex(x):     
    temp = 1.0
    for i in range(space_dimension):
        temp = temp * torch.sin(pi*x[:, i])
    u_temp = x[:, -1] * temp
    return u_temp.reshape([x.size()[0], 1])


# In[4]:


def f(x):
    temp = 1.0
    for i in range(space_dimension):
        temp = temp * torch.sin(pi*x[:, i])
    f_temp = (1.0 + space_dimension * x[:, -1] * pi**2) * temp
    return f_temp.reshape([x.size()[0], 1])


# In[5]:


# build ResNet with three blocks
class Net(nn.Module):
    def __init__(self,input_size,width,output_size):
        super(Net,self).__init__()
        self.layer_in = nn.Linear(input_size,width)
        self.layer_1 = nn.Linear(width,width)
        self.layer_2 = nn.Linear(width,width)
        self.layer_3 = nn.Linear(width,width)
        self.layer_4 = nn.Linear(width,width)
        self.layer_5 = nn.Linear(width,width)
        self.layer_6 = nn.Linear(width,width)
        self.layer_out = nn.Linear(width,output_size)
    def forward(self,x):
        output = self.layer_in(x) 
        output = output + activation(self.layer_2(activation(self.layer_1(output)))) # residual block 1
        output = output + activation(self.layer_4(activation(self.layer_3(output)))) # residual block 2
        output = output + activation(self.layer_6(activation(self.layer_5(output)))) # residual block 3
        output = self.layer_out(output)
        return output


# In[6]:


time_dimension = 1
space_dimension = 3
d = space_dimension + time_dimension # dimension of input include time and space variables
input_size = d 
width_1 = 8
width_2 = 8
output_size_1 = 1 
output_size_2 = d 
data_size = 2000


# In[7]:


CUDA = torch.cuda.is_available()
# print('CUDA is: ', CUDA)
if CUDA:
    net_1 = Net(input_size, width_1, output_size_1).cuda() # network for u on gpu
    net_2 = Net(input_size, width_2, output_size_2).cuda() # network for grad u and u_t on gpu
else:
    net_1 = Net(input_size, width_1, output_size_1) # network for u on cpu
    net_2 = Net(input_size, width_2, output_size_2) # network for grad u and u_t on cpu


# In[8]:


def model_u(x):
    x_temp = (x[:,0:d-1]).cuda()
    D_x_0 = torch.prod(x_temp, axis = 1).reshape([x.size()[0], 1]) 
    D_x_1 = torch.prod(1.0 - x_temp, axis = 1).reshape([x.size()[0], 1]) 
    model_u_temp = D_x_0 * D_x_1 * (x[:, -1]).reshape([x.size()[0], 1]) * net_1(x)
    return model_u_temp.reshape([x.size()[0], 1])


# In[9]:


def generate_sample(data_size_temp):
    sample_temp = torch.rand(data_size_temp, d)
    return sample_temp.cuda()


# In[10]:


def relative_l2_error():
    data_size_temp = 500
    x = generate_sample(data_size_temp).cuda() 
    predict = model_u(x)
    exact = u_ex(x)
    value = torch.sqrt(torch.sum((predict - exact)**2))/torch.sqrt(torch.sum((exact)**2))
    return value


# In[11]:


# Xavier normal initialization for weights:
#             mean = 0 std = gain * sqrt(2 / fan_in + fan_out)
# zero initialization for biases
def initialize_weights(self):
    for m in self.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


# In[12]:


initialize_weights(net_1)
initialize_weights(net_2)


# In[13]:


def loss_function(x):
#     x = generate_sample(data_size).cuda()
#     x.requires_grad = True
    u_hat = model_u(x)
    p = net_2(x)
    p_x = (p[:, 0:d-1]).reshape([x.size()[0], d - 1])
    p_t = (p[:, -1]).reshape([x.size()[0], 1])
    
    laplace_u_hat = torch.zeros([x.size()[0], 1]).cuda()
    for index in range(space_dimension):
        p_temp = p[:, index].reshape([x.size()[0], 1])
        temp = torch.autograd.grad(outputs = p_temp, inputs = x, grad_outputs = torch.ones(p_temp.shape).cuda(), create_graph = True, allow_unused = True)[0]
        laplace_u_hat = temp[:, index].reshape([x.size()[0], 1]) + laplace_u_hat
    part_1 = torch.sum((p_t -laplace_u_hat - f(x))**2) / len(x)
    
    grad_u_hat = torch.autograd.grad(outputs = u_hat, inputs = x, grad_outputs = torch.ones(u_hat.shape).cuda(), create_graph = True)
    part_2 = torch.sum(((grad_u_hat[0][:, 0:d-1]).reshape([x.size()[0], d - 1]) - p_x)**2) / len(x)
    
    u_hat_t = grad_u_hat[0][:, -1].reshape([x.size()[0], 1])
    part_3 = torch.sum((u_hat_t - p[:, -1].reshape([x.size()[0], 1]))**2) / len(x)
    return part_1 + part_2 + part_3


# In[14]:


optimizer = optim.Adam([
                {'params': net_1.parameters()},
                {'params': net_2.parameters()},
            ])


# In[15]:


epoch = 50000
loss_record = np.zeros(epoch)
error_record = np.zeros(epoch)
time_start = time.time()
for i in range(epoch):
    optimizer.zero_grad()
    x = generate_sample(data_size).cuda()
    x.requires_grad = True
    loss = loss_function(x)
    loss_record[i] = float(loss)
    error = relative_l2_error()
    error_record[i] = float(error)
    np.save("MIM_loss_parabolic_3d.npy", loss_record)
    np.save("MIM_error_parabolic_3d.npy", error_record)
    if i % 50 == 0:
        print("current epoch is: ", i)
        print("current loss is: ", loss.detach())
        print("current error is: ", error.detach())
    loss.backward()
    optimizer.step() 
time_end = time.time()
print('total time is: ', time_end-time_start, 'seconds')


# In[16]:


np.save("MIM_loss_parabolic_3d.npy", loss_record)
np.save("MIM_error_parabolic_3d.npy", error_record)


# In[ ]:




