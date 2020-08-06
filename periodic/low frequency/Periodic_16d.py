#!/usr/bin/env python
# coding: utf-8

# ### Numerical example
# ### Consider the following problem over $\Omega = {(-1, 1)}^d$: 
# ### $$-\Delta u + \pi^2 u = f,$$
# ### with periodic condition
# ### $$ u(x_1 + p_1,\cdots,x_{k} + p_k,\cdots,x_d + p_d) = u(x_1,\cdots,x_{k},\cdots,x_d) $$ 
# ### Assume $u(x) = \sum_{i = 1}^d \cos(\pi x_i) + \cos(2 \pi x_i) $, we can get $f(x)$ and $p_1 = \cdots = p_d = 2$.
# ### Network structure
# ### construct a transform $x^{\prime} = \text{transform} (x)$ before the first fully connected layer of our neural network
# ### $$x = (x_1,\cdots,x_d) \in R^d \Rightarrow x^{\prime} \in R^{2d}$$
# ### where $x^{\prime}_{2i - 1} = \sin(2\pi x_i / p_i)$ and $x^{\prime}_{2i} = \cos(2\pi x_i / p_i)$ for $i = 1, 2, \cdots, d$.

# In[1]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
import matplotlib.pyplot as plt
from math import *
import time
# torch.cuda.set_device(0)


# In[2]:


torch.set_default_tensor_type('torch.DoubleTensor')


# In[3]:


# activation function
def activation(x):
    return x * torch.sigmoid(x) 


# In[4]:


def transform(x):
    x_transform = torch.zeros(len(x), 2*d)
    for index in range(d):
        x_transform[:, 2*index] = torch.sin(2*pi*x[:, index] / 2)
        x_transform[:, 2*index+1] = torch.cos(2*pi*x[:, index] / 2)
    return x_transform.cuda()


# In[5]:


# exact solution
def u_ex(x):  
#     x_temp_1 = torch.cos(pi*x)
#     u_temp_1 = (x_temp_1.sum(1)).reshape([x.size()[0], 1]) # x_temp.sum(1) # summation by row for x_temp
#     x_temp_2 = torch.cos(2*pi*x)
#     u_temp_2 = (x_temp_2.sum(1)).reshape([x.size()[0], 1]) 
#     x_temp_3 = torch.cos(4*pi*x)
#     u_temp_3 = (x_temp_3.sum(1)).reshape([x.size()[0], 1]) 
#     x_temp_4 = torch.cos(8*pi*x)
#     u_temp_4 = (x_temp_4.sum(1)).reshape([x.size()[0], 1]) 
#     u_temp = u_temp_1 + u_temp_2 + u_temp_3 + u_temp_4
    
    x_temp_1 = torch.cos(pi*x)
    u_temp_1 = (x_temp_1.sum(1)).reshape([x.size()[0], 1]) # x_temp.sum(1) # summation by row for x_temp
    x_temp_2 = torch.cos(2*pi*x)
    u_temp_2 = (x_temp_2.sum(1)).reshape([x.size()[0], 1]) 
   
    u_temp = u_temp_1 + u_temp_2 
    return u_temp


# In[6]:


def f(x):
#     x_temp_1 = torch.cos(pi*x)
#     f_temp_1 = pi**2 * (x_temp_1.sum(1)).reshape([x.size()[0], 1]) 
#     x_temp_2 = torch.cos(2*pi*x)
#     f_temp_2 = (2*pi)**2 * (x_temp_2.sum(1)).reshape([x.size()[0], 1]) 
#     x_temp_3 = torch.cos(4*pi*x)
#     f_temp_3 = (4*pi)**2 * (x_temp_3.sum(1)).reshape([x.size()[0], 1]) 
#     x_temp_4 = torch.cos(8*pi*x)
#     f_temp_4 = (8*pi)**2 * (x_temp_4.sum(1)).reshape([x.size()[0], 1]) 
#     f_temp = f_temp_1 + f_temp_2 + f_temp_3 + f_temp_4 + pi**2 * u_ex(x)
    
    x_temp_1 = torch.cos(pi*x)
    f_temp_1 = pi**2 * (x_temp_1.sum(1)).reshape([x.size()[0], 1]) 
    x_temp_2 = torch.cos(2*pi*x)
    f_temp_2 = (2*pi)**2 * (x_temp_2.sum(1)).reshape([x.size()[0], 1]) 
    f_temp = f_temp_1 + f_temp_2 + pi**2 * u_ex(x)
    return f_temp


# In[7]:


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
        output = self.layer_in(transform(x)) # transform for periodic
        output = output + activation(self.layer_2(activation(self.layer_1(output)))) # residual block 1
        output = output + activation(self.layer_4(activation(self.layer_3(output)))) # residual block 2
        output = output + activation(self.layer_6(activation(self.layer_5(output)))) # residual block 3
        output = self.layer_out(output)
        return output


# In[8]:


d = 16 # dimension of input
input_size = d * 2
width_1 = 48
width_2 = 48
output_size_1 = 1
output_size_2 = d
data_size = 1000


# In[9]:


CUDA = torch.cuda.is_available()
# print('CUDA is: ', CUDA)
if CUDA:
    net_1 = Net(input_size, width_1, output_size_1).cuda() # network for u on gpu
    net_2 = Net(input_size, width_2, output_size_2).cuda() # network for grad u on gpu
else:
    net_1 = Net(input_size, width_1, output_size_1) # network for u on cpu
    net_2 = Net(input_size, width_2, output_size_2) # network for grad u on cpu


# In[10]:


# device = torch.device("cuda:0" )
# net_1.to(device)
# net_2.to(device)


# In[11]:


def generate_sample(data_size_temp):
    sample_temp = 2.0 * torch.rand(data_size_temp, d) - 1.0
    return sample_temp.cuda()


# In[12]:


def relative_l2_error():
    data_size_temp = 500
    x = generate_sample(data_size_temp).cuda() 
    predict = net_1(x)
    exact = u_ex(x)
    value = torch.sqrt(torch.sum((predict - exact)**2))/torch.sqrt(torch.sum((exact)**2))
    return value


# In[13]:


# Xavier normal initialization for weights:
#             mean = 0 std = gain * sqrt(2 / fan_in + fan_out)
# zero initialization for biases
def initialize_weights(self):
    for m in self.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


# In[14]:


initialize_weights(net_1)
initialize_weights(net_2)


# In[15]:


# number of net_1 and net_2
param_num_1 = sum(x.numel() for x in net_1.parameters())
param_num_2 = sum(x.numel() for x in net_2.parameters())
# print(param_num_1)
# print(param_num_2)


# In[16]:


def loss_function(x):
#     x = generate_sample(data_size).cuda()
#     x.requires_grad = True
    u_hat = net_1(x)
    grad_u_hat = torch.autograd.grad(outputs = u_hat, inputs = x, grad_outputs = torch.ones(u_hat.shape).cuda(), create_graph = True)
    p_hat = net_2(x)
    part_1 = torch.sum((grad_u_hat[0] - p_hat)**2) / len(x)
    laplace_u = torch.zeros([len(p_hat), 1]).cuda()
    for index in range(d):
        p_temp = p_hat[:, index].reshape([len(p_hat), 1])
        temp = torch.autograd.grad(outputs = p_temp, inputs = x, grad_outputs = torch.ones(p_temp.shape).cuda(), create_graph = True, allow_unused = True)[0]
        laplace_u = temp[:, index].reshape([len(p_hat), 1]) + laplace_u
    part_2 = torch.sum((-laplace_u + pi**2 * u_hat - f(x))**2)  / len(x)
    return part_1 + part_2 


# In[17]:


optimizer = optim.Adam([
                {'params': net_1.parameters()},
                {'params': net_2.parameters()},
            ])


# In[ ]:


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
    if i % 50 == 0:
        print("current epoch is: ", i)
        print("current loss is: ", loss.detach())
        print("current error is: ", error.detach())
    loss.backward()
    optimizer.step() 
time_end = time.time()
print('total time is: ', time_end-time_start, 'seconds')


# In[ ]:


np.save("loss_periodic_16d_1.npy", loss_record)
np.save("error_periodic_16d_1.npy", error_record)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




