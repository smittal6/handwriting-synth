import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models.con_model import ConditionedHand
from utils.dataloader import StrokesDataset
from utils import plot_stroke

# Some HyperParams
EPOCHS = 3
SAVE_FREQ = 1

# Get the dataset class
dataset = StrokesDataset()

# Get the model class
random = ConditionedHand(dataset.vec_len)

# DataLoader
dataloader = torch.utils.data.DataLoader(dataset,batch_size = 1)

# Optimizer
optimizer = optim.Adam(random.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    # print "Epoch: ",epoch," of ",EPOCHS
    hidden = None
    for i,data in enumerate(dataloader,):
        # print "MiniBatch Number: ",i
        init,next_stroke,encoding = data['initial'],data['next'],data['encoding']
        # print next_stroke

        init,next_stroke,enocding = Variable(init),Variable(next_stroke),Variable(encoding)
        init = init.view(-1,1,3) # In accordance with nn.LSTM documentation.
        # print init.size()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward + Backward + Step
        # print hidden
        mu1,mu2,sigma1,sigma2,rho,mixprob,eos,hidden = random(init,encoding,hidden)
        total_loss = random.loss(next_stroke,mu1,mu2,sigma1,sigma2,rho,mixprob,eos)
        total_loss.backward()
        nn.utils.clip_grad_norm(random.parameters(), 10)
        optimizer.step()
        hidden.detach_()

        print "Mini, Loss Value: ",i,total_loss.data[0],"\n"

    if epoch % SAVE_FREQ == SAVE_FREQ - 1:
        torch.save(random.state_dict(),'mod_{:04d}.net'.format(epoch))

