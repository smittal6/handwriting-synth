import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models.model import UnconditionedHand
from utils.dataloader import StrokesDataset
from utils import plot_stroke

# Some HyperParams
EPOCHS = 1

# Get the model class
random = UnconditionedHand()

# Get the dataset class
dataset = StrokesDataset()

# DataLoader
dataloader = torch.utils.data.DataLoader(dataset,batch_size = 1)

# Optimizer
optimizer = optim.SGD(random.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    # print "Epoch: ",epoch," of ",EPOCHS
    hidden = None
    for i,data in enumerate(dataloader,):
        # print "MiniBatch Number: ",i
        init,next_stroke = data['initial'],data['next']
        # print next_stroke

        init,next_stroke = Variable(init),Variable(next_stroke)
        print init.size()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward + Backward + Step
        mu1,mu2,sigma1,sigma2,rho,mixprob,eos = random(init,hidden)
        total_loss = random.loss(next_stroke,mu1,mu2,sigma1,sigma2,rho,mixprob,eos)
        total_loss.backward()
        optimizer.step()

        print "Mini, Loss Value: ",i,total_loss.data[0]


