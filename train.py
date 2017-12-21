import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models.model import UnconditionedHand
from utils.dataloader import StrokesDataset
from utils import plot_stroke

def get_testinput():
    start_stroke = np.asarray(0,0,0)
    start = torch.from_numpy(start_stroke)
    start = Variable(start)
    return start

# Some HyperParams
EPOCHS = 100
SAVE_FREQ = 1

# Get the model class
random = UnconditionedHand()

# Get the dataset class
dataset = StrokesDataset()

# DataLoader
dataloader = torch.utils.data.DataLoader(dataset,batch_size = 1)

# Optimizer
optimizer = optim.Adam(random.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    # print "Epoch: ",epoch," of ",EPOCHS
    hidden = None
    for i,data in enumerate(dataloader,):
        # print "MiniBatch Number: ",i
        init,next_stroke = data['initial'],data['next']
        # print next_stroke

        init,next_stroke = Variable(init),Variable(next_stroke)
        init = init.view(-1,1,3) # In accordance with nn.LSTM documentation.
        # print init.size()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward + Backward + Step
        # print hidden
        mu1,mu2,sigma1,sigma2,rho,mixprob,eos,hidden = random(init,hidden)
        total_loss = random.loss(next_stroke,mu1,mu2,sigma1,sigma2,rho,mixprob,eos)
        total_loss.backward()
        nn.utils.clip_grad_norm(random.parameters(), 10)
        optimizer.step()
        hidden.detach_()

        print "Mini, Loss Value: ",i,total_loss.data[0],"\n"

    if epoch % SAVE_FREQ == SAVE_FREQ - 1:
        torch.save(random.state_dict(),'uncon.model')
        test1 = torch.load('uncon.model')
        test_in = get_testinput()
        plot_stroke(test1.get_stroke(test_in))
