import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models.con_model import ConditionedHand
from utils.dataloader import StrokesDataset
from utils import plot_stroke

def get_testinput(dataset,text):
    """
    Get the input, and encoding for the text.
    """
    start_stroke = np.asarray((0,0,0))
    start = torch.from_numpy(start_stroke).float()
    start = Variable(start)
    start = start.view(-1,1,3)
    encoding = dataset.getOneHot(text)
    encoding = Variable(encoding)
    return start,encoding

def test(dataset,save_image=None,text="welcome to lyrebird"):
    """
    Generate and Plot a stroke from already existing model
    """
    test1 = ConditionedHand(dataset.vec_len)
    test1.load_state_dict(torch.load('./save/conditioned.model'))
    test_in,encoding = get_testinput(dataset,text)
    stroke = test1.get_stroke(test_in,encoding)
    plot_stroke(stroke,save_name = save_image)


# Some HyperParams
EPOCHS = 100
SAVE_FREQ = 1

# Get the dataset class
dataset = StrokesDataset()
LEN = dataset.__len__()

# Get the model class
random = ConditionedHand(dataset.vec_len)

# DataLoader
dataloader = torch.utils.data.DataLoader(dataset,batch_size = 1)

# Optimizer
optimizer = optim.Adam(random.parameters(), lr=0.0001)

for epoch in range(EPOCHS):
    try:
        hidden = None
        for i,data in enumerate(dataloader,):
            # print "MiniBatch Number: ",i
            init,next_stroke,encoding = data['initial'],data['next'],data['encoding']

            init,next_stroke,encoding = Variable(init),Variable(next_stroke),Variable(encoding)
            encoding = encoding.float().squeeze()
            init = init.view(-1,1,3) # In accordance with nn.LSTM documentation.
            # print init.size()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward + Backward + Step
            mu1,mu2,sigma1,sigma2,rho,mixprob,eos,hidden1,hidden2 = random(init,encoding,hidden)
            total_loss = random.loss(next_stroke,mu1,mu2,sigma1,sigma2,rho,mixprob,eos)
            total_loss.backward()
            nn.utils.clip_grad_norm(random.parameters(), 10)
            optimizer.step()
            hidden1.detach_()
            hidden2.detach_()

            print "Mini, Loss Value: ",i,total_loss.data[0],"\n"

            if  i == LEN/4 - 1:
                torch.save(random.state_dict(),'./save/conditioned.model')
                # test(dataset,save_image = './save/condtest.jpg')

    except KeyboardInterrupt:
        print "Saving model, and generating a random file"
        torch.save(random.state_dict(),'./save/conditioned.model')
        test(dataset,save_image = './save/condtest.jpg')
        sys.exit()
