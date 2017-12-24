import os
import sys
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models.model import UnconditionedHand
from utils.dataloader import StrokesDataset
from utils import plot_stroke


def get_testinput():
    start_stroke = np.asarray((0,0,0))
    start = torch.from_numpy(start_stroke).float()
    start = Variable(start)
    start = start.view(-1,1,3)
    return start

def test(save_image=None):
    test1 = UnconditionedHand()
    test1.load_state_dict(torch.load('./save/uncon.model'))
    test_in = get_testinput()
    stroke = test1.get_stroke(test_in)
    stroke = stroke.astype(np.float32)
    print stroke.shape
    print stroke
    print stroke.dtype
    plot_stroke(stroke,save_name = save_image)


parser = argparse.ArgumentParser(description = "")
parser.add_argument('--epochs',type=int,default=10)
parser.add_argument('--mode',type=str,default='train')
# parser.add_argument()
args = parser.parse_args()


# Get the dataset class
dataset = StrokesDataset()
LEN = dataset.__len__()


if args.mode == 'train':

    SAVE_FREQ = 1

    # Get the model class
    random = UnconditionedHand()


    # DataLoader
    dataloader = torch.utils.data.DataLoader(dataset,batch_size = 1)

    # Optimizer
    optimizer = optim.Adam(random.parameters(), lr=0.0001)

    for epoch in range(args.epochs):
        try: 
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
                mu1,mu2,sigma1,sigma2,rho,mixprob,eos,hidden = random(init,hidden)
                total_loss = random.loss(next_stroke,mu1,mu2,sigma1,sigma2,rho,mixprob,eos)
                total_loss.backward()
                nn.utils.clip_grad_norm(random.parameters(), 10)
                optimizer.step()
                hidden.detach_()

                print "Mini, Loss Value: ",i,total_loss.data[0],"\n"

                if  i == LEN/4 - 1:
                    torch.save(random.state_dict(),'./save/uncon.model')
                    test(save_image = './save/unctest.png')

        except KeyboardInterrupt:
            print "Saving model, and generating a random file"
            torch.save(random.state_dict(),'./save/uncon.model')
            test(save_image = './save/unctest.png')
            sys.exit()
else:
    test(save_image = None)
