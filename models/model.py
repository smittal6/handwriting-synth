# Imports
import sys
import math
sys.path.insert(0,'..')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import plot_stroke

print "Done importing"

class UnconditionedHand(nn.Module):

    def __init__(self,num_gauss = 20, rnn_size = 512):
        super(UnconditionedHand,self).__init__()

        self.num_gauss = num_gauss
        self.rnn_size = rnn_size

        # 1 for EOS, 6 for means, dev, correlation and mixing component
        self.output_size = 1 + 6*self.num_gauss 
        self.rnn = nn.LSTM(3,self.rnn_size,num_layers = 1)
        self.linear = nn.Linear(self.rnn_size, self.output_size) # For mapping back to R

    def forward(self,input,hidden):
        x, hidden_final = self.rnn(input,hidden)
        x = x.view(-1,self.rnn_size)
        x = self.linear(x) # x is Row X Columns

        ### Now, use the idea of mixture density networks, select to get network params
        # We need to divide each row along dim 1 to get params
        mu1 = x.index_select(1,torch.LongTensor([0,self.num_gauss - 1]))
        mu2 = x.index_select(1,torch.LongTensor([self.num_gauss,2*self.num_gauss - 1]))
        sigma1 = x.index_select(1,torch.LongTensor([2*self.num_gauss,3*self.num_gauss - 1]))
        sigma2 = x.index_select(1,torch.LongTensor([3*self.num_gauss,4*self.num_gauss - 1]))
        rho = x.index_select(1,torch.LongTensor([4*self.num_gauss,5*self.num_gauss - 1]))
        mixprob = x.index_select(1,torch.LongTensor([5*self.num_gauss,6*self.num_gauss - 1]))
        eos = x.index_select(1,torch.LongTensor([-1]))
        return mu1,mu2,sigma1,sigma2,rho,mixprob,eos

    def log_gauss(self,x1,x2,mu1,mu2,sigma1,sigma2,rho,mixprob):

        # In accordance with Equation 19
        mixprob = nn.Softmax(mixprob)

        # In accordance with Equation 21
        sigma1 = sigma1.exp()
        sigma2 = sigma2.exp()

        # Equation 22
        rho = nn.functional.Tanh(rho)

        x1, x2 = x1.repeat(1,self.num_gauss),x2.repeat(1,self.num_gauss)
        z1 = (x1 - mu1)/sigma1
        z2 = (x2 - mu2)/sigma2
        z = z1**2 + z2**2 - 2*z1*z2*rho

        normals = 1/(2*math.pi*sigma1*sigma2*math.sqrt(1-rho**2)) + (-1*z/(2*(1-rho**2))).exp()
        normals = mixprob*normals
        normals = normals.sum()
        return normals

    def loss(self,targets,mu1,mu2,sigma1,sigma2,rho,mixprob,eos):
        
        eos_index = torch.LongTensor([0])
        x_index = torch.LongTensor([1])
        y_index = torch.LongTensor([2])

        # Logits because of equation 18 in [1]
        eos_loss = nn.functional.binary_cross_entropy_with_logits(outputs,targets.index_select(eos_index,dim=1))
        gauss_loss = log_gauss(x1,x2,mu1,mu2,sigma1,sigma2,rho,mixprob)
    
        total_loss = torch.add(eos_loss,gauss_loss)
        return total_loss
