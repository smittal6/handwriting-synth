# Imports
import sys
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

    def loss(self,targets,outputs):
        
        eos_index = torch.LongTensor([0])
        x_index = torch.LongTensor([1])
        y_index = torch.LongTensor([2])

        eos_loss = nn.functional.binary_cross_entropy_with_logits(outputs,targets.index_select(dim=1,eos_index))
        total_loss = torch.add(eos_loss,gauss_loss)
        return total_loss
