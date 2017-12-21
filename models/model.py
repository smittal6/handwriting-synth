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
        self.rnn = nn.RNN(3,self.rnn_size,num_layers = 1)
        self.linear = nn.Linear(self.rnn_size, self.output_size) # For mapping back to R

    def forward(self,input,hidden=None):

        print "Shape of input: ",input.size()
        x, hidden = self.rnn(input,hidden)
        # print "Shape of Hidden_final: ",hidden_final.size()
        # print "Shape of the result returned by the RNN: ",x.size()

        x = x.view(-1,self.rnn_size)
        # print "Shape after view: ",x.size()
        x = self.linear(x) # x is Row X Columns
        # print "Shape of the result after Linear Layer: ",x.size()

        ### Now, use the idea of mixture density networks, select to get network params
        # We need to divide each row along dim 1 to get params
        mu1,mu2,sigma1,sigma2,rho,mixprob,eos = torch.split(x,self.num_gauss,dim = 1)

        # print "Shape of eos: ",eos.size()
        # print "Shape of mu1: ",mu1.size()
        # print "Shape of sigma1: ",sigma1.size()
        return mu1,mu2,sigma1,sigma2,rho,mixprob,eos,hidden

    def log_gauss(self,x1,x2,mu1,mu2,sigma1,sigma2,rho,mixprob):

        # In accordance with Equation 19
        mixprob = nn.functional.softmax(mixprob,dim=1)

        # In accordance with Equation 21
        sigma1,sigma2 = sigma1.exp(),sigma2.exp()

        # Equation 22
        rho = nn.functional.tanh(rho)
        # print "Shape of rho: ",rho.size()
        # print "Shape of mu1: ",mu1.size()
        # print "Shape of sigma1: ",sigma1.size()

        x1, x2 = x1.repeat(1,self.num_gauss),x2.repeat(1,self.num_gauss)
        # print "Shape of x1: ",x1.size()
        z1 = (x1 - mu1)/sigma1
        z2 = (x2 - mu2)/sigma2
        z = z1**2 + z2**2 - 2*z1*z2*rho
        # print "Shape of z: ",z.size()

        t1 = (-1*z/(2*(1-rho**2))).exp()
        # print "Shape of t1: ",t1.size()

        normals = 1/(2*math.pi*sigma1*sigma2*(1-rho**2).sqrt()) + t1
        normals = mixprob*normals
        
        # print "Shape of normals: ",normals.size()
        normals = normals.sum()
        return normals

    def loss(self,targets,mu1,mu2,sigma1,sigma2,rho,mixprob,eos):

        targets = torch.squeeze(targets).float()
        # print "Shape of targets in loss function: ",targets.size()
        
        eos_index = Variable(torch.LongTensor([0]))
        x_index = Variable(torch.LongTensor([1]))
        y_index = Variable(torch.LongTensor([2]))

        # Logits because of equation 18 in [1]
        eos_loss = nn.functional.binary_cross_entropy_with_logits(eos,targets.index_select(1,eos_index))

        # Log Prob loss
        x1 = targets.index_select(1,x_index)
        x2 = targets.index_select(1,y_index)
        gauss_loss = self.log_gauss(x1,x2,mu1,mu2,sigma1,sigma2,rho,mixprob) 
        total_loss = torch.add(eos_loss,gauss_loss)
        return total_loss

    def multiSampler(self,mixprob):
        """
        Sampling from categorical distribution, for chosing the gaussian
        """
        mixprob = nn.functional.softmax(mixprob,dim=1)
        temp = np.random.multinomial(1,list(mixprob.to_numpy()))
        temp = list(temp)
        index = temp.index(max(temp))
        return index

    def gaussSampler(self,mu1,mu2,sigma1,sigma2,rho,index):
        """
        Sample from the Bivariate Gaussian chosen in Step 1
        """
        sigma1,sigma2 = sigma1.exp(),sigma2.exp()
        rho = nn.functional.tanh(rho)
        u1,u2 = mu1[0][index],mu2[0][index]
        s1,s2 = sigma1[0][index],sigma2[0][index]
        r = rho[0][index]
        x,y = np.random.multivariate_normal([u1,u2],[[s1*s1,r*s1*s2],[r*s1*s2,s2*s2]])
        return x,y

    def berSampler(self,eos):
        """
        Sampling from a Bernoulli distribution for 
        """
        prob = nn.functional.sigmoid(eos)
        token = torch.bernoulli(prob)
        return token.data[0]

    def get_stroke(self,input,hidden=None,timesteps = 300):
        """
        Samples the stroke from the currently learned model
        """
        stroke = []
        for step in range(timesteps):
            mu1,mu2,sigma1,sigma2,rho,mixprob,eos,hidden = self.forward(input,hidden)
            index = self.multiSampler(mixprob)
            token = self.berSampler(eos)
            x,y = self.gaussSampler(mu1,mu2,sigma1,sigma2,rho,index)
            to_append = [token,x,y]
            stroke.append(to_append)
        return np.asarray(stroke)
