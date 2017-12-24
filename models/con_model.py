# Imports
import sys
import math
import numpy as np
sys.path.insert(0,'..')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import plot_stroke


class ConditionedHand(nn.Module):
    """
    Model class for conditioned handwriting generation.
    Takes text as input in an encoded fashion.
    """

    def __init__(self,vec_len,num_gauss = 20, num_wgauss = 10, rnn_size = 512):

        super(ConditionedHand,self).__init__()

        self.num_wgauss = num_wgauss # Window Gaussians
        self.num_gauss = num_gauss # Density Gaussians

        self.rnn_size = rnn_size
        self.input_rnn_size = 3 + vec_len
        self.output1_size = 3*self.num_wgauss
        self.output2_size = 1 + 6*self.num_gauss # 1 for EOS, 6 for means, dev, correlation and mixing component

        # Layer defs
        self.rnn1 = nn.RNN(3,self.rnn_size,num_layers = 1) # [inputs, hidden-size,num_layers]
        self.linear1 = nn.Linear(self.rnn_size, self.output1_size) # For mapping back to R, to get 
        self.rnn2 = nn.RNN(self.input_rnn_size,self.rnn_size, num_layers = 1) # [inputs]
        self.linear2 = nn.Linear(self.rnn_size,self.output2_size) # For mapping back and getting MDN outputs


    def forward(self,input,encoding,hidden1=None,hidden2=None):
        """
        Forward pass function.
        Args:
            input: the stroke data, with format: [EOS, Delta_x, Delta_y]
            encoding: one-hot representation of the given string with time along dim 0
            hidden1: hidden state of RNN which models attention params
            hidden2: hidden state of the final RNN modelling the gaussian mixture params
        """

        # print "Shape of input: ",input.size()
        x, hidden1 = self.rnn1(input,hidden1)
        # print "Shape of the result returned by the RNN1: ",x.size()

        x = x.view(-1,self.rnn_size)
        x = self.linear1(x)

        # Obtain alpha,beta,kappa
        alpha, beta, kappa = self.get_params(x)

        # Calculate Window at each time t using the output of linear1 layer
        window = self.obtain_window(alpha.clone(),beta.clone(),kappa.clone(),encoding)

        # Concatenate this window along with input
        x = torch.cat((input.view(-1,3), window),dim = 1)

        # Feed into rnn2
        x = x.view(-1,1,self.input_rnn_size)
        x, hidden2 = self.rnn2(x,hidden2)
        # print "Shape of the result after RNN2 Layer: ",x.size()

        # Feed into linear2 layer
        x = x.view(-1,self.rnn_size)
        x = self.linear2(x)
        # print "Shape of the result after Linear2 Layer: ",x.size()

        # Use the idea of mixture density networks, select to get network params
        mu1,mu2,sigma1,sigma2,rho,mixprob,eos = torch.split(x,self.num_gauss,dim = 1)

        return mu1,mu2,sigma1,sigma2,rho,mixprob,eos,hidden1,hidden2


    def obtain_window(self,alpha,beta,kappa,encoding):
        """
        Get the window for all timesteps.
        Args:
            alpha,beta,kappa: shape ~ [timesteps,num_wgauss]
            Encoding: shape ~ [char_len,vec_len], char_len = U in paper.
        Returns:
            Window: shape ~ [timesteps, vec_len]
        """

        # Get some crucial lengths
        timesteps, vec_len, char_len  = np.asarray(alpha.size())[0], np.asarray(encoding.size())[1], np.asarray(encoding.size())[0]
        # print "Shape of encoding in obtain_window: ",encoding.size()

        window = Variable(torch.Tensor(timesteps,vec_len))

        for t in range(timesteps):
            # From this for loop I get window for each t.
            phi = self.obtain_phi(alpha,beta,kappa,t,vec_len,char_len) # phi ~ [1, char_len]
            window[t,:] = torch.mm(phi,encoding)

        return window


    def obtain_phi(self,alpha,beta,kappa,which_t,vec_len,char_len):
        """
        Args:
            alpha,beta,kappa: shape ~ [timesteps,num_wgauss]
            which_t: for indexing in alpha,beta,kappa according to timestep currently being processed
        Returns:
            phi_t: phi for a timestep for all u. Shape ~ [1, char_len]
        """

        u_vec = Variable(torch.linspace(0,char_len-1,char_len),requires_grad = False)
        u_vec = u_vec.view(char_len,1)
        u_vec = u_vec.repeat(1,self.num_wgauss)
        # print u_vec

        phi_t = u_vec - kappa[which_t,:]
        phi_t = (-1*beta[which_t,:]*phi_t**2).exp()
        phi_t = alpha[which_t,:] * phi_t 
        phi_t = phi_t.sum(dim = 1) # Summing along num_wgauss
        phi_t = phi_t.view(1,char_len) # Reshaping

        return phi_t


    def get_params(self,x):
        """
        Args:
            x: The output of linear1 layer
        Returns: 
            alpha,beta,kappa for all timesteps        
        """

        alpha_hat, beta_hat, kappa_hat = torch.split(x,self.num_wgauss,dim = 1)
        alpha = alpha_hat.exp()
        beta = beta_hat.exp()
        kappa_exp = kappa_hat.exp()

        kappa = Variable(torch.Tensor(kappa_exp.size()))
        kappa[0,:] = kappa_exp[0,:]
        for i in range(1,np.asarray(kappa_hat.size())[0]):
            kappa[i,:] = kappa[i-1,:] + kappa_exp[i,:]
        return alpha,beta,kappa


    def log_gauss(self,x1,x2,mu1,mu2,sigma1,sigma2,rho,mixprob):

        # In accordance with Equation 19
        mixprob = nn.functional.softmax(mixprob,dim=1)

        # In accordance with Equation 21
        sigma1,sigma2 = sigma1.exp(),sigma2.exp()

        # Equation 22
        rho = nn.functional.tanh(rho)

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
        
        eos_true,x1,x2 = torch.split(targets,1,dim = 1)

        # Logits because of equation 18 in [1]
        eos_loss = nn.functional.binary_cross_entropy_with_logits(eos,eos_true)

        # Log Prob loss
        gauss_loss = self.log_gauss(x1,x2,mu1,mu2,sigma1,sigma2,rho,mixprob) 

        # Add both the losses
        total_loss = torch.add(eos_loss,gauss_loss)

        return total_loss


    def multiSampler(self,mixprob):
        """
        Sampling from categorical distribution, for chosing the gaussian
        """

        mixprob = nn.functional.softmax(mixprob.double(),dim=1)
        probab_list = mixprob.data.numpy().flatten()
        temp = np.random.multinomial(1,probab_list)
        temp = list(temp)
        index = temp.index(max(temp))
        return index


    def gaussSampler(self,mu1,mu2,sigma1,sigma2,rho,index):
        """
        Sample from the Bivariate Gaussian chosen in Step 1
        """

        sigma1,sigma2 = sigma1.exp(),sigma2.exp()
        rho = nn.functional.tanh(rho)
        u1,u2 = mu1[0][index].data,mu2[0][index].data
        s1,s2 = sigma1[0][index].data,sigma2[0][index].data
        r = rho[0][index].data
        x,y = np.random.multivariate_normal([u1,u2],[[s1*s1,r*s1*s2],[r*s1*s2,s2*s2]])
        return x,y


    def berSampler(self,eos):
        """
        Sampling from a Bernoulli distribution for EOS
        """

        prob = nn.functional.sigmoid(eos)
        token = torch.bernoulli(prob)
        return token.data[0]


    def get_stroke(self,input,encoding,hidden1 = None,hidden2 = None,timesteps = 500):
        """
        Samples the stroke from the currently learned model
        """

        stroke = [[0,0,0]]
        for step in range(timesteps):
            mu1,mu2,sigma1,sigma2,rho,mixprob,eos,hidden1,hidden2 = self.forward(input,encoding,hidden1,hidden2)
            index = self.multiSampler(mixprob)
            token = self.berSampler(eos)
            x,y = self.gaussSampler(mu1,mu2,sigma1,sigma2,rho,index)
            to_append = [token,x,y]
            stroke.append(to_append)
        return np.asarray(stroke)
