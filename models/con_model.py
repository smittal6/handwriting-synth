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

print "Done importing"

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

        # 1 for EOS, 6 for means, dev, correlation and mixing component
        self.output2_size = 1 + 6*self.num_gauss 

        self.rnn1 = nn.RNN(3,self.rnn_size,num_layers = 1) # [inputs, hidden-size,num_layers]
        self.linear1 = nn.Linear(self.rnn_size, self.output1_size) # For mapping back to R, to get 
        self.rnn2 = nn.RNN(self.input_rnn_size,self.rnn_size, num_layers = 1) # [inputs]
        self.linear2 = nn.Linear(self.rnn_size,self.output2_size) # For mapping back and getting MDN outputs

    def forward(self,input,encoding,hidden1=None,hidden2=None):
        """
        Forward pass function.
        """

        # encoding = encoding.float().squeeze() # Don't care about batch now
        # print "Shape of encoding: ",encoding.size()

        print "Shape of input: ",input.size()
        x, hidden1 = self.rnn1(input,hidden1)
        # print "Shape of H idden_final: ",hidden1.size()
        # print "Shape of the result returned by the RNN1: ",x.size()

        x = x.view(-1,self.rnn_size)
        # print "Shape after view: ",x.size()
        x = self.linear1(x) # These are the outputs as required to calculate alpha,beta,kappa
        # print "Shape of the result after Linear1 Layer: ",x.size()
        # print "ID of x after linear1 layer: ",id(x)

        # Obtain alpha,beta,kappa
        alpha, beta, kappa = self.get_params(x)
        # print "Shape of alpha: ",alpha.size()
        # print "Shape of beta: ",beta.size()
        # print "Shape of kappa: ",kappa.size()

        # Calculate Window at each time t using the output of linear1 layer
        window = self.obtain_window(alpha,beta,kappa,encoding)

        # Concatenate this window along with input
        x = torch.cat((input.view(-1,3), window),dim = 1)
        # print "ID of x after concatenating with window layer: ",id(x)
        # print "Shape after concatenation: ",x.size()

        # Feed into rnn2
        x = x.view(-1,1,self.input_rnn_size)
        x, hidden2 = self.rnn2(x,hidden2)
        # print "Shape of the result after RNN2 Layer: ",x.size()

        # Feed into linear2 layer
        x = x.view(-1,self.rnn_size)
        x = self.linear2(x)
        # print "Shape of the result after Linear2 Layer: ",x.size()

        ### Now, use the idea of mixture density networks, select to get network params
        # We need to divide each row along dim 1 to get params
        mu1,mu2,sigma1,sigma2,rho,mixprob,eos = torch.split(x,self.num_gauss,dim = 1)

        return mu1,mu2,sigma1,sigma2,rho,mixprob,eos,hidden1,hidden2

    def obtain_window(self,alpha,beta,kappa,encoding):
        """

        Get the window for all timesteps
        Window shape: [TimeSteps, vec_len]
        Phi shape: [TimeSteps, char_len]
        Encoding Shape: [char_len,vec_len]

        """

        timesteps, vec_len, char_len  = np.asarray(alpha.size())[0], np.asarray(encoding.size())[1], np.asarray(encoding.size())[0]

        encoding = encoding.repeat(timesteps,1,1)
        # print "Shape of time repeated encoding: ",encoding.size() # [timesteps,char_len,vec_len]

        phi = self.obtain_phi(alpha,beta,kappa,encoding,timesteps,vec_len,char_len)
        # phi = Variable(torch.rand(timesteps,char_len)) # For testing the correctness
        # print "ID of phi in obtain_window after obtaining phi: ",id(phi)
        # print "Shape of phi: ",phi.size()

        # phi = phi.repeat(vec_len,1,1)
        # print "ID of phi in obtain_window: ",id(phi)
        # phi = phi.permute(1,2,0)
        # print "ID of phi in obtain_window: ",id(phi)
        # print "Type of encoding: ",type(encoding)
        # print "Type of phi ",type(phi)
        window = encoding*phi
        # print "ID of window in obtain_window: ",id(window)
        window = window.sum(dim = 1)
        # print "ID of window in obtain_window: ",id(window)
        # print "Check window size: ",window.size()

        return window


    def obtain_phi(self,alpha,beta,kappa,encoding,timesteps,vec_len,char_len):
        """
        Get Phi for t and u
        Alpha, Beta, Kappa Shape: [TimeSteps,num_wgauss]
        Phi Shape: [TimeSteps, Char_len]
        """

        calpha = alpha.repeat(char_len,1,1)
        # print "ID of calpha in obtain_phi: ",id(calpha)
        calpha = calpha.view(-1,char_len,self.num_wgauss)
        # print "ID of calpha in obtain_phi: ",id(calpha)
        cbeta = beta.repeat(char_len,1,1)
        # print "ID of cbeta in obtain_phi: ",id(cbeta)
        cbeta = cbeta.view(-1,char_len,self.num_wgauss)
        # print "ID of cbeta in obtain_phi: ",id(cbeta)
        ckappa = kappa.repeat(char_len,1,1)
        # print "ID of ckappa in obtain_phi: ",id(ckappa)
        ckappa = ckappa.view(-1,char_len,self.num_wgauss)
        # print "ID of ckappa in obtain_phi: ",id(ckappa)

        u_vec = Variable(torch.linspace(0,char_len-1,char_len),requires_grad = False)
        # print "ID of u_vec in obtain_phi: ",id(u_vec)
        u_vec = u_vec.view(char_len,1)
        # print "ID of u_vec in obtain_phi: ",id(u_vec)
        u_vec = u_vec.repeat(1,self.num_wgauss)
        # print "ID of u_vec in obtain_phi: ",id(u_vec)
        u_vec = u_vec.repeat(timesteps,1,1)
        # print "ID of u_vec in obtain_phi: ",id(u_vec)

        # print "Shape of cKappa: ",ckappa.size()
        # print "Shape of u_vec: ",u_vec.size()

        phi = ckappa.sub(u_vec)
        phi = (-1*cbeta*phi).exp()
        phi = calpha * phi
        phi = phi.sum(dim = 2)

        # phi = Variable(torch.rand(timesteps,char_len)) # For testing the correctness
        phi = phi.repeat(vec_len,1,1)
        phi = phi.permute(1,2,0)
        # print "ID of phi in obtain_phi: ",id(phi)
        return phi

    def get_params(self,x):
        """
        Gets the output of linear1 layer, and return alpha,beta,kappa at all times
        """

        alpha_hat, beta_hat, kappa_hat = torch.split(x,self.num_wgauss,dim = 1)
        alpha = alpha_hat.exp()
        beta = beta_hat.exp()
        kappa = kappa_hat.exp()
        for i in range(1,np.asarray(kappa_hat.size())[0]):
            kappa[i,:] = kappa[i-1,:] + kappa[i,:]
        return alpha,beta,kappa

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
        Sampling from a Bernoulli distribution for 
        """

        prob = nn.functional.sigmoid(eos)
        token = torch.bernoulli(prob)
        return token.data[0]

    def get_stroke(self,input,encoding,hidden1 = None,hidden2 = None,timesteps = 500):
        """
        Samples the stroke from the currently learned model
        """

        stroke = []
        for step in range(timesteps):
            mu1,mu2,sigma1,sigma2,rho,mixprob,eos,hidden1,hidden2 = self.forward(input,encoding,hidden1,hidden2)
            index = self.multiSampler(mixprob)
            token = self.berSampler(eos)
            x,y = self.gaussSampler(mu1,mu2,sigma1,sigma2,rho,index)
            to_append = [token,x,y]
            stroke.append(to_append)
        return np.asarray(stroke)
