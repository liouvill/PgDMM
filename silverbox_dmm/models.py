import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
import data_loader as poly
import scipy.io as sio
import numpy as np
from scipy.linalg import expm


# the model p(z^{phy}_t | z^{phy}_{t-1})
class Transition_0(nn.Module):
    def __init__(self, z_dim, u_dim, transition_dim, Ad, Bd):
        super().__init__()
        # initialize the linear transformations used in the neural network
        self.lin_transfer = nn.Linear(z_dim, z_dim ,bias = False)
        self.lin_transfer.weight.data = Ad
        for param in self.lin_transfer.parameters():
            param.requires_grad = False  
        
        self.lin_u = nn.Linear(u_dim, z_dim, bias = False)           
        self.lin_u.weight.data = Bd
        for param in self.lin_u.parameters():
            param.requires_grad = False

        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_u_to_hidden = nn.Linear(u_dim, transition_dim)
        self.lin_gate_hidden_to_hidden = nn.Linear(transition_dim, transition_dim)
        self.lin_gate_hidden_to_loc = nn.Linear(transition_dim, z_dim)
        self.lin_gate_hidden_to_scale = nn.Linear(transition_dim, z_dim)

        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t, u_t):

            h1 = self.relu( self.lin_gate_z_to_hidden(z_t))
            h2 = self.relu( self.lin_gate_hidden_to_hidden(h1) )
            loc = self.lin_transfer(z_t) + self.lin_u(u_t)
            scale = self.softplus( self.lin_gate_hidden_to_scale(h2) )

            return loc, scale
        
# the model p(z^{NN}_t | z^{NN}_{t-1})
class Transition_1(nn.Module):
    def __init__(self, z_dim, u_dim, transition_dim, Bd):
        super().__init__()
        # initialize the linear transformations used in the neural network
        self.lin_gate_z_u_to_hidden = nn.Linear(z_dim+u_dim, transition_dim)
        self.lin_gate_u_to_hidden = nn.Linear(u_dim, transition_dim)
        self.lin_gate_hidden_to_hidden = nn.Linear(transition_dim, transition_dim)
        self.lin_gate_hidden_to_loc = nn.Linear(transition_dim, z_dim)
        self.lin_gate_hidden_to_scale = nn.Linear(transition_dim, z_dim)
        self.lin_u = nn.Linear(u_dim, z_dim)

        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t, u_t):
            
            z_u_t = torch.cat( (z_t,u_t),1 )
            
            h1 = self.relu( self.lin_gate_z_u_to_hidden(z_u_t) )
            h2 = self.relu( self.lin_gate_hidden_to_hidden(h1) )
            loc = self.lin_gate_hidden_to_loc(h2)
            scale = self.softplus( self.lin_gate_hidden_to_scale(h2) )
            
            return loc, scale

# the model p(x_t | z_t)
class Emission(nn.Module):
    def __init__(self, input_dim, z_dim, emission_dim, Cd):
        super().__init__()
        
        self.lin_emitter = nn.Linear(z_dim, input_dim, bias = True)

        # initialize the linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_scale = nn.Linear(emission_dim, input_dim)
        self.lin_hidden_to_loc = nn.Linear(emission_dim, input_dim)
        # initialize the two non-linearities used in the neural networks
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t):            

            h1 = self.relu(self.lin_z_to_hidden(z_t))
            h2 = self.relu(self.lin_hidden_to_hidden(h1))
            loc = self.lin_emitter(z_t) 
            scale = self.softplus(self.lin_hidden_to_scale(h2))
            
            return loc, scale

# the model q(z^{phy}_t | z^{phy}_{t-1}, x_{1:T})
class Combiner_0(nn.Module):
    """
    Parameterizes `q(z^{phy}_t | z^{phy}_{t-1}, x_{1:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{1:T}` is through
    the hidden state of the RNN
    """

    def __init__(self, z_dim, u_dim, rnn_dim, Bd):
        super().__init__()
        # initialize the linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        
        self.lin_u_inf = nn.Linear(u_dim, z_dim, bias = False)
        self.lin_u_inf.weight.data = Bd
        for param in self.lin_u_inf.parameters():
            param.requires_grad = False  
        
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

    def forward(self, z_t_1, u_t, h_rnn_1, h_rnn_2):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden states 
        of the RNN `h(x_{1:t})` and `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z^{phy}_t | z^{phy}_{t-1}, x_{1:T})`
        """
        # combine the rnn hidden state with a transformed version of z_t_1     
        h_combined = 1/3.0 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn_1 + h_rnn_2)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(self.relu(h_combined)) + self.lin_u_inf(u_t)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(self.relu(h_combined)))
        
        # use the combined hidden state to compute the scale used to sample z_t
        # scale = self.softplus(self.lin_hidden_to_scale(self.relu(h1)))
        # return loc, scale which can be fed into Normal
        return loc, scale
        
# the model q(z^{NN}_t | z^{NN}_{t-1}, x_{1:T})
class Combiner_1(nn.Module):
    """
    Parameterizes `q(z^{NN}_t | z^{NN}_{t-1}, x_{1:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{1:T}` is through
    the hidden state of the RNN
    """

    def __init__(self, z_dim, u_dim, rnn_dim, Bd):
        super().__init__()
        # initialize the linear transformations used in the neural network
        self.lin_z_u_to_hidden = nn.Linear(z_dim + u_dim, rnn_dim)

        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)

        self.lin_u_inf = nn.Linear(u_dim, z_dim)

        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()
        

    def forward(self, z_t_1, u_t, h_rnn_1, h_rnn_2):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden states 
        of the RNN `h(x_{1:t})` and `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z^{NN}_t | z^{NN}_{t-1}, x_{1:T})`
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        z_u_t_1 = torch.cat( (z_t_1,u_t), 1  )
        h_combined = 1/3.0 * (self.tanh(self.lin_z_u_to_hidden(z_u_t_1)) + h_rnn_1 + h_rnn_2)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(self.relu(h_combined))
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(self.relu(h_combined)))
        
        # use the combined hidden state to compute the scale used to sample z_t
        # scale = self.softplus(self.lin_hidden_to_scale(self.relu(h1)))
        # return loc, scale which can be fed into Normal
        return loc, scale

# encapsulate the model as well as the variational distribution (the guide) for the Physics-guided Deep Markov Model
class DMM(nn.Module):

    def __init__(self, input_dim= 1, z_dim=2, u_dim=1, emission_dim=30,
                 transition_dim=30, rnn_dim=70, num_layers=1, rnn_dropout_rate=0.0,
                 num_iafs=0, iaf_dim=50, alpha = 0.5, use_cuda=False):
        super().__init__()
        
        ns = 1
        fs0 = 10**7/2**14
        fs = fs0/ns
        dt = 1/fs

        self.Ad = torch.tensor([[0.7636, 0.0014],[-272.4851, 0.6917]])
        self.Bd = torch.tensor([[0.2505],[288.7715]])
        self.Cd = torch.tensor([[1.0, 0.0]])
        
        # instantiate PyTorch modules used in the model and guide below
        self.trans_0 = Transition_0( z_dim, u_dim, transition_dim, self.Ad, self.Bd)
        self.trans_1 = Transition_1( z_dim, u_dim, transition_dim,self.Bd)
        self.emission = Emission(input_dim,z_dim,emission_dim, self.Cd)
        self.combiner_0 = Combiner_0(z_dim, u_dim, rnn_dim, self.Bd)
        self.combiner_1 = Combiner_1(z_dim, u_dim, rnn_dim, self.Bd)
        
        # dropout just takes effect on inner layers of rnn
        rnn_dropout_rate = 0. if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=rnn_dim, 
                          batch_first=True, bidirectional=True, num_layers=num_layers,
                          dropout=rnn_dropout_rate)

        # if we're using normalizing flows, instantiate those too
        self.iafs = [affine_autoregressive(z_dim, hidden_dims=[iaf_dim]) for _ in range(num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_0_1 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0_1 = nn.Parameter(torch.zeros(z_dim))
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(2, 1, rnn_dim))

        self.use_cuda = use_cuda
        self.alpha = nn.Parameter(torch.ones(1,)*alpha)
        self.alpha.requires_grad = False
        
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()

    # the model p(x_{1:T} | z_{1:T}) p(z_{1:T})
    def model(self, mini_batch_u, mini_batch, mini_batch_reversed, mini_batch_mask,
              mini_batch_seq_lengths, annealing_factor=1.0):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("dmm", self)

        # set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev_0 = self.z_0_0.expand(mini_batch.size(0), self.z_0_0.size(0))
        z_prev_1 = self.z_0_1.expand(mini_batch.size(0), self.z_0_1.size(0))
        
        # we enclose all the sample statements in the model in a plate.
        # this marks that each datapoint is conditionally independent of the others
        with pyro.plate("z_minibatch", len(mini_batch)):
            # sample the latents z and observed x's one time step at a time
            for t in pyro.markov(range(1, T_max + 1)):
                # the next chunk of code samples z_t ~ p(z_t | z_{t-1})
                # note that (both here and elsewhere) we use poutine.scale to take care
                # of KL annealing. we use the mask() method to deal with raggedness
                # in the observed data (i.e. different sequences in the mini-batch
                # have different lengths)

                # first compute the parameters of the diagonal gaussian distribution p(z_t | z_{t-1})
                z_loc_0, z_scale_0 = self.trans_0(z_prev_0, mini_batch_u[:, t-1, :])
                z_loc_1, z_scale_1 = self.trans_1(z_prev_1, mini_batch_u[:, t-1, :])
                # then sample z_t according to dist.Normal(z_loc, z_scale)
                # note that we use the reshape method so that the univariate Normal distribution
                # is treated as a multivariate Normal distribution with a diagonal covariance.
                with poutine.scale(scale=annealing_factor):
                    z_t_0 = pyro.sample("z_%d_0" % t,
                                      dist.Normal(z_loc_0, z_scale_0)
                                          #.mask(mini_batch_mask[:, t - 1:t])
                                          .to_event(1))
                    z_t_1 = pyro.sample("z_%d_1" % t,
                                      dist.Normal(z_loc_1, z_scale_1)
                                          #.mask(mini_batch_mask[:, t - 1:t])
                                          .to_event(1))
                    z_t = self.alpha * z_t_0 + (1-self.alpha) * z_t_1

                # compute the probabilities that parameterize the bernoulli likelihood
                x_loc, x_scale = self.emission(z_t)
                # the next statement instructs pyro to observe x_t according to the
                # gaussian distribution p(x_t|z_t)
                pyro.sample("obs_x_%d" % t,
                            dist.Normal(x_loc, x_scale)
                            #.mask(mini_batch_mask[:, t - 1:t])
                            .to_event(1),
                            obs=mini_batch[:, t - 1, :])
                # the latent sampled at this time step will be conditioned upon
                # in the next time step so keep track of it
                z_prev_0 = z_t_0
                z_prev_1 = z_t_1

    # the guide q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
    def guide(self, mini_batch_u, mini_batch, mini_batch_reversed, mini_batch_mask,
              mini_batch_seq_lengths, annealing_factor=1.0):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)
        # register all PyTorch (sub)modules with pyro
        pyro.module("dmm", self)

        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        h_0_contig = self.h_0.expand(2, mini_batch.size(0), self.rnn.hidden_size).contiguous()
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, _ = self.rnn(mini_batch, h_0_contig)
        rnn_output = rnn_output.view(rnn_output.size(0),rnn_output.size(1),
                                      2,self.rnn.hidden_size)
        h1 = rnn_output[:,:,0,:]
        h2 = rnn_output[:,:,1,:]
        
        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev_0 = self.z_q_0_0.expand(mini_batch.size(0), self.z_q_0_0.size(0))
        z_prev_1 = self.z_q_0_1.expand(mini_batch.size(0), self.z_q_0_1.size(0))
        
        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", len(mini_batch)):
            # sample the latents z one time step at a time
            for t in pyro.markov(range(1, T_max + 1)):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
                z_loc_0, z_scale_0 = self.combiner_0(z_prev_0, mini_batch_u[:, t-1, :], h1[:, t-1, :], h2[:,t-1,:])
                z_loc_1, z_scale_1 = self.combiner_1(z_prev_1, mini_batch_u[:, t-1, :], h1[:, t-1, :], h2[:,t-1,:])
                # if we are using normalizing flows, we apply the sequence of transformations
                # parameterized by self.iafs to the base distribution defined in the previous line
                # to yield a transformed distribution that we use for q(z_t|...)
                if len(self.iafs) > 0:
                    z_dist_0 = TransformedDistribution(dist.Normal(z_loc_0, z_scale_0), self.iafs)
                    z_dist_1 = TransformedDistribution(dist.Normal(z_loc_1, z_scale_1), self.iafs)
                    assert z_dist_0.event_shape == (self.z_q_0_0.size(0),)
                    assert z_dist_1.event_shape == (self.z_q_0_1.size(0),)
                    assert z_dist_0.batch_shape[-1:] == (len(mini_batch),)
                    assert z_dist_1.batch_shape[-1:] == (len(mini_batch),)
                else:
                    z_dist_0 = dist.Normal(z_loc_0, z_scale_0)
                    z_dist_1 = dist.Normal(z_loc_1, z_scale_1)
                    assert z_dist_0.event_shape == ()
                    assert z_dist_1.event_shape == ()
                    assert z_dist_0.batch_shape[-2:] == (len(mini_batch), self.z_q_0_0.size(0))
                    assert z_dist_1.batch_shape[-2:] == (len(mini_batch), self.z_q_0_1.size(0))

                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(scale=annealing_factor):
                    if len(self.iafs) > 0:
                        # in output of normalizing flow, all dimensions are correlated (event shape is not empty)
                        z_t_0 = pyro.sample("z_%d_0" % t, z_dist_0)
                                          #z_dist_0.mask(mini_batch_mask[:, t - 1]))
                        z_t_1 = pyro.sample("z_%d_1" % t, z_dist_1)
                                          #z_dist_1.mask(mini_batch_mask[:, t - 1]))
                    else:
                        # when no normalizing flow used, ".to_event(1)" indicates latent dimensions are independent
                        z_t_0 = pyro.sample("z_%d_0" % t,
                                          #z_dist_0.mask(mini_batch_mask[:, t - 1:t])
                                          z_dist_0.to_event(1))
                        z_t_1 = pyro.sample("z_%d_1" % t,
                                          #z_dist_1.mask(mini_batch_mask[:, t - 1:t])
                                          z_dist_1.to_event(1))
                # the latent sampled at this time step will be conditioned upon in the next time step
                # so keep track of it
                z_prev_0 = z_t_0
                z_prev_1 = z_t_1