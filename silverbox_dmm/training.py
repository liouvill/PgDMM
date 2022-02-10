import argparse
import logging
import time
from os.path import exists

import numpy as np
import scipy.io as sio
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorboardX import SummaryWriter

import pyro
import data_loader as poly
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import SVI, config_enumerate
from pyro.optim import ClippedAdam

from models import DMM
from loss import Analytic_ELBO


def derivative(f,a,method='central',h=0.01):

    if method == 'central':
        return (f(a + h) - f(a - h))/(2*h)
    elif method == 'forward':
        return (f(a + h) - f(a))/h
    elif method == 'backward':
        return (f(a) - f(a - h))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")

def data_proccessing(data_dict, mode = 1):
    
    data = sio.loadmat(data_dict)
    
    if mode == 1:
        data_in = data["V2"]
    elif mode == 2:
        data_in = data["V1"]
    else:
        raise ValueError
        
    ns = 1
    data_in = data_in[:,::ns]
    
    fs0 = 10**7/2**14
    fs = fs0/ns
    dt = 1/fs
    
    data_train = data_in[0,40000:100000]
    data_test = data_in[0,0:40000]
    
    d_data_train = np.insert( np.diff(data_train) / dt,0,0 )
    d_data_test = np.insert( np.diff(data_test) / dt,0,0 )
    
    d_data_train = d_data_train.reshape(600,100,1)
    d_data_test = d_data_test.reshape(400,100,1)

    data_train = data_train.reshape(600,100,1)
    data_train = data_train.astype(np.float32)
   
    data_test = data_test.reshape(400,100,1)
    data_test = data_test.astype(np.float32)
   
    data_test, data_val = data_test, data_train  
    
    return data_train, data_test, data_val, d_data_train, d_data_test
    
    
# setup, training, and evaluation
def main(args,data_dict):
    # setup logging
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=args.log, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logging.info(args)

    writer = SummaryWriter()

    data_train, data_test, data_val,  d_data_train, d_data_test = data_proccessing(data_dict, mode = 1)
    
    N_re = np.size(data_train,0)
    N_len = np.size(data_train,1)
    training_seq_lengths = N_len * np.ones((N_re,), dtype = int)
    
    N_re_test = np.size(data_test,0)
    N_len_test = np.size(data_test,1)
    test_seq_lengths = N_len_test * np.ones((N_re_test,), dtype = int)
    
    N_re_val = np.size(data_val,0)
    N_len_val = np.size(data_val,1)
    val_seq_lengths = N_len_val * np.ones((N_re_val,), dtype = int)
    
    data_train_u, data_test_u, data_val_u, _, _ = data_proccessing(data_dict, mode = 2)
    
    training_seq_lengths = torch.tensor(training_seq_lengths)
    training_data_sequences = torch.tensor(data_train)
    training_data_sequences_u = torch.tensor(data_train_u)

    val_seq_lengths = torch.tensor(val_seq_lengths)
    val_data_sequences = torch.tensor(data_val)
    val_data_sequences_u = torch.tensor(data_val_u)

    test_seq_lengths = torch.tensor(test_seq_lengths)
    test_data_sequences = torch.tensor(data_test)
    test_data_sequences_u = torch.tensor(data_test_u)
    
    N_train_data = len(training_seq_lengths)
    N_train_time_slices = float(torch.sum(training_seq_lengths))
    N_mini_batches = int(N_train_data / args.mini_batch_size +
                         int(N_train_data % args.mini_batch_size > 0))

    logging.info("N_train_data: %d     avg. training seq. length: %.2f    N_mini_batches: %d" %
                 (N_train_data, training_seq_lengths.float().mean(), N_mini_batches))

    # how often we do validation/test evaluation during training
    val_test_frequency = 100
    # the number of samples we use to do the evaluation
    n_eval_samples = 1

    # package repeated copies of val/test data for faster evaluation
    # (i.e. set us up for vectorization)
    def rep(x):
        rep_shape = torch.Size([x.size(0) * n_eval_samples]) + x.size()[1:]
        repeat_dims = [1] * len(x.size())
        repeat_dims[0] = n_eval_samples
        return x.repeat(repeat_dims).reshape(n_eval_samples, -1).transpose(1, 0).reshape(rep_shape)

    # get the validation/test data ready for the dmm: pack into sequences, etc.
    val_seq_lengths = rep(val_seq_lengths)
    test_seq_lengths = rep(test_seq_lengths)
    val_batch, val_batch_reversed, val_batch_mask, val_seq_lengths = poly.get_mini_batch(
        torch.arange(n_eval_samples * val_data_sequences.shape[0]), rep(val_data_sequences),
        val_seq_lengths, cuda=args.cuda)
    test_batch, test_batch_reversed, test_batch_mask, test_seq_lengths = poly.get_mini_batch(
        torch.arange(n_eval_samples * test_data_sequences.shape[0]), rep(test_data_sequences),
        test_seq_lengths, cuda=args.cuda)
    val_batch_u, val_batch_reversed_u, val_batch_mask_u, val_seq_lengths_u = poly.get_mini_batch(
        torch.arange(n_eval_samples * val_data_sequences_u.shape[0]), rep(val_data_sequences_u),
        val_seq_lengths, cuda=args.cuda)
    test_batch_u, test_batch_reversed_u, test_batch_mask_u, test_seq_lengths_u = poly.get_mini_batch(
        torch.arange(n_eval_samples * test_data_sequences_u.shape[0]), rep(test_data_sequences_u),
        test_seq_lengths, cuda=args.cuda)
        
    # instantiate the dmm
    dmm = DMM(input_dim=args.input_dim, z_dim=args.z_dim, u_dim=args.u_dim, emission_dim=args.emission_dim,
                  transition_dim=args.transition_dim, rnn_dim=args.rnn_dim,
                  rnn_dropout_rate=args.rnn_dropout_rate,
                  num_iafs=args.num_iafs, iaf_dim=args.iaf_dim, alpha = args.alpha,
                  use_cuda=args.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dmm.to(device)

    # setup optimizer
    adam_params = {"lr": args.learning_rate, "betas": (args.beta1, args.beta2),
                   "clip_norm": args.clip_norm, "lrd": args.lr_decay,
                   "weight_decay": args.weight_decay}
    adam = ClippedAdam(adam_params)
               
    svi = SVI(dmm.model, dmm.guide, adam, 
           loss = Analytic_ELBO()  # Analytic ELBO for Gaussian distributions
          )

    # now we're going to define some functions we need to form the main training loop

    # saves the model and optimizer states to disk
    def save_checkpoint():
        state = {'net':dmm.state_dict(),'epoch':epoch}
        logging.info("saving model to %s..." % args.save_model)
        torch.save(state, args.save_model)
        logging.info("saving optimizer states to %s..." % args.save_opt)
        adam.save(args.save_opt)
        logging.info("done saving model and optimizer checkpoints to disk.")

    # loads the model and optimizer states from disk
    def load_checkpoint():
        assert exists(args.load_opt) and exists(args.load_model), \
            "--load-model and/or --load-opt misspecified"
        checkpoint = torch.load(args.load_model)
        logging.info("loading model from %s..." % args.load_model)
        dmm.load_state_dict(checkpoint['net'])
        logging.info("loading optimizer states from %s..." % args.load_opt)
        adam.load(args.load_opt)
        args.start_epoch = checkpoint['epoch'] + 1
        logging.info("done loading model and optimizer states.")

    # prepare a mini-batch and take a gradient step to minimize -elbo
    def process_minibatch(epoch, which_mini_batch, shuffled_indices):
        if args.annealing_epochs > 0 and epoch < args.annealing_epochs and args.resume == False:
            # compute the KL annealing factor approriate for the current mini-batch in the current epoch
            min_af = args.minimum_annealing_factor
            annealing_factor = min_af + (1.0 - min_af) * \
                               (float(which_mini_batch + epoch * N_mini_batches + 1) /
                                float(args.annealing_epochs * N_mini_batches))
        else:
            # by default the KL annealing factor is unity
            annealing_factor = 1.0

        # compute which sequences in the training set we should grab
        mini_batch_start = (which_mini_batch * args.mini_batch_size)
        mini_batch_end = np.min([(which_mini_batch + 1) * args.mini_batch_size, N_train_data])
        mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
        # grab a fully prepped mini-batch using the helper function in the data loader
        mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
            = poly.get_mini_batch(mini_batch_indices, training_data_sequences,
                                  training_seq_lengths, cuda=args.cuda)
        mini_batch_u, mini_batch_u_reversed, mini_batch_u_mask, mini_batch_u_seq_lengths \
            = poly.get_mini_batch(mini_batch_indices, training_data_sequences_u,
                                  training_seq_lengths, cuda=args.cuda)
        # do an actual gradient step
        loss = svi.step(mini_batch_u, mini_batch, mini_batch_reversed, mini_batch_mask,
                        mini_batch_seq_lengths, annealing_factor)

        # keep track of the training loss
        return loss

    # helper function for doing evaluation
    def do_evaluation():
        # put the RNN into evaluation mode (i.e. turn off drop-out if applicable)
        dmm.rnn.eval()

        # compute the validation and test loss n_samples many times
        val_nll = svi.evaluate_loss(val_batch_u, val_batch, val_batch_reversed, val_batch_mask,
                                    val_seq_lengths) / float(torch.sum(val_seq_lengths))
        test_nll = svi.evaluate_loss(test_batch_u, test_batch, test_batch_reversed, test_batch_mask,
                                     test_seq_lengths) / float(torch.sum(test_seq_lengths))

        # put the RNN back into training mode (i.e. turn on drop-out if applicable)
        dmm.rnn.train()
        return val_nll, test_nll

    # if checkpoint files provided, load model and optimizer states from disk before we start training
    if args.load_opt != '' and args.load_model != '':
        load_checkpoint()

    # helper function for plotting
    def do_plot(dmm, training_data_sequences,training_data_sequences_u, test_data_sequences,test_data_sequences_u, n_re,n_epoch,z_dim = args.z_dim):
        
        data_x, data_x_test, _, data_v, data_v_test  = data_proccessing(data_dict, mode = 1)

        plt.rc('font', family='Times New Roman')
        training_data_sequences = training_data_sequences.to(device)
        training_data_sequences_u = training_data_sequences_u.to(device)
        test_data_sequences = test_data_sequences.to(device)
        test_data_sequences_u = test_data_sequences_u.to(device)
        N_re = training_data_sequences.size(0)
        N_len = training_data_sequences.size(1)
        mm =  training_data_sequences.size(2)
        
        N_re_test = test_data_sequences.size(0)
        N_len_test = test_data_sequences.size(1)
        
        data_in = np.reshape(data_x, (N_re*N_len,))
        data_in_test = np.reshape(data_x_test, (N_re_test*N_len_test,))
        V = np.reshape(data_v, (N_re*N_len,))
        V_test = np.reshape(data_v_test, (N_re_test*N_len_test,))
        
        h_0_contig = dmm.h_0.expand(2, training_data_sequences.size(0), dmm.rnn.hidden_size).contiguous() 
        rnn_output, _ = dmm.rnn(training_data_sequences, h_0_contig)
        rnn_output = rnn_output.view(rnn_output.size(0),rnn_output.size(1),
                                          2,dmm.rnn.hidden_size)
        h1 = rnn_output[:,:,0,:]
        h2 = rnn_output[:,:,1,:]

        h_0_contig_test = dmm.h_0.expand(2, test_data_sequences.size(0), dmm.rnn.hidden_size).contiguous() 
        rnn_output_test, _ = dmm.rnn(test_data_sequences, h_0_contig_test)
        rnn_output_test = rnn_output_test.view(rnn_output_test.size(0),rnn_output_test.size(1),
                                          2,dmm.rnn.hidden_size)
        h1_test = rnn_output_test[:,:,0,:]
        h2_test = rnn_output_test[:,:,1,:]

        z_prev_pos_0 = dmm.z_q_0_0.expand(N_re,z_dim)      # initializing z_q_0_0
        z_prev_pos_1 = dmm.z_q_0_1.expand(N_re,z_dim)      # initializing z_q_0_1
        z_prev_pri_0 = dmm.z_0_0.expand(N_re,z_dim)
        z_prev_pri_1 = dmm.z_0_1.expand(N_re,z_dim)
        
        Z_loc_pos, Z_loc_pri =  torch.zeros(N_re,N_len,z_dim), torch.zeros(N_re,N_len,z_dim)
        Z_scale_pos, Z_scale_pri = torch.zeros(N_re,N_len,z_dim), torch.zeros(N_re,N_len,z_dim)
        
        Z_loc_pri_0, Z_scale_pri_0 = torch.zeros(N_re,N_len,z_dim), torch.zeros(N_re,N_len,z_dim)
        
        Obs_loc_pos, Obs_loc_pri = torch.zeros(N_re,N_len,mm), torch.zeros(N_re,N_len,mm)
        Obs_scale_pos, Obs_scale_pri = torch.zeros(N_re,N_len,mm), torch.zeros(N_re,N_len,mm)
        
        
        z_prev_pos_0_test = dmm.z_q_0_0.expand(N_re_test,z_dim)      # initializing z_q_0_0
        z_prev_pos_1_test = dmm.z_q_0_1.expand(N_re_test,z_dim)      # initializing z_q_0_1
        z_prev_pri_0_test = dmm.z_0_0.expand(N_re_test,z_dim)
        z_prev_pri_1_test = dmm.z_0_1.expand(N_re_test,z_dim)
        
        Z_loc_pos_test, Z_loc_pri_test =  torch.zeros(N_re_test,N_len_test,z_dim), torch.zeros(N_re_test,N_len_test,z_dim)
        Z_scale_pos_test, Z_scale_pri_test = torch.zeros(N_re_test,N_len_test,z_dim), torch.zeros(N_re_test,N_len_test,z_dim)
        
        Z_loc_pri_0_test, Z_scale_pri_0_test = torch.zeros(N_re_test,N_len_test,z_dim), torch.zeros(N_re_test,N_len_test,z_dim)
        
        Obs_loc_pos_test, Obs_loc_pri_test = torch.zeros(N_re_test,N_len_test,mm), torch.zeros(N_re_test,N_len_test,mm)
        Obs_scale_pos_test, Obs_scale_pri_test = torch.zeros(N_re_test,N_len_test,mm), torch.zeros(N_re_test,N_len_test,mm)
        
        for t in range(1, N_len+1):
            
            z_loc_pos_0, z_scale_pos_0 = dmm.combiner_0(z_prev_pos_0, training_data_sequences_u[:,t-1,:], h1[:, t-1, :], h2[:, t-1, :])
            z_loc_pos_1, z_scale_pos_1 = dmm.combiner_1(z_prev_pos_1, training_data_sequences_u[:,t-1,:], h1[:, t-1, :], h2[:, t-1, :])
            z_loc_pos = dmm.alpha * z_loc_pos_0 + (1 - dmm.alpha) * z_loc_pos_1
            z_scale_pos = torch.sqrt(dmm.alpha**2 * z_scale_pos_0**2 + (1 - dmm.alpha)**2 * z_scale_pos_1**2)
            
            z_loc_pri_0, z_scale_pri_0 = dmm.trans_0(z_prev_pri_0, training_data_sequences_u[:,t-1,:])
            z_loc_pri_1, z_scale_pri_1 = dmm.trans_1(z_prev_pri_1, training_data_sequences_u[:,t-1,:]) 
            z_loc_pri = dmm.alpha * z_loc_pri_0 + (1 - dmm.alpha) * z_loc_pri_1
            z_scale_pri = torch.sqrt(dmm.alpha**2 * z_scale_pri_0**2 + (1 - dmm.alpha)**2 * z_scale_pri_1**2)

            obs_loc_pos, obs_scale_pos = dmm.emission(z_loc_pos) 

            z_prev_pos_0 = z_loc_pos_0
            z_prev_pos_1 = z_loc_pos_1
            z_prev_pri_0 = z_loc_pri_0
            z_prev_pri_1 = z_loc_pri_1

            Z_loc_pos[:,t-1,:] = z_loc_pos 
            Z_loc_pri[:,t-1,:] = z_loc_pri           
            Z_scale_pos[:,t-1,:] = z_scale_pos         
            Z_scale_pri[:,t-1,:] = z_scale_pri      
            Z_loc_pri_0[:,t-1,:] = z_loc_pri_0     
            Z_scale_pri_0[:,t-1,:] = z_scale_pri_0
            
            Obs_loc_pos[:,t-1,:] = obs_loc_pos
            Obs_scale_pos[:,t-1,:] = obs_scale_pos


            z_loc_pos_0_test, z_scale_pos_0_test = dmm.combiner_0(z_prev_pos_0_test, test_data_sequences_u[:,t-1,:], h1_test[:, t-1, :], h2_test[:, t-1, :])
            z_loc_pos_1_test, z_scale_pos_1_test = dmm.combiner_1(z_prev_pos_1_test, test_data_sequences_u[:,t-1,:], h1_test[:, t-1, :], h2_test[:, t-1, :])
            z_loc_pos_test = dmm.alpha * z_loc_pos_0_test + (1 - dmm.alpha) * z_loc_pos_1_test
            z_scale_pos_test = torch.sqrt(dmm.alpha**2 * z_scale_pos_0_test**2 + (1 - dmm.alpha)**2 * z_scale_pos_1_test**2)
            
            z_loc_pri_0_test, z_scale_pri_0_test = dmm.trans_0(z_prev_pri_0_test, test_data_sequences_u[:,t-1,:])
            z_loc_pri_1_test, z_scale_pri_1_test = dmm.trans_1(z_prev_pri_1_test, test_data_sequences_u[:,t-1,:]) 
            z_loc_pri_test = dmm.alpha * z_loc_pri_0_test + (1 - dmm.alpha) * z_loc_pri_1_test
            z_scale_pri_test = torch.sqrt(dmm.alpha**2 * z_scale_pri_0_test**2 + (1 - dmm.alpha)**2 * z_scale_pri_1_test**2)

            obs_loc_pos_test, obs_scale_pos_test = dmm.emission(z_loc_pos_test) 
            z_prev_pos_0_test = z_loc_pos_0_test
            z_prev_pos_1_test = z_loc_pos_1_test
            z_prev_pri_0_test = z_loc_pri_0_test
            z_prev_pri_1_test = z_loc_pri_1_test

            Z_loc_pos_test[:,t-1,:] = z_loc_pos_test
            Z_loc_pri_test[:,t-1,:] = z_loc_pri_test      
            Z_scale_pos_test[:,t-1,:] = z_scale_pos_test     
            Z_scale_pri_test[:,t-1,:] = z_scale_pri_test 
            Z_loc_pri_0_test[:,t-1,:] = z_loc_pri_0_test   
            Z_scale_pri_0_test[:,t-1,:] = z_scale_pri_0_test
            
            Obs_loc_pos_test[:,t-1,:] = obs_loc_pos_test
            Obs_scale_pos_test[:,t-1,:] = obs_scale_pos_test
           
            
        Ylabels = ["displacement","velocity"]
        
        fig1 = plt.figure(figsize=(40,40))        
        plt.ioff()
        for i in range(2):   
            ax = plt.subplot(2,1,i+1)
            latent_est = torch.reshape(Z_loc_pos[:,:,i],(N_re*N_len,))
            latent_est_scale = torch.reshape(Z_scale_pos[:,:,i],(N_re*N_len,))
            
            if i == 0:
                plt.plot(data_in, color = "silver", label = "ground-truth", lw = 20)
                plt.plot(latent_est.data-data_in, color='orange',label = "error", lw=2)
            elif i == 1:
                plt.plot(V, color = "silver", label = "ground-truth", lw = 20)
                plt.plot(latent_est.data-V, color='orange',label = "error", lw=2)
            plt.plot(latent_est.data, color='b',label = "inference", lw=2)
            plt.ylabel(Ylabels[i], fontdict={'family' : 'Times New Roman', 'size' : 120})
            plt.xlabel("$t$", fontdict={'family' : 'Times New Roman', 'size' : 120})
            plt.xticks(fontproperties = 'Times New Roman', size=120)
            plt.yticks(fontproperties = 'Times New Roman', size=120)
            if i == 0:
                ax.legend(loc="lower left", bbox_to_anchor= (0.0, 1.1), ncol= 3, prop={'family' : 'Times New Roman', 'size' : 100})
                ax.set_title("Displacement", fontdict={'family' : 'Times New Roman', 'size' : 125})
            if i == 1:
                ax.set_title("Velocity", fontdict={'family' : 'Times New Roman', 'size' : 125})
        plt.tight_layout()
        plt.close()
            
        fig2 = plt.figure(figsize=(40,40))        
        plt.ioff()
        for i in range(2):   
            ax = plt.subplot(2,1,i+1)
            latent_est = torch.reshape(Z_loc_pri[:,:,i],(N_re*N_len,))
            latent_est_scale = torch.reshape(Z_scale_pri[:,:,i],(N_re*N_len,))
            
            if i == 0:
                plt.plot(data_in, color = "silver", label = "ground-truth", lw = 20)
            elif i == 1:
                plt.plot(V, color = "silver", label = "ground-truth", lw = 20)
            plt.plot(latent_est.data, color='b',label = "inference", lw=2)
            plt.ylabel(Ylabels[i], fontdict={'family' : 'Times New Roman', 'size' : 120})
            plt.xlabel("$t$", fontdict={'family' : 'Times New Roman', 'size' : 120})
            plt.xticks(fontproperties = 'Times New Roman', size=120)
            plt.yticks(fontproperties = 'Times New Roman', size=120)     
            if i == 0:
                ax.legend(loc="lower left", bbox_to_anchor= (0.0, 1.1), ncol= 3, prop={'family' : 'Times New Roman', 'size' : 100})
                ax.set_title("Displacement", fontdict={'family' : 'Times New Roman', 'size' : 125})
            if i == 1:
                ax.set_title("Velocity", fontdict={'family' : 'Times New Roman', 'size' : 125})
        plt.tight_layout()
        plt.close()
            
    
        Ylabels_m = ["displacement","velocity"]

        
        
        fig4 = plt.figure(figsize = (40,20))
        plt.ioff()
        for i in range(mm):
            obs_est_loc = torch.reshape(Obs_loc_pos[:,:,i],(N_re*N_len,))
            obs_est_scale = torch.reshape(Obs_scale_pos[:,:,i],(N_re*N_len,))
            obs_measured = np.reshape(data_x[:,:,i],(N_re*N_len,))
            
            plt.subplot(mm,1,i+1)
            plt.plot(obs_measured, color = "silver", lw = 20, label = "ground-truth")
            plt.plot(obs_est_loc.data, color = 'b', label = "inference observation", lw=2)
            plt.xlabel("$t$", fontdict={'family' : 'Times New Roman', 'size' : 120})
            plt.ylabel(Ylabels_m[i], fontdict={'family' : 'Times New Roman', 'size' : 120})
            plt.xticks(fontproperties = 'Times New Roman', size=120)
            plt.yticks(fontproperties = 'Times New Roman', size=120)
            if i == 0:
                ax.legend(loc="lower left", bbox_to_anchor= (0.0, 1.1), ncol= 3, prop={'family' : 'Times New Roman', 'size' : 100})
                ax.set_title("Displacement", fontdict={'family' : 'Times New Roman', 'size' : 125})
            if i == 1:
                ax.set_title("Velocity", fontdict={'family' : 'Times New Roman', 'size' : 125})
        plt.tight_layout()
        plt.close()

        fig5 = plt.figure(figsize=(60,20))
        plt.ioff()
        latent_x = torch.reshape(Z_loc_pos[:,:,0],(N_re*N_len,))
        latent_v = torch.reshape(Z_loc_pos[:,:,1],(N_re*N_len,))
        
        latent_x_trans = torch.reshape(Z_loc_pri[:,:,0],(N_re*N_len,))
        latent_v_trans = torch.reshape(Z_loc_pri[:,:,1],(N_re*N_len,))
        
        ax = plt.subplot(1,3,1)
        plt.plot(latent_x.data, latent_v.data,'-',lw=1, label = "inference")
        plt.xlabel("inferred displacement", fontdict={'family' : 'Times New Roman', 'size' : 80})
        plt.ylabel("inferred velocity", fontdict={'family' : 'Times New Roman', 'size' : 80})
        plt.xticks(fontproperties = 'Times New Roman', size=80)
        plt.yticks(fontproperties = 'Times New Roman', size=80) 
        ax.legend(loc="lower left", bbox_to_anchor= (0.0, 1.01), ncol= 3, prop={'family' : 'Times New Roman', 'size' : 100})
        
        ax = plt.subplot(1,3,2)
        plt.plot(latent_x_trans.data, latent_v_trans.data,'-',lw=1, label = "generative")
        plt.xlabel("generative displacement", fontdict={'family' : 'Times New Roman', 'size' : 80})
        plt.ylabel(" velocity", fontdict={'family' : 'Times New Roman', 'size' : 80})
        plt.xticks(fontproperties = 'Times New Roman', size=80)
        plt.yticks(fontproperties = 'Times New Roman', size=80) 
        ax.legend(loc="lower left", bbox_to_anchor= (0.0, 1.01), ncol= 3, prop={'family' : 'Times New Roman', 'size' : 100})
        
        ax = plt.subplot(1,3,3)
        plt.plot(data_in, V,'-',lw=1, label = "ground-truth")
        plt.xlabel("true displacement", fontdict={'family' : 'Times New Roman', 'size' : 80})
        plt.ylabel("true velocity", fontdict={'family' : 'Times New Roman', 'size' : 80})
        plt.xticks(fontproperties = 'Times New Roman', size=80)
        plt.yticks(fontproperties = 'Times New Roman', size=80) 
        ax.legend(loc="lower left", bbox_to_anchor= (0.0, 1.01), ncol= 3, prop={'family' : 'Times New Roman', 'size' : 100})
        plt.tight_layout()
        plt.close()
        
        reg_1 = LinearRegression(fit_intercept=True)
        reg_2 = LinearRegression(fit_intercept=True)
        res_1 = reg_1.fit(data_in.reshape(-1,1), latent_x.data)
        res_2 = reg_2.fit(V.reshape(-1,1), latent_v.data)
        R_square_1 = reg_1.score(data_in.reshape(-1,1), latent_x.data)
        R_square_2 = reg_2.score(V.reshape(-1,1), latent_v.data)
        RMSE_1 = np.sqrt(mean_squared_error(data_in.reshape(-1,1), latent_x.data))
        RMSE_2 = np.sqrt(mean_squared_error(V.reshape(-1,1), latent_v.data))
        
        fig6 = plt.figure(figsize=(40,20))
        plt.ioff()
        latent_x = torch.reshape(Z_loc_pos[:,:,0],(N_re*N_len,))
        latent_v = torch.reshape(Z_loc_pos[:,:,1],(N_re*N_len,))
        ax = plt.subplot(1,2,1)
        plt.scatter(data_in, latent_x.data)
        plt.plot(data_in,reg_1.predict(data_in.reshape(-1,1)),"r-",lw=2)
        plt.xlabel("true displacement", fontdict={'family' : 'Times New Roman', 'size' : 120})
        plt.ylabel("inferred displacement", fontdict={'family' : 'Times New Roman', 'size' : 120})
        plt.xticks(fontproperties = 'Times New Roman', size=100)
        plt.yticks(fontproperties = 'Times New Roman', size=100) 
        ax.set_title("Displacement", fontdict={'family' : 'Times New Roman', 'size' : 125})
        ax = plt.subplot(1,2,2)
        plt.scatter(V, latent_v.data)
        plt.plot(V,reg_2.predict(V.reshape(-1,1)),"r-",lw=2)
        plt.xlabel("true velocity", fontdict={'family' : 'Times New Roman', 'size' : 120})
        plt.ylabel("inferred velocity", fontdict={'family' : 'Times New Roman', 'size' : 120})
        plt.xticks(fontproperties = 'Times New Roman', size=100)
        plt.yticks(fontproperties = 'Times New Roman', size=100) 
        ax.set_title("Velocity", fontdict={'family' : 'Times New Roman', 'size' : 125})
        plt.tight_layout()
        plt.close()
        
        fig7 = plt.figure(figsize=(40,40))      
        plt.ioff()
        for i in range(2):   
            ax = plt.subplot(2,1,i+1)
            latent_est_test = torch.reshape(Z_loc_pos_test[:,:,i],(N_re_test*N_len_test,))
            latent_est_scale_test = torch.reshape(Z_scale_pos_test[:,:,i],(N_re_test*N_len_test,))
            
            if i == 0:
                plt.plot(data_in_test, color = "silver", label = "ground-truth", lw = 20)
            elif i == 1:
                plt.plot(V_test.T, color = "silver", label = "ground-truth", lw = 20)
            plt.plot(latent_est_test.data, color='b',label = "inference", lw=2)
            
            plt.ylabel(Ylabels[i], fontdict={'family' : 'Times New Roman', 'size' : 120})
            plt.xlabel("$t$", fontdict={'family' : 'Times New Roman', 'size' : 120})
            plt.xticks(fontproperties = 'Times New Roman', size=120)
            plt.yticks(fontproperties = 'Times New Roman', size=120)
            if i == 0:
                ax.legend(loc="lower left", bbox_to_anchor= (0.0, 1.1), ncol= 3, prop={'family' : 'Times New Roman', 'size' : 100})
                ax.set_title("Displacement", fontdict={'family' : 'Times New Roman', 'size' : 125})
            if i == 1:
                ax.set_title("Velocity", fontdict={'family' : 'Times New Roman', 'size' : 125})                
        plt.tight_layout()
        plt.close()

        fig3 = plt.figure(figsize=(100,60))
        plt.ioff()
        Ylabels_m = ["displacement","velocity"]
        for i in range(4):
            gs = gridspec.GridSpec(2, 2, width_ratios=[3.25, 1]) 
            ax = plt.subplot(gs[i])
            if i == 0 or 2:
                if i == 0:
                    plt.plot(data_x[n_re,:,0], color = "silver", label = "ground-truth (displacement)", lw = 30)
                    temp = np.max(np.abs(data_x[n_re,:,0]))
                    plt.plot(Z_loc_pos[n_re,:,0].data, color = "b",label = "inference ($z_1$)", lw=8)
                    plt.ylim([-1.5*temp, 1.5*temp])
                    plt.xlabel("$t$", fontdict={'family' : 'Times New Roman', 'size' : 130})
                    plt.xticks(fontproperties = 'Times New Roman', size=135)
                    plt.yticks(fontproperties = 'Times New Roman', size=135) 
                    ax.legend(loc="lower left", bbox_to_anchor= (0.0, 1.1), ncol= 3, prop={'family' : 'Times New Roman', 'size' : 120})
                    ax.set_title("Displacement", fontdict={'family' : 'Times New Roman', 'size' : 140})
                if i == 2:
                    plt.plot(data_v[n_re,:,0], color = "silver", label = "ground-truth (velocity)", lw = 30)
                    temp = np.max(np.abs(data_v[n_re,:,0]))
                    plt.plot(Z_loc_pos[n_re,:,1].data, color = "b",label = "inference ($z_2$)", lw=8)
                    plt.ylim([-1.5*temp, 1.5*temp])
                    plt.xlabel("$t$", fontdict={'family' : 'Times New Roman', 'size' : 130})
                    plt.xticks(fontproperties = 'Times New Roman', size=135)
                    plt.yticks(fontproperties = 'Times New Roman', size=135) 
                    ax.legend(loc="lower left", bbox_to_anchor= (0.0, 1.1), ncol= 3, prop={'family' : 'Times New Roman', 'size' : 120})
                    ax.set_title("Velocity", fontdict={'family' : 'Times New Roman', 'size' : 140})
            if i == 1 or 3:
                latent_x = torch.reshape(Z_loc_pos[:,:,0],(N_re*N_len,))
                latent_v = torch.reshape(Z_loc_pos[:,:,1],(N_re*N_len,))
                if i == 1:
                    plt.scatter(data_in, latent_x.data)
                    plt.plot(data_in,reg_1.predict(data_in.reshape(-1,1)),"r-",lw=4)
                    plt.xlabel("true displacement", fontdict={'family' : 'Times New Roman', 'size' : 145})
                    plt.ylabel("$z_1$", fontdict={'family' : 'Times New Roman', 'size' : 145})
                    plt.xticks(fontproperties = 'Times New Roman', size=130)
                    plt.yticks(fontproperties = 'Times New Roman', size=130) 
                if i == 3:
                    plt.scatter(V, latent_v.data)
                    plt.plot(V,reg_2.predict(V.reshape(-1,1)),"r-",lw=4)
                    plt.xlabel("true velocity", fontdict={'family' : 'Times New Roman', 'size' : 145})
                    plt.ylabel("$z_2$", fontdict={'family' : 'Times New Roman', 'size' : 145})
                    plt.xticks(fontproperties = 'Times New Roman', size=130)
                    plt.yticks(fontproperties = 'Times New Roman', size=130) 
        plt.tight_layout()
        plt.close()
        
        latent_x_test = torch.reshape(Z_loc_pos_test[:,:,0],(N_re_test*N_len_test,))
        latent_v_test = torch.reshape(Z_loc_pos_test[:,:,1],(N_re_test*N_len_test,))
        
        reg_1_test = LinearRegression(fit_intercept=True)
        reg_2_test = LinearRegression(fit_intercept=True)
        res_1_test = reg_1_test.fit(data_in_test.reshape(-1,1), latent_x_test.data)
        res_2_test = reg_2_test.fit(V_test.reshape(-1,1), latent_v_test.data)
        R_square_1_test = reg_1_test.score(data_in_test.reshape(-1,1), latent_x_test.data)
        R_square_2_test = reg_2_test.score(V_test.reshape(-1,1), latent_v_test.data)
        RMSE_1_test = np.sqrt(mean_squared_error(data_in_test.reshape(-1,1), latent_x_test.data))
        RMSE_2_test = np.sqrt(mean_squared_error(V_test.reshape(-1,1), latent_v_test.data))

        fig8 = plt.figure(figsize=(100,60))
        plt.ioff()
        Ylabels_m = ["displacement","velocity"]
        for i in range(4):
            gs = gridspec.GridSpec(2, 2, width_ratios=[3.25, 1]) 
            ax = plt.subplot(gs[i])
            if i == 0 or 2:
                if i == 0:
                    plt.plot(data_x_test[n_re,:,0], color = "silver", label = "ground-truth (displacement)", lw = 30)
                    temp = np.max(np.abs(data_x_test[n_re,:,0]))
                    plt.plot(Z_loc_pos_test[n_re,:,0].data, color = "b",label = "inference ($z_1$)", lw=8)
                    plt.ylim([-1.5*temp, 1.5*temp])
                    plt.xlabel("$t$", fontdict={'family' : 'Times New Roman', 'size' : 130})
                    plt.xticks(fontproperties = 'Times New Roman', size=135)
                    plt.yticks(fontproperties = 'Times New Roman', size=135) 
                    ax.legend(loc="lower left", bbox_to_anchor= (0.0, 1.1), ncol= 3, prop={'family' : 'Times New Roman', 'size' : 120})
                    ax.set_title("Displacement", fontdict={'family' : 'Times New Roman', 'size' : 140})
                if i == 2:
                    plt.plot(data_v_test[n_re,:,0], color = "silver", label = "ground-truth (velocity)", lw = 30)
                    temp = np.max(np.abs(data_v_test[n_re,:,0]))
                    plt.plot(Z_loc_pos_test[n_re,:,1].data, color = "b",label = "inference ($z_2$)", lw=8)
                    plt.ylim([-1.5*temp, 1.5*temp])
                    plt.xlabel("$t$", fontdict={'family' : 'Times New Roman', 'size' : 130})
                    plt.xticks(fontproperties = 'Times New Roman', size=135)
                    plt.yticks(fontproperties = 'Times New Roman', size=135) 
                    ax.legend(loc="lower left", bbox_to_anchor= (0.0, 1.1), ncol= 3, prop={'family' : 'Times New Roman', 'size' : 120})
                    ax.set_title("Velocity", fontdict={'family' : 'Times New Roman', 'size' : 140})
            if i == 1 or 3:
                latent_x_test = torch.reshape(Z_loc_pos_test[:,:,0],(N_re_test*N_len_test,))
                latent_v_test = torch.reshape(Z_loc_pos_test[:,:,1],(N_re_test*N_len_test,))
                if i == 1:
                    plt.scatter(data_in_test, latent_x_test.data)
                    plt.plot(data_in_test,reg_1_test.predict(data_in_test.reshape(-1,1)),"r-",lw=4)
                    plt.xlabel("true displacement", fontdict={'family' : 'Times New Roman', 'size' : 145})
                    plt.ylabel("$z_1$", fontdict={'family' : 'Times New Roman', 'size' : 145})
                    plt.xticks(fontproperties = 'Times New Roman', size=130)
                    plt.yticks(fontproperties = 'Times New Roman', size=130) 
                if i == 3:
                    plt.scatter(V_test, latent_v_test.data)
                    plt.plot(V_test,reg_2_test.predict(V_test.reshape(-1,1)),"r-",lw=4)
                    plt.xlabel("true velocity", fontdict={'family' : 'Times New Roman', 'size' : 145})
                    plt.ylabel("$z_2$", fontdict={'family' : 'Times New Roman', 'size' : 145})
                    plt.xticks(fontproperties = 'Times New Roman', size=130)
                    plt.yticks(fontproperties = 'Times New Roman', size=130) 
        plt.tight_layout()
        plt.close()

        fig9 = plt.figure(figsize=(40,20))
        plt.ioff()
        latent_x_test = torch.reshape(Z_loc_pos_test[:,:,0],(N_re_test*N_len_test,))
        latent_v_test = torch.reshape(Z_loc_pos_test[:,:,1],(N_re_test*N_len_test,))
        ax = plt.subplot(1,2,1)
        plt.scatter(data_in_test, latent_x_test.data)
        plt.plot(data_in_test,reg_1_test.predict(data_in_test.reshape(-1,1)),"r-",lw=2)
        plt.xlabel("true displacement", fontdict={'family' : 'Times New Roman', 'size' : 120})
        plt.ylabel("inferred displacement", fontdict={'family' : 'Times New Roman', 'size' : 120})
        plt.xticks(fontproperties = 'Times New Roman', size=100)
        plt.yticks(fontproperties = 'Times New Roman', size=100) 
        ax.set_title("Displacement", fontdict={'family' : 'Times New Roman', 'size' : 125})
        ax = plt.subplot(1,2,2)
        plt.scatter(V_test, latent_v_test.data)
        plt.plot(V_test,reg_2_test.predict(V_test.reshape(-1,1)),"r-",lw=2)
        plt.xlabel("true velocity", fontdict={'family' : 'Times New Roman', 'size' : 120})
        plt.ylabel("inferred velocity", fontdict={'family' : 'Times New Roman', 'size' : 120})
        plt.xticks(fontproperties = 'Times New Roman', size=100)
        plt.yticks(fontproperties = 'Times New Roman', size=100) 
        ax.set_title("Velocity", fontdict={'family' : 'Times New Roman', 'size' : 125})
        plt.tight_layout()
        plt.close()

        fig10 = plt.figure(figsize=(40,28))
        plt.ioff()
        for i in range(2):
            ax = plt.subplot(2,1,i+1)
            if i == 0:
                plt.plot(data_v[n_re,:,0], data_x[n_re,:,0],color='silver',lw=10,label = "ground-truth")
                plt.xlabel("velocity", fontdict={'family' : 'Times New Roman', 'size' : 120})
                plt.ylabel("displacement", fontdict={'family' : 'Times New Roman', 'size' : 120})
            elif i == 1:
                plt.plot(Z_loc_pos[n_re,:,1].data, Z_loc_pos[n_re,:,0].data,color='blue',lw=10,label = "inference")
                plt.xlabel("$z_2$", fontdict={'family' : 'Times New Roman', 'size' : 120})
                plt.ylabel("$z_1$", fontdict={'family' : 'Times New Roman', 'size' : 120})
            plt.xticks(fontproperties = 'Times New Roman', size=100)
            plt.yticks([-0.2,-0.1,0,0.1,0.2],fontproperties = 'Times New Roman', size=100)
            plt.xlim([-65, 65])
            plt.ylim([-0.2, 0.2])
            if i == 0:
                ax.set_title("Ground-truth", fontdict={'family' : 'Times New Roman', 'size' : 125})
            if i == 1:
                ax.set_title("Inference", fontdict={'family' : 'Times New Roman', 'size' : 125})
        plt.tight_layout()
        plt.close()

        fig11 = plt.figure(figsize=(40,28))
        plt.ioff()
        for i in range(2):
            ax = plt.subplot(2,1,i+1)
            if i == 0:
                plt.plot(data_v_test[n_re,:,0], data_x_test[n_re,:,0],color='silver',lw=10,label = "ground-truth")
                plt.xlabel("velocity", fontdict={'family' : 'Times New Roman', 'size' : 120})
                plt.ylabel("displacement", fontdict={'family' : 'Times New Roman', 'size' : 120})
            elif i == 1:
                plt.plot(Z_loc_pos_test[n_re,:,1].data, Z_loc_pos_test[n_re,:,0].data,color='blue',lw=10,label = "inference")
                plt.xlabel("$z_2$", fontdict={'family' : 'Times New Roman', 'size' : 120})
                plt.ylabel("$z_1$", fontdict={'family' : 'Times New Roman', 'size' : 120})
            plt.xticks(fontproperties = 'Times New Roman', size=100)
            plt.yticks([-0.2,-0.1,0,0.1,0.2],fontproperties = 'Times New Roman', size=100)
            plt.xlim([-65, 65])
            plt.ylim([-0.2, 0.2])
            if i == 0:
                ax.set_title("Ground-truth", fontdict={'family' : 'Times New Roman', 'size' : 125})
            if i == 1:
                ax.set_title("Inference", fontdict={'family' : 'Times New Roman', 'size' : 125})
        plt.tight_layout()
        plt.close()
        
        return fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9,fig10,fig11,R_square_1,R_square_2,R_square_1_test,R_square_2_test,RMSE_1,RMSE_2,RMSE_1_test,RMSE_2_test

    #################
    # TRAINING LOOP #
    #################
    times = [time.time()]
    #step = 0
    for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
        # accumulator for our estimate of the negative log likelihood (or rather -elbo) for this epoch
        epoch_nll = 0.0
        # prepare mini-batch subsampling indices for this epoch
        shuffled_indices = torch.randperm(N_train_data)

        # process each mini-batch; this is where we take gradient steps
        for which_mini_batch in range(N_mini_batches):
            batch_nll = process_minibatch(epoch, which_mini_batch, shuffled_indices)
            epoch_nll += batch_nll
        
        #step += 1
        writer.add_scalar('loss/training_loss', epoch_nll / N_train_time_slices, epoch)

        # writer.add_scalar('loss/learning_rate', adam.get_state()['dmm$$$akf.k_f']['param_groups'][0]['lr'], step)

        # report training diagnostics
        times.append(time.time())
        epoch_time = times[-1] - times[-2]
        logging.info("[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)" %
                     (epoch, epoch_nll / N_train_time_slices, epoch_time))

        # do evaluation on test and validation data and report results
        if val_test_frequency > 0 and epoch >= 0 and epoch % val_test_frequency == 0:
            val_nll, test_nll = do_evaluation()
            logging.info("[val/test epoch %04d]  %.4f  %.4f" % (epoch, val_nll, test_nll))
            writer.add_scalar('loss/validation_loss', val_nll, epoch)
            writer.add_scalar('loss/testing_loss', test_nll, epoch)
            fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9,fig10,fig11,R_square_1,R_square_2,R_square_1_test,R_square_2_test,RMSE_1,RMSE_2,RMSE_1_test,RMSE_2_test = do_plot(dmm, training_data_sequences,training_data_sequences_u, test_data_sequences,test_data_sequences_u,372,epoch)

            writer.add_figure("latent_space_inference",fig1,global_step = epoch)
            writer.add_figure("latent_space_transition",fig2,global_step = epoch)
            writer.add_figure("observation",fig3,global_step = epoch)
            writer.add_figure("observation_total",fig4,global_step = epoch)
            writer.add_figure("phase_plot",fig5,global_step = epoch)
            writer.add_figure("regression_plot",fig6,global_step = epoch)
            writer.add_figure("latent_space_inference_test",fig7,global_step = epoch)
            writer.add_figure("observation_test",fig8,global_step = epoch)
            writer.add_figure("regression_plot_test",fig9,global_step = epoch)
            writer.add_figure("phase_plots",fig10,global_step = epoch)
            writer.add_figure("phase_plots_test",fig11,global_step = epoch)
            writer.add_scalar('R_square_1', R_square_1, epoch)
            writer.add_scalar('R_square_2', R_square_2, epoch)
            writer.add_scalar('R_square_1_test', R_square_1_test, epoch)
            writer.add_scalar('R_square_2_test', R_square_2_test, epoch)
            writer.add_scalar('RMSE_1', RMSE_1, epoch)
            writer.add_scalar('RMSE_2', RMSE_2, epoch)
            writer.add_scalar('RMSE_1_test', RMSE_1_test, epoch)
            writer.add_scalar('RMSE_2_test', RMSE_2_test, epoch)
            writer.add_scalar('alpha', dmm.alpha.data, epoch)
        # if specified, save model and optimizer states to disk every checkpoint_freq epochs
        if args.checkpoint_freq > 0 and epoch > 0 and epoch % args.checkpoint_freq == 0:
            save_checkpoint()
            
# parse command-line arguments and execute the main method
if __name__ == '__main__':
    # assert pyro.__version__.startswith('1.5.2')
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-in', '--input-dim', type=int, default= 1)  # 5 measurements
    parser.add_argument('-z', '--z-dim', type=int, default=2)  # augmented latent space
    parser.add_argument('-u', '--u-dim', type=int, default=1)
    parser.add_argument('-e', '--emission-dim', type=int, default= 40)
    parser.add_argument('-tr', '--transition-dim', type=int, default=40)
    parser.add_argument('-rnn', '--rnn-dim', type=int, default=100)
    parser.add_argument('-s', '--start-epoch', type=int, default=1)
    parser.add_argument('-n', '--num-epochs', type=int, default=20000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('-b1', '--beta1', type=float, default=0.96)
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-cn', '--clip-norm', type=float, default=50)
    parser.add_argument('-lrd', '--lr-decay', type=float, default=0.99996)
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.0)
    parser.add_argument('-mbs', '--mini-batch-size', type=int, default=200)
    parser.add_argument('-ae', '--annealing-epochs', type=int, default=1000)
    parser.add_argument('-maf', '--minimum-annealing-factor', type=float, default=0.2)
    parser.add_argument('-rdr', '--rnn-dropout-rate', type=float, default=0.1)
    parser.add_argument('-iafs', '--num-iafs', type=int, default=0)
    parser.add_argument('-id', '--iaf-dim', type=int, default=100)
    parser.add_argument('-cf', '--checkpoint-freq', type=int, default= 100)
    parser.add_argument('-lopt', '--load-opt', type=str, default='')
    parser.add_argument('-lmod', '--load-model', type=str, default='')
    parser.add_argument('-sopt', '--save-opt', type=str, default='')
    parser.add_argument('-smod', '--save-model', type=str, default='')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--tmc', action='store_true')
    parser.add_argument('--tmcelbo', action='store_true')
    parser.add_argument('--tmc-num-samples', default=10, type=int)
    parser.add_argument('-l', '--log', type=str, default='dmm.log')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('-a',"--alpha", type = float, default = 0.5 )
    args = parser.parse_args('')
