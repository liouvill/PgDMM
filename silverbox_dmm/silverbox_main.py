import scipy.io as sio

import training as DMM_run
import argparse
import data_loader as poly
import pyro
pyro.enable_validation(True)
pyro.set_rng_seed(10000)

load_models = 0
save_models = 1

if load_models == 1:
    resume = True
    loaded_opt_name = "./saved_opt.pt"
    loaded_model_name = "./saved_model.pt"
else:
    resume = False
    loaded_opt_name = ""
    loaded_model_name = ""
    
if save_models == 1:
    saved_opt_name = "./saved_opt.pt"
    saved_model_name = "./saved_model.pt"
else:
    saved_opt_name = ""
    saved_model_name = ""

parser = argparse.ArgumentParser(description="parse args")
parser.add_argument('-in', '--input-dim', type=int, default= 1)
parser.add_argument('-z', '--z-dim', type=int, default=2)  # augmented latent space
parser.add_argument('-u', '--u-dim', type=int, default=1)
parser.add_argument('-e', '--emission-dim', type=int, default= 50)
parser.add_argument('-tr', '--transition-dim', type=int, default=50)
parser.add_argument('-rnn', '--rnn-dim', type=int, default=100)
parser.add_argument('-s', '--start-epoch', type=int, default=1)
parser.add_argument('-n', '--num-epochs', type=int, default=200000)
parser.add_argument('-lr', '--learning-rate', type=float, default= 1e-3)
parser.add_argument('-b1', '--beta1', type=float, default=0.96)
parser.add_argument('-b2', '--beta2', type=float, default=0.999)
parser.add_argument('-cn', '--clip-norm', type=float, default=50)
parser.add_argument('-lrd', '--lr-decay', type=float, default=0.99996)
parser.add_argument('-wd', '--weight-decay', type=float, default=0.0)
parser.add_argument('-mbs', '--mini-batch-size', type=int, default=600)
parser.add_argument('-ae', '--annealing-epochs', type=int, default=0)
parser.add_argument('-maf', '--minimum-annealing-factor', type=float, default=0.2)
parser.add_argument('-rdr', '--rnn-dropout-rate', type=float, default=0.1)
parser.add_argument('-iafs', '--num-iafs', type=int, default=0)
parser.add_argument('-id', '--iaf-dim', type=int, default=100)
parser.add_argument('-cf', '--checkpoint-freq', type=int, default= 100)
parser.add_argument('-lopt', '--load-opt', type=str, default= loaded_opt_name)
parser.add_argument('-lmod', '--load-model', type=str, default= loaded_model_name)
parser.add_argument('-sopt', '--save-opt', type=str, default= saved_opt_name)
parser.add_argument('-smod', '--save-model', type=str, default= saved_model_name)
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--jit', action='store_true')
parser.add_argument('--tmc', action='store_true')
parser.add_argument('--tmcelbo', action='store_true')
parser.add_argument('--tmc-num-samples', default=10, type=int)
parser.add_argument('-l', '--log', type=str, default='dmm.log')
parser.add_argument('--resume', action='store_true')
parser.add_argument('-a',"--alpha", type = float, default = 0)

args = parser.parse_args()

data_dict = '.\SNLS80mV.mat'
DMM_run.main(args,data_dict)