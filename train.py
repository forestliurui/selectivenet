import argparse
import sys
sys.path.append("..")

from models.catdog_vgg_selectivenet import CatsvsDogVgg as CatsvsDogSelective
from models.cifar10_cnn_selectivenet import cifar10cnn as cifar10Selective_cnn
from models.cifar10_vgg_selectivenet import cifar10vgg as cifar10Selective_vgg
from models.cifar10_vgg_selectivenet_curriculum import cifar10vgg_curr as cifar10Selective_curr_vgg
from models.cifar10_cnn_selectivenet_curriculum import cifar10cnn_curr as cifar10Selective_curr_cnn
from models.cifar10_cnn_selectivenet_self_taught import cifar10cnn_self as cifar10Selective_self_cnn
from models.cifar10_cnn_selectivenet_modified_curriculum import cifar10cnn_modi_curr as cifar10Selective_modi_curr_cnn
from models.cifar10_vgg_selectivenet_modified_curriculum import cifar10vgg_modi_curr as cifar10Selective_modi_curr_vgg
from models.cifar10_vgg_selectivenet_self_taught import cifar10vgg_self as cifar10Selective_self_vgg
from models.cifar10_vgg_selectivenet_modified_self import cifar10vgg_modi_self as cifar10Selective_modi_self_vgg
from models.cifar10_cnn_selectivenet_modified_self import cifar10cnn_modi_self as cifar10Selective_modi_self_cnn
from models.svhn_vgg_selectivenet import SvhnVgg as SVHNSelective
from models.svhn_cnn_selectivenet_curriculum import Svhncnn_curr as SVHNSelective_curr_cnn
from models.svhn_cnn_selectivenet_modified_curriculum import Svhncnn_modi_curr as SVHNSelective_modi_curr_cnn
from models.cifar10_vgg_selectivenet_modified import cifar10vgg_modi as cifar10Selective_modi
from models.cifar10_vgg_selectivenet_modified_veri import cifar10vgg_modi_veri as cifar10Selective_modi_veri
from models.cifar10_svgg_selectivenet import cifar10svgg as cifar10Selective_s
from models.emnist_cnn_selectivenet_curriculum import emnistcnn_curr as EMNISTSelective_curr_cnn
from selectivnet_utils import *

MODELS = {"cifar10_vanilla_cnn": cifar10Selective_cnn,
          "cifar10_vanilla_vgg": cifar10Selective_vgg, 
          "cifar10_curriculum_vgg": cifar10Selective_curr_vgg,
          "cifar10_curriculum_cnn": cifar10Selective_curr_cnn,
          "cifar10_self_cnn": cifar10Selective_self_cnn,
          "cifar10_modi_curr_cnn": cifar10Selective_modi_curr_cnn,
          "cifar10_modi_curr_vgg": cifar10Selective_modi_curr_vgg,
          "cifar10_self_vgg": cifar10Selective_self_vgg,
          "cifar10_modi_self_vgg": cifar10Selective_modi_self_vgg,
          "cifar10_modi": cifar10Selective_modi,
          "cifar10_modi_veri": cifar10Selective_modi_veri,
          "cifar10_s": cifar10Selective_s, 
          "cifar100_vanilla_vgg": cifar10Selective_vgg,
          "cifar100_curriculum_vgg": cifar10Selective_curr_vgg,
          "cifar100_curriculum_cnn": cifar10Selective_curr_cnn,
          "cifar100_self_cnn": cifar10Selective_self_cnn,
          "cifar100_modi_curr_cnn": cifar10Selective_modi_curr_cnn,
          "cifar100_modi_curr_vgg": cifar10Selective_modi_curr_vgg,
          "cifar100_self_vgg": cifar10Selective_self_vgg,
          "cifar100_modi_self_cnn": cifar10Selective_modi_self_cnn,
          "cifar100_modi_self_vgg": cifar10Selective_modi_self_vgg,
          "cifar100_modi": cifar10Selective_modi,
          #"cifar_10_s_modi": cifar10Selective_s_modi,
          "catsdogs": CatsvsDogSelective, 
          "SVHN_vanilla": SVHNSelective,
          "SVHN_curriculum_cnn": SVHNSelective_curr_cnn, 
          "SVHN_modi_curr_cnn": SVHNSelective_modi_curr_cnn, 
          "EMNIST_curriculum_cnn": EMNISTSelective_curr_cnn 
          }



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vanilla')
parser.add_argument('--dataset', type=str, default='cifar10')

parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--baseline', type=str, default='none')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--lamda', type=float, default=32)
parser.add_argument('--random_percent', type=int, default=-1)
parser.add_argument('--random_strategy', type=str, default='feature')
parser.add_argument('--curriculum_strategy', type=str, default='curriculum')
parser.add_argument('--order_strategy', type=str, default='inception')
parser.add_argument('--logfile', type=str, default='training.log')
parser.add_argument('--datapath', type=str, default=None)
parser.add_argument('--repeats', type=int, default=1)

args = parser.parse_args()

print("experiment arguments: {}".format(args))

model_cls = MODELS[args.dataset+"_"+args.model]
exp_name = args.exp_name
baseline_name = args.baseline
logfile = args.logfile
datapath = args.datapath
random_percent = args.random_percent
random_strategy = args.random_strategy

coverages = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]

for repeat in range(args.repeats):
    print("====================repeat {}==============".format(repeat))
    if baseline_name == "none":
        results = train_profile(exp_name, model_cls, coverages, dataset=args.dataset, alpha=args.alpha, beta=args.beta, lamda=args.lamda, random_percent=random_percent, random_strategy=random_strategy, order_strategy=args.order_strategy, logfile=logfile, datapath=datapath, args=args)
    else:
        model = model_cls(train=True,
                          filename="{}_{}.h5".format(exp_name, "mc+"),
                          dataset=args.dataset,
                          alpha=args.alpha,
                          beta=args.beta,
                          lamda = args.lamda,
                          random_percent = random_percent,
                          random_strategy = random_strategy,
                          order_strategy = args.order_strategy,
                          logfile=logfile,
                          datapath=datapath,
                          baseline=True,
                          args=args
                          )
        results = train_profile(exp_name, model_cls, coverages, dataset=args.dataset, model_baseline=model, baseline_name=baseline_name, alpha=args.alpha, beta=args.beta, random_percent=random_percent, random_strategy=random_strategy, order_strategy=args.order_strategy, logfile=logfile, datapath=datapath, args=args)
