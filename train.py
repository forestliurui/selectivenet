import argparse
import sys
sys.path.append("..")

from models.catdog_vgg_selectivenet import CatsvsDogVgg as CatsvsDogSelective
from models.cifar10_vgg_selectivenet import cifar10vgg as cifar10Selective
from models.cifar10_vgg_selectivenet_curriculum import cifar10vgg_curr as cifar10Selective_curr
from models.svhn_vgg_selectivenet import SvhnVgg as SVHNSelective
from models.cifar10_vgg_selectivenet_modified import cifar10vgg_modi as cifar10Selective_modi
from models.cifar10_vgg_selectivenet_modified_veri import cifar10vgg_modi_veri as cifar10Selective_modi_veri
from models.cifar10_svgg_selectivenet import cifar10svgg as cifar10Selective_s
from selectivnet_utils import *

MODELS = {"cifar10_vanilla": cifar10Selective, 
          "cifar10_curriculum": cifar10Selective_curr,
          "cifar10_modi": cifar10Selective_modi,
          "cifar10_modi_veri": cifar10Selective_modi_veri,
          "cifar10_s": cifar10Selective_s, 
          "cifar100_vanilla": cifar10Selective,
          "cifar100_curriculum": cifar10Selective_curr,
          "cifar100_modi": cifar10Selective_modi,
          #"cifar_10_s_modi": cifar10Selective_s_modi,
          "catsdogs": CatsvsDogSelective, 
          "SVHN": SVHNSelective}



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
        results = train_profile(exp_name, model_cls, coverages, dataset=args.dataset, alpha=args.alpha, beta=args.beta, lamda=args.lamda, random_percent=random_percent, random_strategy=random_strategy, logfile=logfile, datapath=datapath)
    else:
        model_baseline = model_cls(train=to_train("{}.h5".format(baseline_name)),
                                   filename="{}.h5".format(baseline_name),
                                   baseline=True)
        results = train_profile(exp_name, model_cls, coverages, dataset=args.dataset, model_baseline=model_baseline, alpha=args.alpha, beta=args.beta, random_percent=random_percent, random_strategy=random_strategy, logfile=logfile, datapath=datapath)
