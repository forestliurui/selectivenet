import argparse

from models.catdog_vgg_selectivenet import CatsvsDogVgg as CatsvsDogSelective
from models.cifar10_vgg_selectivenet import cifar10vgg as cifar10Selective
from models.svhn_vgg_selectivenet import SvhnVgg as SVHNSelective
from models.cifar10_vgg_selectivenet_modified import cifar10vgg_modi as cifar10Selective_modi
from models.cifar10_svgg_selectivenet import cifar10svgg as cifar10Selective_s
from selectivnet_utils import *

MODELS = {"cifar_10": cifar10Selective, 
          "cifar_10_modi": cifar10Selective_modi,
          "cifar_10_s": cifar10Selective_s, 
          #"cifar_10_s_modi": cifar10Selective_s_modi,
          "catsdogs": CatsvsDogSelective, 
          "SVHN": SVHNSelective}



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar_10')

parser.add_argument('--model_name', type=str, default='test')
parser.add_argument('--baseline', type=str, default='none')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--logfile', type=str, default='training.log')

args = parser.parse_args()

model_cls = MODELS[args.dataset]
model_name = args.model_name
baseline_name = args.baseline
logfile = args.logfile

coverages = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]


if baseline_name == "none":
    results = train_profile(model_name, model_cls, coverages, alpha=args.alpha, logfile=logfile)
else:
    model_baseline = model_cls(train=to_train("{}.h5".format(baseline_name)),
                               filename="{}.h5".format(baseline_name),
                               baseline=True)
    results = train_profile(model_name, model_cls, coverages, model_baseline=model_baseline, alpha=args.alpha, logfile=logfile)
