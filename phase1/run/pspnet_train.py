import os
import argparse

from pspnet import PSPNet

parser = argparse.ArgumentParser()
parser.add_argument('--id', default=0,type=int)
args = parser.parse_args()

pspnet = PSPNet(DEVICE=args.id)

pspnet.fine_tune()
