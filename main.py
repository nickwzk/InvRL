import argparse
import sys
import logging
from tiktok.load_data import dataset as ds_tiktok
import numpy as np
import torch

from UltraGCN import UltraGCN
from InvRL import InvRL


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for models')
    parser.add_argument('--dataset', nargs='?', default='tiktok',
                        help='dataset')
    parser.add_argument('--model', nargs='?', default='UltraGCN',
                        help='model type')
    parser.add_argument('--p_emb', nargs='?', default='[0.001,0.0001]',
                        help='lr and reg')
    parser.add_argument('--p_proj', nargs='?', default='[0.001,0.01]',
                        help='lr and reg for W')
    parser.add_argument('--p_embp', nargs='?', default='[0.001,0.0001]',
                        help='lr and reg in predictor')
    parser.add_argument('--p_ctx', nargs='?', default='[0.001,0.1]',
                        help='lr and reg for W in predictor')
    parser.add_argument('--p_w', nargs='?', default='[1,1,1,1]',
                        help='w1, w2, w3, w4')
    parser.add_argument('--feat_dim', type=int, default=64,
                        help='feature dim')
    parser.add_argument('--tolog', type=int, default=1,
                        help='0: output to stdout, 1: output to logfile')
    parser.add_argument('--bsz', type=int, default=512,
                        help='batch size')
    parser.add_argument('--ssz', type=int, default=512,
                        help='size of test samples, including positive and negative samples')
    parser.add_argument('--neg_num', type=int, default=50,
                        help='negative samples each batch')
    parser.add_argument('--neighbor_num', type=int, default=10,
                        help='number of item neighbors')
    parser.add_argument('--num_domains', type=int, default=10,
                        help='number of domains')
    parser.add_argument('--regi', type=float, default=0.0,
                        help='reg for item-item graph')
    parser.add_argument('--device', nargs='?', default='cuda:0',
                        help='device')
    parser.add_argument('--num_epoch', type=int, default=500,
                        help='epoch number')
    parser.add_argument('--epoch', type=int, default=5,
                        help='frequency to evaluate')
    parser.add_argument('--lam', type=float, default=0.1,
                        help='lambda')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='lr2')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='alpha')
    parser.add_argument('--sigma', type=float, default=0.01,
                        help='sigma')
    parser.add_argument('--f_epoch', type=int, default=20,
                        help='frontmodel epochs')
    parser.add_argument('--b_epoch', type=int, default=40,
                        help='backmodel epochs')
    parser.add_argument('--ite', type=int, default=5,
                        help='iterator')
    parser.add_argument('--f_max', type=int, default=10,
                        help='frontmodel iterator')
    parser.add_argument('--reuse', type=int, default=0,
                        help='if reuse past_domains')
    parser.add_argument('--pretrained', type=int, default=0,
                        help='if pretrained')
    parser.add_argument('--wdi', type=int, default=2,
                        help='weight decay bias for item embedding')
    parser.add_argument('--sift', type=int, default=0,
                        help='if sift pos items')
    return parser.parse_args()


args = parse_args()

args.p_emb = eval(args.p_emb)
args.p_embp = eval(args.p_embp)
args.p_ctx = eval(args.p_ctx)
args.p_proj = eval(args.p_proj)
args.p_w = eval(args.p_w)

if args.tolog == 0:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
else:
    logfilename = 'logs/%s_%s.log' % (args.dataset, args.model)
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
    logging.info('log info to ' + logfilename)

logging.info(args)

if args.dataset == 'tiktok':
    ds = ds_tiktok(logging, args)
else:
    raise Exception('no dataset' + args.dataset)

if args.model == 'UltraGCN':
    model = UltraGCN(ds, args, logging)
elif args.model == 'InvRL':
    model = InvRL(ds, args, logging)
else:
    raise Exception('unknown model type', args.model)

model.train()