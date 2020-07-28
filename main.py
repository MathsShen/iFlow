import argparse
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader

from lib.data import SyntheticDataset, DataLoaderGPU, create_if_not_exist_dataset
from lib.metrics import mean_corr_coef as mcc
from lib.models import iVAE
from lib.planar_flow import *
from lib.iFlow import *
from lib.utils import Logger, checkpoint

import os
import os.path as osp
import pdb

import datetime
now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

EXPERIMENT_FOLDER = osp.join('experiments/', now)
LOG_FOLDER = osp.join(EXPERIMENT_FOLDER, 'log/')
TENSORBOARD_RUN_FOLDER = osp.join(EXPERIMENT_FOLDER, 'runs/')
TORCH_CHECKPOINT_FOLDER = osp.join(EXPERIMENT_FOLDER, 'ckpt/')
Z_EST_FOLDER = osp.join(EXPERIMENT_FOLDER, 'z_est/')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default=None, help='path to data file in .npz format. (default None)')
    parser.add_argument('-x', '--data-args', type=str, default=None,
                        help='argument string to generate a dataset. '
                             'This should be of the form nps_ns_dl_dd_nl_s_p_a_u_n. '
                             'Usage explained in lib.data.create_if_not_exist_dataset. '
                             'This will overwrite the `file` argument if given. (default None). '
                             'In case of this argument and `file` argument being None, a default dataset '
                             'described in data.py will be created.')
    parser.add_argument('-z', '--latent-dim', type=int, default=None,
                        help='latent dimension. If None, equals the latent dim of the dataset. (default None)')
    parser.add_argument('-b', '--batch-size', type=int, default=64, help='batch size (default 64)')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='number of epochs (default 20)')
    parser.add_argument('-m', '--max-iter', type=int, default=None, help='max iters, overwrites --epochs')
    parser.add_argument('-g', '--hidden-dim', type=int, default=50, help='hidden dim of the networks (default 50)')
    parser.add_argument('-d', '--depth', type=int, default=3, help='depth (n_layers) of the networks (default 3)')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='learning rate (default 1e-3)')
    parser.add_argument('-s', '--seed', type=int, default=1, help='random seed (default 1)')
    parser.add_argument('-c', '--cuda', action='store_true', default=False, help='train on gpu')
    parser.add_argument('-p', '--preload-gpu', action='store_true', default=False, dest='preload',
                        help='preload data on gpu for faster training.')
    parser.add_argument('-a', '--anneal', action='store_true', default=False, help='use annealing in learning')
    parser.add_argument('-n', '--no-log', action='store_true', default=False, help='run without logging')
    parser.add_argument('-q', '--log-freq', type=int, default=25, help='logging frequency (default 25).')

    parser.add_argument('-i', '--i-what', type=str, default="iFlow")
    parser.add_argument('-ft', '--flow_type', type=str, default="RQSplineFlow")
    parser.add_argument('-nb', '--num_bins', type=int, default=8)
    parser.add_argument('-npa', '--nat_param_act', type=str, default="Sigmoid")
    parser.add_argument('-u', '--gpu_id', type=str, default='0')
    parser.add_argument('-fl', '--flow_length', type=int, default=10)
    parser.add_argument('-lr_df', '--lr_drop_factor', type=float, default=0.5)
    parser.add_argument('-lr_pn', '--lr_patience', type=int, default=10)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    st = time.time()

    if args.file is None:
        args.file = create_if_not_exist_dataset(root='data/{}/'.format(args.seed), arg_str=args.data_args)
    
    metadata = vars(args).copy()
    del metadata['no_log'], metadata['data_args']

    device = torch.device('cuda' if args.cuda else 'cpu')
    print('training on {}'.format(torch.cuda.get_device_name(device) if args.cuda else 'cpu'))

    # load data
    if not args.preload:
        dset = SyntheticDataset(args.file, 'cpu') # originally 'cpu' ????
        loader_params = {'num_workers': 1, 'pin_memory': True} if args.cuda else {} ###############
        train_loader = DataLoader(dset, shuffle=True, batch_size=args.batch_size, **loader_params)
        data_dim, latent_dim, aux_dim = dset.get_dims()
        args.N = len(dset)
        metadata.update(dset.get_metadata())
    else:
        train_loader = DataLoaderGPU(args.file, shuffle=True, batch_size=args.batch_size)
        data_dim, latent_dim, aux_dim = train_loader.get_dims()
        args.N = train_loader.dataset_len
        metadata.update(train_loader.get_metadata())
    
    if args.max_iter is None:
        args.max_iter = len(train_loader) * args.epochs

    if args.latent_dim is not None:
        latent_dim = args.latent_dim
        metadata.update({"train_latent_dim": latent_dim})

    # define model and optimizer
    model = None
    if args.i_what == 'iVAE':
        model = iVAE(latent_dim, \
                 data_dim, \
                 aux_dim, \
                 n_layers=args.depth, \
                 activation='lrelu', \
                 device=device, \
                 hidden_dim=args.hidden_dim, \
                 anneal=args.anneal) # False
    elif args.i_what == 'iFlow':
        metadata.update({"device": device})
        model = iFlow(args=metadata).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                                                     factor=args.lr_drop_factor, \
                                                     patience=args.lr_patience, \
                                                     verbose=True) # factor=0.1 and patience=4

    ste = time.time()
    print('setup time: {}s'.format(ste - st))

    # setup loggers
    logger = Logger(logdir=LOG_FOLDER) # 'log/'
    exp_id = logger.get_id() # 1

    tensorboard_run_name = TENSORBOARD_RUN_FOLDER + 'exp' + str(exp_id) + '_'.join(
        map(str, ['', args.batch_size, args.max_iter, args.lr, args.hidden_dim, args.depth, args.anneal]))
    # 'runs/exp1_64_12500_0.001_50_3_False'
    
    writer = SummaryWriter(logdir=tensorboard_run_name)

    if args.i_what == 'iFlow':
        logger.add('log_normalizer')
        logger.add('neg_log_det')
        logger.add('neg_trace')

    logger.add('loss')
    logger.add('perf')
    print('Beginning training for exp: {}'.format(exp_id))

    # training loop
    epoch = 0
    model.train()
    while epoch < args.epochs: #args.max_iter:  #12500
        est = time.time()
        for itr, (x, u, z) in enumerate(train_loader):
            acc_itr = itr + epoch * len(train_loader)

            # x is of shape [64, 4]
            # u is of shape [64, 40], one-hot coding of 40 classes
            # z is of shape [64, 2]

            #it += 1
            #model.anneal(args.N, args.max_iter, it)
            optimizer.zero_grad()

            if args.cuda and not args.preload:
                x = x.cuda(device=device, non_blocking=True)
                u = u.cuda(device=device, non_blocking=True)

            if args.i_what == 'iVAE':
                elbo, z_est = model.elbo(x, u) #elbo is a scalar loss while z_est is of shape [64, 2]
                loss = elbo.mul(-1)

            elif args.i_what == 'iFlow':
                (log_normalizer, neg_trace, neg_log_det), z_est = model.neg_log_likelihood(x, u)
                loss = log_normalizer + neg_trace + neg_log_det
            
            loss.backward()
            optimizer.step()

            logger.update('loss', loss.item())
            if args.i_what == 'iFlow':
                logger.update('log_normalizer', log_normalizer.item())
                logger.update('neg_trace', neg_trace.item())
                logger.update('neg_log_det', neg_log_det.item())

            perf = mcc(z.cpu().numpy(), z_est.cpu().detach().numpy())
            logger.update('perf', perf)

            if acc_itr % args.log_freq == 0: # % 25
                logger.log()
                writer.add_scalar('data/performance', logger.get_last('perf'), acc_itr)
                writer.add_scalar('data/loss', logger.get_last('loss'), acc_itr)

                if args.i_what == 'iFlow':
                    writer.add_scalar('data/log_normalizer', logger.get_last('log_normalizer'), acc_itr)
                    writer.add_scalar('data/neg_trace', logger.get_last('neg_trace'), acc_itr)
                    writer.add_scalar('data/neg_log_det', logger.get_last('neg_log_det'), acc_itr)

                scheduler.step(logger.get_last('loss'))
                #scheduler.step(-perf)

            if acc_itr % int(args.max_iter / 5) == 0 and not args.no_log:
                checkpoint(TORCH_CHECKPOINT_FOLDER, \
                           exp_id, \
                           acc_itr, \
                           model, \
                           optimizer, \
                           logger.get_last('loss'), \
                           logger.get_last('perf'))
            
            """
            if args.i_what == 'iVAE':
                print('----epoch {} iter {}:\tloss: {:.4f};\tperf: {:.4f}'.format(\
                                                                   epoch, \
                                                                   itr, \
                                                                   loss.item(), \
                                                                   perf))
            elif args.i_what == 'iFlow':
                print('----epoch {} iter {}:\tloss: {:.4f} (l1: {:.4f}, l2: {:.4f}, l3: {:.4f});\tperf: {:.4f}'.format(\
                                                                    epoch, \
                                                                    itr, \
                                                                    loss.item(), \
                                                                    log_normalizer.item(), \
                                                                    neg_trace.item(), \
                                                                    neg_log_det.item(), \
                                                                    perf))
            """
        
        epoch += 1
        eet = time.time()
        if args.i_what == 'iVAE':
            print('epoch {}: {:.4f}s;\tloss: {:.4f};\tperf: {:.4f}'.format(epoch, \
                                                                   eet-est, \
                                                                   logger.get_last('loss'), \
                                                                   logger.get_last('perf')))
        elif args.i_what == 'iFlow':
            print('epoch {}: {:.4f}s;\tloss: {:.4f} (l1: {:.4f}, l2: {:.4f}, l3: {:.4f});\tperf: {:.4f}'.format(\
                                                                    epoch, \
                                                                    eet-est, \
                                                                    logger.get_last('loss'), \
                                                                    logger.get_last('log_normalizer'), \
                                                                    logger.get_last('neg_trace'), \
                                                                    logger.get_last('neg_log_det'), \
                                                                    logger.get_last('perf')))

    et = time.time()
    print('training time: {}s'.format(et - ste))

    writer.close()
    if not args.no_log:
        logger.add_metadata(**metadata)
        logger.save_to_json()
        logger.save_to_npz()

    print('total time: {}s'.format(et - st))

    ###### Run Test Here
    model.eval()
    if args.i_what == 'iFlow':
        import operator
        from functools import reduce
        total_num_examples = reduce(operator.mul, map(int, args.data_args.split('_')[:2]))
        model.set_mask(total_num_examples)
        
    assert args.file is not None
    A = np.load(args.file)

    x = A['x'] # of shape
    x = torch.from_numpy(x).to(device)
    print("x.shape ==", x.shape)
    
    s = A['s'] # of shape
    #s = torch.from_numpy(s).to(device)
    print("s.shape ==", s.shape)

    u = A['u'] # of shape
    u = torch.from_numpy(u).to(device)
    print("u.shape ==", u.shape)

    if args.i_what == 'iVAE':
        _, z_est = model.elbo(x, u)
    elif args.i_what == 'iFlow':
        #(_, _, _), z_est = model.neg_log_likelihood(x, u)
        z_est, nat_params = model.inference(x, u)

    z_est = z_est.cpu().detach().numpy()
    #nat_params = nat_params.cpu().detach().numpy()
    #os.makedirs(Z_EST_FOLDER)
    #np.save("{}/z_est.npy".format(Z_EST_FOLDER), z_est)
    #np.save("{}/nat_params.npy".format(Z_EST_FOLDER), nat_params)
    #print("z_est.shape ==", z_est.shape)
    
    perf = mcc(s, z_est)
    print("EVAL PERFORMANCE: {}".format(perf))
    print("DONE.")

