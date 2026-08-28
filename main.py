#!/usr/bin/env python
import argparse
from utils.model_utils import create_model
import os
import glog as logger
import json
import time

from FLAlgorithms.servers.serverPreciseFCL import FedPrecise
from utils.utils import setup_seed, set_log_file, print_args

def create_server_n_user(args, i):
    
    # create base model, irreverent to FedXXX
    model = create_model(args)
    
    server=FedPrecise(args, model, i)
    return server


def run_job(args, seed):

    logger.info('random seed is: %d'%(seed))
    logger.info("\n\n         [ Start training iteration, seed: {} ]           \n\n".format(seed))
    # Generate model
    server = create_server_n_user(args, seed)
    if args.train:
        server.train(args)

def main(args):
    start_time = time.perf_counter()
    run_job(args, args.seed)
    elapsed_seconds = time.perf_counter() - start_time
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(elapsed_seconds, 60)
    
    logger.info("Finished training.")
    logger.info("Total runtime for %s: %.2f seconds (%.2f minutes, %02d:%02d:%05.2f)",
                args.algorithm, elapsed_seconds, elapsed_seconds / 60.0,
                int(hours), int(minutes), seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST-SVHN-FASHION", choices=['EMNIST-Letters', 'EMNIST-Letters-malicious', 
                                                                            'EMNIST-Letters-shuffle', 'CIFAR100', 'MNIST-SVHN-FASHION', 'JSIEC'])
    parser.add_argument("--datadir", type=str, default="datasets/PreciseFCL/")
    parser.add_argument("--data_split_file", type=str, default="data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl")
    parser.add_argument("--malicious_client_num", type=int, default=0)
    parser.add_argument("--algorithm", type=str, default="PreciseFCL", choices=['FedAvg', 'PreciseFCL', 'FedProx', 'FLwF2T', 'GLFC', 'Re-Fed+', 'FedRNC', 'CGoFed'])
    parser.add_argument("--seed", type=int, default=4396)

    # PreciseFCL
    parser.add_argument("--k_loss_flow", type=float, default=0.1)
    parser.add_argument("--k_kd_global_cls", type=float, default=0)
    parser.add_argument("--k_kd_last_cls", type=float, default=0.2)
    parser.add_argument("--k_kd_feature", type=float, default=0.5)
    parser.add_argument("--k_kd_output", type=float, default=0.1)
    parser.add_argument("--k_flow_lastflow", type=float, default=0.4)
    parser.add_argument("--flow_epoch", type=int, default=5)
    parser.add_argument("--flow_explore_theta", type=float, default=0.2)
    parser.add_argument("--classifier_global_mode", type=str, default='all', help='[head, extractor, none, all]')
    parser.add_argument('--flow_lr', type=float, default=1e-4)  
    parser.add_argument('--fedprox_k', type=float, default=0) 
    parser.add_argument('--use_lastflow_x', action="store_true")
    parser.add_argument("--flwf2t_alpha", type=float, default=0.4, help="weight for classification loss in FLwF2T")
    parser.add_argument("--flwf2t_beta", type=float, default=0.3, help="weight for client-teacher distillation loss in FLwF2T")
    parser.add_argument("--flwf2t_temperature", type=float, default=2.0, help="temperature for FLwF2T distillation")
    parser.add_argument("--glfc_lambda_gc", type=float, default=0.5, help="weight for class-aware gradient compensation loss in GLFC")
    parser.add_argument("--glfc_lambda_rd", type=float, default=0.5, help="weight for class-semantic relation distillation loss in GLFC")
    parser.add_argument("--refedplus_pim_lambda", type=float, default=0.5, help="local-global balance lambda for Re-Fed+ PIM")
    parser.add_argument("--refedplus_memory_buffer_size", type=int, default=500, help="memory buffer size for Re-Fed+ replay")
    parser.add_argument("--refedplus_pim_rounds", type=int, default=3, help="PIM update rounds for Re-Fed+ sample scoring")
    parser.add_argument("--fedrnc_memory_buffer_size", type=int, default=500, help="class-balanced replay memory size for FedRNC")
    parser.add_argument("--fedrnc_lambda_proto", type=float, default=0.5, help="weight for FedRNC prototype contrastive loss")
    parser.add_argument("--fedrnc_lambda_align", type=float, default=0.1, help="weight for FedRNC positive prototype alignment loss")
    parser.add_argument("--fedrnc_temperature", type=float, default=0.2, help="temperature for FedRNC prototype contrastive loss")
    parser.add_argument("--cgofed_mu_init", type=float, default=1.0, help="initial relaxed gradient constraint strength for CGoFed")
    parser.add_argument("--cgofed_alpha", type=float, default=0.98, help="exponential decay rate for CGoFed relaxed constraint")
    parser.add_argument("--cgofed_tau", type=float, default=0.05, help="reserved forgetting threshold for CGoFed constraint reset")
    parser.add_argument("--cgofed_svd_energy", type=float, default=0.95, help="energy threshold for CGoFed SVD representation memory")
    parser.add_argument("--cgofed_sample_size", type=int, default=256, help="max samples used to build CGoFed task representations")
    parser.add_argument("--cgofed_topk", type=int, default=2, help="number of similar historical task models selected for CGoFed")
    parser.add_argument("--cgofed_reg_lambda", type=float, default=1e-4, help="weight for CGoFed cross-task model regularization")


    # optimizer
    parser.add_argument('--lr', type=float, default=1e-04)  
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=0)

    parser.add_argument("--num_glob_iters", type=int, default=60)
    parser.add_argument("--local_epochs", type=int, default=100)

    parser.add_argument("--train", type=int, default=1, choices=[0,1])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"], help="run device (cpu | cuda)")
    
    # model
    parser.add_argument('--c-channel-size', type=int, default=64)
    parser.add_argument("--model", type=str, default="cnn")

    # run routine
    parser.add_argument('--target_dir_name', type = str, default="output_dir", help="the dim of the solution")
    parser.add_argument("--debug", action="store_true", help="debug or not")
    parser.add_argument("--ssh", action="store_true", help="whether is run by search")

    # whether to use LTC & GTR
    parser.add_argument("--using_LTC", type=int, default=0, choices=[0, 1], help="whether to use LTC")
    parser.add_argument("--using_GTR", type=int, default=0, choices=[0, 1], help="whether to use GTR")

    args = parser.parse_args()

    if args.algorithm == 'FLwF2T' and args.flwf2t_alpha + args.flwf2t_beta > 1:
        raise ValueError("For FLwF2T, flwf2t_alpha + flwf2t_beta must be <= 1.")
    if args.algorithm == 'Re-Fed+':
        if not (0 < args.refedplus_pim_lambda < 1):
            raise ValueError("For Re-Fed+, refedplus_pim_lambda must be in (0, 1).")
        if args.refedplus_memory_buffer_size <= 0:
            raise ValueError("For Re-Fed+, refedplus_memory_buffer_size must be positive.")
        if args.refedplus_pim_rounds <= 0:
            raise ValueError("For Re-Fed+, refedplus_pim_rounds must be positive.")

    os.makedirs(args.target_dir_name, exist_ok=True)
    setup_seed(args.seed)
    # args.target_dir = '_'.join([args.target_dir, args.dataset, args.fed_alg])
    if not args.debug:
        log_name = 'run.log'
        args.log_pth = os.path.join(args.target_dir_name, log_name)
        set_log_file(args.log_pth, file_only=args.ssh)
    else:
        logger.info('------------------Debug--------------------')

    print_args(args)
    with open(os.path.join(args.target_dir_name, 'args.json'), "w") as f:
        json.dump(args.__dict__, f, indent =2)
        
    main(args)
