from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import torch
import numpy as np
import random
from modules.Reward_mre import RL

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_threshold",
                        default=0.7,
                        type=float,
                        help="ThredShold in Training Set")
    parser.add_argument("--eval_threshold",
                        default=0.5,
                        type=float,
                        help="ThredShold in Eval Set")
    parser.add_argument("--estimator_batch",
                        default=128,
                        type=int,
                        help="Total batch size for estimator.")
    parser.add_argument("--estimator_learning_rate",
                        default=8e-5,
                        type=float,
                        help="The initial learning rate for Adam of estimator.")
    parser.add_argument("--num_estimator_epochs",
                        default=800.0,
                        type=float,
                        help="Total number of training estimator epochs to perform.")
    parser.add_argument('--alpha',
                        type=float,
                        default=0.7,
                        help="the rate of loss about bert and umt")

    # Relationship extraction model and data parameters
    parser.add_argument('--metric', default='micro_f1', choices=['micro_f1', 'acc'],
                        help='Metric for picking up best checkpoint')
    parser.add_argument('--dataset', default='ours',
                        choices=['none', 'semeval', 'wiki80', 'tacred', 'nyt10', 'ours'],
                        help='Dataset. If not none, the following args can be ignored')
    parser.add_argument('--train_file', default='', type=str,
                        help='Training data file')
    parser.add_argument('--val_file', default='', type=str,
                        help='Validation data file')
    parser.add_argument('--test_file', default='', type=str,
                        help='Test data file')
    parser.add_argument('--rel2id_file', default='', type=str,
                        help='Relation to ID file')
    parser.add_argument('--rel_num', default='1', type=str,
                        help='Number of aligned weights')
    parser.add_argument('--pretrain_path', default='../pretrained_models/bert-base-uncased',
                        help='Pre-trained ckpt path / model name (hugginface)')

    # Other parameters
    parser.add_argument("--output_dir",
                        default='./model_weights',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--clip_dir",
                        default="../pretrained_models/CLIP_model")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to test.")
    parser.add_argument("--max_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--seed',
                        type=int,
                        default=59,
                        help="random seed for initialization")

    args = parser.parse_args()


    # The location for saving parameters
    args.output_estimator_file = os.path.join(args.output_dir, "estimator.bin")  # Parameters of the dataset discriminator
    args.output_mega_file = os.path.join(args.output_dir, "MEGA.pth.tar")  # MEGA.pth.tar
    args.output_mtb_file = os.path.join(args.output_dir, "MTB.pth.tar")  # MEGA_text.pth.tar


    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打印重要的超参数
    logger.info(" train_threshold %s ", str(args.train_threshold))
    logger.info(" eval_threshold %s ", str(args.eval_threshold))
    logger.info(" estimator_batch %s ", str(args.estimator_batch))
    logger.info(" estimator_learning_rate %s ", str(args.estimator_learning_rate))
    logger.info(" num_estimator_epochs %s ", str(args.num_estimator_epochs))
    logger.info(" seed %s ", str(args.seed))
    logger.info(" alpha %s ", str(args.alpha))



    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root_path = '.'
    args.train_file = os.path.join(root_path, 'benchmark', args.dataset,'txt/{}_train_rl.txt'.format(args.dataset))
    args.val_file = os.path.join(root_path, 'benchmark', args.dataset, 'txt/{}_val.txt'.format(args.dataset))
    args.test_file = os.path.join(root_path, 'benchmark', args.dataset, 'txt/{}_test.txt'.format(args.dataset))
    args.pic_train_file = os.path.join(root_path, 'benchmark', args.dataset, 'imgSG/train')
    args.pic_val_file = os.path.join(root_path, 'benchmark', args.dataset, 'imgSG/val')
    args.pic_test_file = os.path.join(root_path, 'benchmark', args.dataset, 'imgSG/test')
    args.rel_train_file = os.path.join(root_path, 'benchmark', args.dataset, 'rel_{}/train').format(args.rel_num)
    args.rel_val_file = os.path.join(root_path, 'benchmark', args.dataset, 'rel_{}/val').format(args.rel_num)
    args.rel_test_file = os.path.join(root_path, 'benchmark', args.dataset, 'rel_{}/test').format(args.rel_num)
    args.rel2id_file = os.path.join(root_path, 'benchmark', args.dataset, '{}_rel2id.json'.format(args.dataset))



    rl = RL(args=args)

    if args.do_train:
        rl.train()

    elif args.do_test:
        mega_file = os.path.join('./model_weights', 'MEGA_all.pth.tar')
        mtb_file = os.path.join('./model_weights', 'MTB_all.pth.tar')
        prob_file = os.path.join('./model_weights', 'prob_mre.txt')
        rl.test(mega_file, mtb_file, prob_file)

