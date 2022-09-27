from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import torch
import numpy as np
import random
from modules.Reward import RL

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
        parser.add_argument("--bert_model", default='../pretrained_models/bert-base-cased', type=str,
                            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                                 "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                                 "bert-base-multilingual-cased, bert-base-chinese.")
        parser.add_argument("--task_name",
                            default='twitter2015',
                            type=str,
                            help="The name of the task to train.")
        parser.add_argument("--output_dir",
                            default='./model_weights',
                            type=str,
                            help="The output directory where the model predictions and checkpoints will be written.")

        ## Other parameters
        parser.add_argument("--clip_dir",
                            default="../pretrained_models/CLIP_model")
        parser.add_argument("--max_seq_length",
                            default=128,
                            type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. \n"
                                 "Sequences longer than this will be truncated, and sequences shorter \n"
                                 "than this will be padded.")
        parser.add_argument("--do_train",
                            action='store_true',
                            help="Whether to run training.")
        parser.add_argument("--do_test",
                            action='store_true',
                            help="Whether to test.")
        parser.add_argument("--do_lower_case",
                            action='store_true',
                            help="Set this flag if you are using an uncased model.")
        parser.add_argument("--estimator_batch",
                            default=256,
                            type=int,
                            help="Total batch size for estimator.")
        parser.add_argument("--train_batch_size",
                            default=8,
                            type=int,
                            help="Total batch size for training.")
        parser.add_argument("--eval_batch_size",
                            default=8,
                            type=int,
                            help="Total batch size for eval.")
        parser.add_argument("--estimator_learning_rate",
                            default=5e-5,
                            type=float,
                            help="The initial learning rate for Adam of estimator.")
        parser.add_argument("--learning_rate",
                            default=5e-5,
                            type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--num_estimator_epochs",
                            default=600.0,
                            type=float,
                            help="Total number of training estimator epochs to perform.")
        parser.add_argument("--warmup_proportion",
                            default=0.1,
                            type=float,
                            help="Proportion of training to perform linear learning rate warmup for. "
                                 "E.g., 0.1 = 10%% of training.")
        parser.add_argument('--seed',
                            type=int,
                            default=59,
                            help="random seed for initialization")
        parser.add_argument('--alpha',
                            type=float,
                            default=0.5,
                            help="the rate of loss about bert and umt")
        parser.add_argument('--gradient_accumulation_steps',
                            type=int,
                            default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")

        parser.add_argument('--fine_tune_cnn', action='store_true', help='fine tune pre-trained CNN if True')
        parser.add_argument('--resnet_root', default='./out_res', help='path the pre-trained cnn models')
        parser.add_argument('--crop_size', type=int, default=224, help='crop size of image')
        parser.add_argument('--path_image', default='./IJCAI2019_data/twitter2015_images/', help='path to images')

        args = parser.parse_args()

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        output_dir = args.output_dir
        # The location for saving parameters
        args.output_estimator_file = os.path.join(args.output_dir, "estimator.bin") # Parameters of the dataset discriminator
        args.output_config_file = os.path.join(args.output_dir, "config.json") # 'config.json' Configuration of the BERT model
        args.output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder.bin") # Parameters of the image encoder


        # The location of the trained model
        # The trained model is trained with 80% of the data
        args.output_umt_file = os.path.join(args.output_dir, "umt_2015.bin")
        args.output_bert_file = os.path.join(args.output_dir, "bert_crf_2015.bin")

        if args.task_name == "twitter2017":
            args.path_image = "./IJCAI2019_data/twitter2017_images/"
            args.data_dir = "./data/twitter2017/"
        elif args.task_name == "twitter2015":
            args.path_image = "./IJCAI2019_data/twitter2015_images/"
            args.data_dir = "./data/twitter2015/"


        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps


        rl = RL(args=args, logger=logger)
        if args.do_train:
            rl.train()
        elif args.do_test:
            if args.task_name == "twitter2017":
                args.output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder_2017.bin")
                # The trained model is trained with 100% of the data
                umt_file = os.path.join(args.output_dir, "umt_2017_all.bin")
                bert_file = os.path.join(args.output_dir, "bert_crf_2017_all.bin")
                # Probability of data discriminator output
                prob_file = os.path.join(args.output_dir, "prob_2017.txt")
                rl.test(umt_file, bert_file, prob_file)
            elif args.task_name == "twitter2015":
                args.output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder_2015.bin")
                umt_file = os.path.join(args.output_dir, "umt_2015_all.bin")
                bert_file = os.path.join(args.output_dir, "bert_crf_2015_all.bin")
                prob_file = os.path.join(args.output_dir, "prob_2015.txt")
                rl.test(umt_file, bert_file, prob_file)

