from __future__ import absolute_import, division, print_function
import os
import argparse
import logging
import os
import random
import json
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
from torchvision import transforms
from PIL import Image
from transformers.models.clip import CLIPProcessor, CLIPModel
from modules.Partitioner import *
from opennre.encoder.bert_encoder import MEGABERTEntityEncoder, BERTEntityEncoder
from opennre.framework.sentence_re import SentenceRE
from opennre.model.softmax_nn import SoftmaxNN
from opennre.framework.data_loader import SentenceREDataset, SentenceRELoader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class RL(object):

    def __init__(self, args):
        self.args = args


    def eval_union(self, mega_true_list, mega_pred_list, mtb_true_list, mtb_pred_list):
        true_result = mega_true_list + mtb_true_list
        pred_result = mega_pred_list + mtb_pred_list


        correct = 0
        total = len(true_result)

        correct_positive = 0

        pred_positive = 0

        gold_positive = 0

        neg = -1

        for i in range(total):

            if true_result[i] == pred_result[i]:
                correct += 1
                if true_result[i] != 0:
                    correct_positive += 1
            if true_result[i] != 0:
                gold_positive += 1
            if pred_result[i] != 0:
                pred_positive += 1
        acc = float(correct) / float(total)
        try:
            micro_p = float(correct_positive) / float(pred_positive)
        except:
            micro_p = 0
        try:
            micro_r = float(correct_positive) / float(gold_positive)
        except:
            micro_r = 0
        try:
            micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
        except:
            micro_f1 = 0

        result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
        logging.info('Evaluation result: {}.'.format(result))

        return result



    # Start the entire reinforcement learning training process
    def train(self):

        # Print important hyperparameters
        logger.info(" estimator_batch %s ", str(self.args.estimator_batch))
        logger.info(" estimator_learning_rate %s ", str(self.args.estimator_learning_rate))
        logger.info(" seed %s ", str(self.args.seed))
        logger.info(" alpha %s ", str(self.args.alpha))

        output_estimator_file = self.args.output_estimator_file
        output_mega_file = self.args.output_mega_file
        output_mtb_file = self.args.output_mtb_file

        device = self.args.device
        alpha = self.args.alpha

        rel2id = json.load(open(self.args.rel2id_file))


        Mega_sentence_encoder = MEGABERTEntityEncoder(
            max_length=self.args.max_length,
            pretrain_path=self.args.pretrain_path,
            mask_entity=False
        )
        Mega_model = SoftmaxNN(Mega_sentence_encoder, len(rel2id), rel2id)
        Mega_framework = SentenceRE(
            train_path=self.args.train_file,
            train_rel_path=self.args.rel_train_file,
            train_pic_path=self.args.pic_train_file,
            val_path=self.args.val_file,
            val_rel_path=self.args.rel_val_file,
            val_pic_path=self.args.pic_val_file,
            test_path=self.args.test_file,
            test_rel_path=self.args.rel_test_file,
            test_pic_path=self.args.pic_test_file,
            model=Mega_model,
            ckpt=None,
            batch_size=16,
            max_epoch=10,
            lr=0.1,
            opt='sgd'
        )


        MTB_sentence_encoder = BERTEntityEncoder(
            max_length=self.args.max_length,
            pretrain_path=self.args.pretrain_path,
            mask_entity=False
        )
        MTB_model = SoftmaxNN(MTB_sentence_encoder, len(rel2id), rel2id)
        MTB_framework = SentenceRE(
            train_path=None,
            train_rel_path=None,
            train_pic_path=None,
            val_path=None,
            val_rel_path=None,
            val_pic_path=None,
            test_path=None,
            test_rel_path=None,
            test_pic_path=None,
            model=MTB_model,
            ckpt=None,
            batch_size=16,
            max_epoch=10,
            lr=0.1,
            opt='sgd'
        )

        Mega_framework.load_state_dict(torch.load(output_mega_file)['state_dict'])
        MTB_framework.load_state_dict(torch.load(output_mtb_file)['state_dict'])


        ## Training set features
        train_dataloder = Mega_framework.train_loader
        all_label_train = []
        all_indexed_tokens_train = []
        all_att_mask_train = []
        all_pos1_train = []
        all_pos2_train = []
        all_pic_feat_train = []
        all_rel_feat_train = []
        all_clip_input_ids_train = []
        all_clip_attention_mask_train = []
        all_clip_pixel_values_train = []
        for i, batch_data in enumerate(train_dataloder):
            label = batch_data[0]   # B
            indexed_tokens = batch_data[1] # B * L
            att_mask = batch_data[2] # B * L
            pos1 = batch_data[3] # B * 1
            pos2 = batch_data[4] # B * 1
            pic_feat = batch_data[5] # (B * P_L) * L
            rel_feat = batch_data[6] # (B * rel_L) * L
            clip_input_ids = batch_data[7]  # B * L
            clip_attention_mask = batch_data[8] # B * L
            clip_pixel_values = batch_data[9]   # B * 3 * 224 * 224

            all_label_train.append(label)
            all_indexed_tokens_train.append(indexed_tokens)
            all_att_mask_train.append(att_mask)
            all_pos1_train.append(pos1)
            all_pos2_train.append(pos2)
            all_pic_feat_train.append(pic_feat)
            all_rel_feat_train.append(rel_feat)
            all_clip_input_ids_train.append(clip_input_ids)
            all_clip_attention_mask_train.append(clip_attention_mask)
            all_clip_pixel_values_train.append(clip_pixel_values)


        all_label_train = torch.cat(all_label_train,dim=0)  # B*x
        all_indexed_tokens_train = torch.cat(all_indexed_tokens_train,dim=0) # B * L
        all_att_mask_train = torch.cat(all_att_mask_train,dim=0)    # B * L
        all_pos1_train = torch.cat(all_pos1_train,dim=0)    # B * 1
        all_pos2_train = torch.cat(all_pos2_train,dim=0)    # B * 1

        all_pic_feat_train = torch.cat(all_pic_feat_train,dim=0)    # B*x * L
        all_pic_feat_train = all_pic_feat_train.reshape(-1, 10, 4096)

        all_rel_feat_train = torch.cat(all_rel_feat_train,dim=0)    # B*x * L
        all_rel_feat_train = all_rel_feat_train.reshape(-1, 10, 128)

        all_clip_input_ids_train = torch.cat(all_clip_input_ids_train,dim=0)    # B*x * L
        all_clip_attention_mask_train = torch.cat(all_clip_attention_mask_train,dim=0)  # B*x * L
        all_clip_pixel_values_train = torch.cat(all_clip_pixel_values_train,dim=0)  # B * 3 * 224 * 224

        # Validation data set
        dev_dataloder = Mega_framework.val_loader
        all_label_dev = []
        all_indexed_tokens_dev = []
        all_att_mask_dev = []
        all_pos1_dev = []
        all_pos2_dev = []
        all_pic_feat_dev = []
        all_rel_feat_dev = []
        all_clip_input_ids_dev = []
        all_clip_attention_mask_dev = []
        all_clip_pixel_values_dev = []
        for i, batch_data in enumerate(dev_dataloder):
            label = batch_data[0]  # B
            indexed_tokens = batch_data[1]  # B * L
            att_mask = batch_data[2]  # B * L
            pos1 = batch_data[3]  # B * 1
            pos2 = batch_data[4]  # B * 1
            pic_feat = batch_data[5]  # (B * P_L) * L
            rel_feat = batch_data[6]  # (B * rel_L) * L
            clip_input_ids = batch_data[7]  # B * L
            clip_attention_mask = batch_data[8]  # B * L
            clip_pixel_values = batch_data[9]  # B * 3 * 224 * 224

            all_label_dev.append(label)
            all_indexed_tokens_dev.append(indexed_tokens)
            all_att_mask_dev.append(att_mask)
            all_pos1_dev.append(pos1)
            all_pos2_dev.append(pos2)
            all_pic_feat_dev.append(pic_feat)
            all_rel_feat_dev.append(rel_feat)
            all_clip_input_ids_dev.append(clip_input_ids)
            all_clip_attention_mask_dev.append(clip_attention_mask)
            all_clip_pixel_values_dev.append(clip_pixel_values)


        all_label_dev = torch.cat(all_label_dev, dim=0)  # B*x
        all_indexed_tokens_dev = torch.cat(all_indexed_tokens_dev, dim=0)  # B * L
        all_att_mask_dev = torch.cat(all_att_mask_dev, dim=0)  # B * L
        all_pos1_dev = torch.cat(all_pos1_dev, dim=0)  # B * 1
        all_pos2_dev = torch.cat(all_pos2_dev, dim=0)  # B * 1

        all_pic_feat_dev = torch.cat(all_pic_feat_dev, dim=0)  # B*x * L
        all_pic_feat_dev = all_pic_feat_dev.reshape(-1, 10, 4096)

        all_rel_feat_dev = torch.cat(all_rel_feat_dev, dim=0)  # B*x * L
        all_rel_feat_dev = all_rel_feat_dev.reshape(-1, 10, 128)

        all_clip_input_ids_dev = torch.cat(all_clip_input_ids_dev, dim=0)  # B*x * L
        all_clip_attention_mask_dev = torch.cat(all_clip_attention_mask_dev, dim=0)  # B*x * L
        all_clip_pixel_values_dev = torch.cat(all_clip_pixel_values_dev, dim=0)  # B * 3 * 224 * 224

        eval_data = TensorDataset(all_label_dev, all_indexed_tokens_dev, all_att_mask_dev, all_pos1_dev,
                                  all_pos2_dev, all_pic_feat_dev, all_rel_feat_dev, all_clip_input_ids_dev,
                                  all_clip_attention_mask_dev, all_clip_pixel_values_dev)

        eval_dataloader = DataLoader(eval_data, shuffle=False,
                                     batch_size=self.args.eval_batch_size)


        # Test set features
        test_dataloder = Mega_framework.test_loader
        all_label_test = []
        all_indexed_tokens_test = []
        all_att_mask_test = []
        all_pos1_test = []
        all_pos2_test = []
        all_pic_feat_test = []
        all_rel_feat_test = []
        all_clip_input_ids_test = []
        all_clip_attention_mask_test = []
        all_clip_pixel_values_test = []
        for i, batch_data in enumerate(test_dataloder):
            label = batch_data[0]  # B
            indexed_tokens = batch_data[1]  # B * L
            att_mask = batch_data[2]  # B * L
            pos1 = batch_data[3]  # B * 1
            pos2 = batch_data[4]  # B * 1
            pic_feat = batch_data[5]  # (B * P_L) * L
            rel_feat = batch_data[6]  # (B * rel_L) * L
            clip_input_ids = batch_data[7]  # B * L
            clip_attention_mask = batch_data[8]  # B * L
            clip_pixel_values = batch_data[9]  # B * 3 * 224 * 224

            all_label_test.append(label)
            all_indexed_tokens_test.append(indexed_tokens)
            all_att_mask_test.append(att_mask)
            all_pos1_test.append(pos1)
            all_pos2_test.append(pos2)
            all_pic_feat_test.append(pic_feat)
            all_rel_feat_test.append(rel_feat)
            all_clip_input_ids_test.append(clip_input_ids)
            all_clip_attention_mask_test.append(clip_attention_mask)
            all_clip_pixel_values_test.append(clip_pixel_values)

        all_label_test = torch.cat(all_label_test, dim=0)  # B*x
        all_indexed_tokens_test = torch.cat(all_indexed_tokens_test, dim=0)  # B * L
        all_att_mask_test = torch.cat(all_att_mask_test, dim=0)  # B * L
        all_pos1_test = torch.cat(all_pos1_test, dim=0)  # B * 1
        all_pos2_test = torch.cat(all_pos2_test, dim=0)  # B * 1

        all_pic_feat_test = torch.cat(all_pic_feat_test, dim=0)  # B*x * L
        all_pic_feat_test = all_pic_feat_test.reshape(-1, 10, 4096)

        all_rel_feat_test = torch.cat(all_rel_feat_test, dim=0)  # B*x * L
        all_rel_feat_test = all_rel_feat_test.reshape(-1, 10, 128)

        all_clip_input_ids_test = torch.cat(all_clip_input_ids_test, dim=0)  # B*x * L
        all_clip_attention_mask_test = torch.cat(all_clip_attention_mask_test, dim=0)  # B*x * L
        all_clip_pixel_values_test = torch.cat(all_clip_pixel_values_test, dim=0)  # B * 3 * 224 * 224

        test_data = TensorDataset(all_label_test, all_indexed_tokens_test, all_att_mask_test, all_pos1_test,
                                 all_pos2_test,all_pic_feat_test, all_rel_feat_test, all_clip_input_ids_test,
                                 all_clip_attention_mask_test, all_clip_pixel_values_test)

        test_dataloader = DataLoader(test_data, shuffle=False,
                                    batch_size=self.args.eval_batch_size)


        clip_model = CLIPModel.from_pretrained(self.args.clip_dir)


        estimator = Estimator(clip_model)
        estimator.to(device)


        optimizer_grouped_parameters = [
            {'params': estimator.classfier.parameters()}
        ]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args.estimator_learning_rate)

        max_f1 = 0.0

        for i in trange(int(self.args.num_estimator_epochs)):
            torch.cuda.empty_cache()
            # Shuffle the data to get `self.args.estimator_batch` random numbers for randomly selecting the data.
            batch_idx = random.sample([i for i in range(all_label_train.shape[0])], self.args.estimator_batch)  # batch

            batch_label_train = all_label_train[batch_idx]
            batch_indexed_tokens_train = all_indexed_tokens_train[batch_idx]
            batch_att_mask_train = all_att_mask_train[batch_idx]
            batch_pos1_train = all_pos1_train[batch_idx]
            batch_pos2_train = all_pos2_train[batch_idx]
            batch_pic_feat_train = all_pic_feat_train[batch_idx]
            batch_rel_feat_train = all_rel_feat_train[batch_idx]
            batch_clip_input_ids_train = all_clip_input_ids_train[batch_idx]
            batch_clip_attention_mask_train = all_clip_attention_mask_train[batch_idx]
            batch_clip_pixel_values_train = all_clip_pixel_values_train[batch_idx]

            logger.info(" Sample Index Length = %d ", len(batch_idx))
            logger.info(" Data Length = %d ", batch_label_train.shape[0])


            batch_clip_input_ids_train = batch_clip_input_ids_train.to(device)
            batch_clip_attention_mask_train = batch_clip_attention_mask_train.to(device)
            batch_clip_pixel_values_train = batch_clip_pixel_values_train.to(device)


            # Returns the probability that an image needs to be added to each data: (batch, 1)
            estimator.train()
            probs = estimator(input_ids=batch_clip_input_ids_train, input_mask=batch_clip_attention_mask_train,
                              pixel_values=batch_clip_pixel_values_train)  # (batch,1)

            batch_clip_input_ids_train = batch_clip_input_ids_train.cpu()
            batch_clip_attention_mask_train = batch_clip_attention_mask_train.cpu()
            batch_clip_pixel_values_train = batch_clip_pixel_values_train.cpu()


            logger.info("  Probs Shape =  %s", probs.shape)

            # Define the sampler to select the data to be trained according to the probability of the output, initialize the sampler
            sampler = Sampler(probs)

            thredshold = self.args.train_threshold
            select_idx, unselect_idx, action_holder = sampler.get_index(thredshold)
            logger.info(" Length of Select_idx = %d  ", len(select_idx))
            logger.info(" Length of Unselect_idx = %d  ", len(unselect_idx))
            logger.info(" Length of Action_holder = %d ", len(action_holder))

            select_idx = torch.tensor(select_idx, dtype=torch.long)
            unselect_idx = torch.tensor(unselect_idx, dtype=torch.long)

            action_holder = torch.from_numpy(np.array(action_holder)).float()
            action_holder = action_holder.unsqueeze(1).to(device)


            mega_batch_label_train = torch.index_select(batch_label_train, dim=0, index=select_idx)
            mega_batch_indexed_tokens_train = torch.index_select(batch_indexed_tokens_train, dim=0, index=select_idx)
            mega_batch_att_mask_train = torch.index_select(batch_att_mask_train, dim=0, index=select_idx)
            mega_batch_pos1_train = torch.index_select(batch_pos1_train, dim=0, index=select_idx)
            mega_batch_pos2_train = torch.index_select(batch_pos2_train, dim=0, index=select_idx)
            mega_batch_pic_feat_train = torch.index_select(batch_pic_feat_train, dim=0, index=select_idx)
            mega_batch_rel_feat_train = torch.index_select(batch_rel_feat_train, dim=0, index=select_idx)

            mtb_batch_label_train = torch.index_select(batch_label_train, dim=0, index=unselect_idx)
            mtb_batch_indexed_tokens_train = torch.index_select(batch_indexed_tokens_train, dim=0, index=unselect_idx)
            mtb_batch_att_mask_train = torch.index_select(batch_att_mask_train, dim=0, index=unselect_idx)
            mtb_batch_pos1_train = torch.index_select(batch_pos1_train, dim=0, index=unselect_idx)
            mtb_batch_pos2_train = torch.index_select(batch_pos2_train, dim=0, index=unselect_idx)
            mtb_batch_pic_feat_train = torch.index_select(batch_pic_feat_train, dim=0, index=unselect_idx)
            mtb_batch_rel_feat_train = torch.index_select(batch_rel_feat_train, dim=0, index=unselect_idx)


            mega_train_data = TensorDataset(mega_batch_label_train, mega_batch_indexed_tokens_train, mega_batch_att_mask_train, mega_batch_pos1_train,
                                            mega_batch_pos2_train, mega_batch_pic_feat_train, mega_batch_rel_feat_train)
            mega_train_dataloader = DataLoader(mega_train_data, shuffle=False, batch_size=self.args.eval_batch_size)


            mtb_train_data = TensorDataset(mtb_batch_label_train, mtb_batch_indexed_tokens_train, mtb_batch_att_mask_train, mtb_batch_pos1_train,
                                           mtb_batch_pos2_train, mtb_batch_pic_feat_train, mtb_batch_rel_feat_train)
            mtb_train_dataloader = DataLoader(mtb_train_data, shuffle=False, batch_size=self.args.eval_batch_size)


            logger.info(" ******** Eval Mega In Multi_data ********")
            _, __, mega_in_multi_result = Mega_framework.eval_model(mega_train_dataloader)
            logger.info("Training Mega Model In Multimodal Dataset Accuracy = %s, Micro_p = %s, Micro_r = %s, Micro_f1 = %s  ",
                        str(mega_in_multi_result['acc']), str(mega_in_multi_result['micro_p']), str(mega_in_multi_result['micro_r']),
                        str(mega_in_multi_result['micro_f1']))


            logger.info(" ******** Eval Mega In Text_data ********")
            _, __, mega_in_text_result = Mega_framework.eval_model(mtb_train_dataloader)
            logger.info("Training Mega Model In Text Dataset Accuracy = %s, Micro_p = %s, Micro_r = %s, Micro_f1 = %s  ",
                        str(mega_in_text_result['acc']), str(mega_in_text_result['micro_p']),
                        str(mega_in_text_result['micro_r']),
                        str(mega_in_text_result['micro_f1']))


            logger.info(" ******** Eval MTB In Multi_data ********")
            _, __, mtb_in_multi_result = MTB_framework.eval_model(mega_train_dataloader)
            logger.info("Training MTB Model In Multimodal Dataset Accuracy = %s, Micro_p = %s, Micro_r = %s, Micro_f1 = %s  ",
                        str(mtb_in_multi_result['acc']), str(mtb_in_multi_result['micro_p']),
                        str(mtb_in_multi_result['micro_r']),
                        str(mtb_in_multi_result['micro_f1']))


            logger.info(" ******** Eval MTB In Text_data ********")
            _, __, mtb_in_text_result = MTB_framework.eval_model(mtb_train_dataloader)
            logger.info("Training MTB Model In Text Dataset Accuracy = %s, Micro_p = %s, Micro_r = %s, Micro_f1 = %s  ",
                        str(mtb_in_text_result['acc']), str(mtb_in_text_result['micro_p']),
                        str(mtb_in_text_result['micro_r']),
                        str(mtb_in_text_result['micro_f1']))


            delta1 = mega_in_multi_result['micro_f1'] - mtb_in_multi_result['micro_f1']
            logger.info("  delta1 = %s  ", str(delta1))
            delta2 = mtb_in_text_result['micro_f1'] - mega_in_text_result['micro_f1']
            logger.info("  delta2 = %s  ", str(delta2))
            reward = alpha * delta1 + (1 - alpha) * delta2
            logger.info("  reward = %s  ", str(reward))

            reward_holder = torch.from_numpy(np.array([reward])).float()
            logger.info(" reward_holder = %s", str(reward_holder))
            reward_holder = reward_holder.to(device)
            pi = action_holder * probs + (1.0 - action_holder) * (1.0 - probs)
            loss = -1.0 * torch.sum(torch.log(pi) * reward_holder)

            logger.info("   **** loss = %s ****  ", str(loss.tolist()))
            loss.backward()
            optimizer.step()

            estimator.eval()
            logger.info("***** Running Estimator Eval evaluation *****")
            logger.info("  Num examples = %d", len(eval_data))
            logger.info("  Batch size = %d", self.args.eval_batch_size)

            eval_result = []

            for batch_label, batch_indexed_tokens, batch_att_mask, batch_pos1, batch_pos2,batch_pic_feat, \
                batch_rel_feat, batch_clip_input_ids, batch_clip_attention_mask, batch_clip_pixel_values \
                    in tqdm(eval_dataloader, desc="Evaluating"):

                batch_clip_input_ids = batch_clip_input_ids.to(device)
                batch_clip_attention_mask = batch_clip_attention_mask.to(device)
                batch_clip_pixel_values = batch_clip_pixel_values.to(device)
                with torch.no_grad():
                    predicted_probs = estimator(input_ids=batch_clip_input_ids, input_mask=batch_clip_attention_mask,
                                                pixel_values=batch_clip_pixel_values)

                    eval_result.extend(predicted_probs)

            thredshold = self.args.eval_threshold

            eval_result = torch.tensor(eval_result, dtype=torch.float32)
            sampler.probs = eval_result
            eval_result_select_index, eval_result_unselect_index = sampler.get_result_index(thredshold)

            logger.info(" Length of Eval_result_select_index = %d  ", len(eval_result_select_index))
            logger.info(" Length of Eval_result_unselect_index = %d  ", len(eval_result_unselect_index))

            eval_selected_data = TensorDataset(all_label_dev[eval_result_select_index],
                                               all_indexed_tokens_dev[eval_result_select_index],
                                               all_att_mask_dev[eval_result_select_index],
                                               all_pos1_dev[eval_result_select_index],
                                               all_pos2_dev[eval_result_select_index],
                                               all_pic_feat_dev[eval_result_select_index],
                                               all_rel_feat_dev[eval_result_select_index])

            eval_unselected_data = TensorDataset(all_label_dev[eval_result_unselect_index],
                                               all_indexed_tokens_dev[eval_result_unselect_index],
                                               all_att_mask_dev[eval_result_unselect_index],
                                               all_pos1_dev[eval_result_unselect_index],
                                               all_pos2_dev[eval_result_unselect_index],
                                               all_pic_feat_dev[eval_result_unselect_index],
                                               all_rel_feat_dev[eval_result_unselect_index])

            eval_selected_dataloder = DataLoader(eval_selected_data, shuffle=False, batch_size=self.args.eval_batch_size)
            eval_unselected_dataloder = DataLoader(eval_unselected_data, shuffle=False, batch_size=self.args.eval_batch_size)



            logger.info(" ******** Eval Mega In Multi_data ********")
            mega_true_list, mega_pred_list, mega_in_multi_result = Mega_framework.eval_model(eval_selected_dataloder)
            logger.info("Evaluation Mega Model In Multimodal Dataset Accuracy = %s, Micro_p = %s, Micro_r = %s, Micro_f1 = %s  ",
                        str(mega_in_multi_result['acc']), str(mega_in_multi_result['micro_p']),
                        str(mega_in_multi_result['micro_r']),
                        str(mega_in_multi_result['micro_f1']))


            logger.info(" ******** Eval Mega In Text_data ********")
            _, __, mega_in_text_result = Mega_framework.eval_model(eval_unselected_dataloder)
            logger.info("Evaluation Mega Model In Text Dataset Accuracy = %s, Micro_p = %s, Micro_r = %s, Micro_f1 = %s  ",
                        str(mega_in_text_result['acc']), str(mega_in_text_result['micro_p']),
                        str(mega_in_text_result['micro_r']),
                        str(mega_in_text_result['micro_f1']))


            logger.info(" ******** Test - Eval MTB In Multi_data ********")
            _, __, mtb_in_multi_result = MTB_framework.eval_model(eval_selected_dataloder)
            logger.info("Evaluation MTB Model In Multimodal Dataset Accuracy = %s, Micro_p = %s, Micro_r = %s, Micro_f1 = %s  ",
                        str(mtb_in_multi_result['acc']), str(mtb_in_multi_result['micro_p']),
                        str(mtb_in_multi_result['micro_r']),
                        str(mtb_in_multi_result['micro_f1']))

            logger.info(" ******** Test - Eval MTB In Text_data ********")
            mtb_true_list, mtb_pred_list, mtb_in_text_result = MTB_framework.eval_model(eval_unselected_dataloder)
            logger.info("Evaluation MTB Model In Text Dataset Accuracy = %s, Micro_p = %s, Micro_r = %s, Micro_f1 = %s  ",
                        str(mtb_in_text_result['acc']), str(mtb_in_text_result['micro_p']),
                        str(mtb_in_text_result['micro_r']),
                        str(mtb_in_text_result['micro_f1']))

            logger.info(" ********** Union Prediction *********")

            result = self.eval_union(mega_true_list, mega_pred_list, mtb_true_list, mtb_pred_list)

            logger.info(" union_acc:{}  union_micro_p:{}  union_micro_r:{}  union_micro_f1:{} ".format(
                str(result['acc']), str(result['micro_p']), str(result['micro_r']), str(result['micro_f1']))
            )

            if result['micro_f1'] > max_f1:
                max_f1 = result['micro_f1']
                torch.save(estimator.state_dict(), output_estimator_file)


    def test(self, mega_file, mtb_file, prob_file):
        output_mega_file = mega_file
        output_mtb_file = mtb_file
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        root_path = '.'
        self.args.train_file = os.path.join(root_path, 'benchmark', self.args.dataset,
                                       'txt/{}_train_rl.txt'.format(self.args.dataset))
        self.args.val_file = os.path.join(root_path, 'benchmark', self.args.dataset, 'txt/{}_val.txt'.format(self.args.dataset))
        self.args.test_file = os.path.join(root_path, 'benchmark', self.args.dataset, 'txt/{}_test.txt'.format(self.args.dataset))
        self.args.pic_train_file = os.path.join(root_path, 'benchmark', self.args.dataset, 'imgSG/train')
        self.args.pic_val_file = os.path.join(root_path, 'benchmark', self.args.dataset, 'imgSG/val')
        self.args.pic_test_file = os.path.join(root_path, 'benchmark', self.args.dataset, 'imgSG/test')
        self.args.rel_train_file = os.path.join(root_path, 'benchmark', self.args.dataset, 'rel_{}/train').format(
            self.args.rel_num)
        self.args.rel_val_file = os.path.join(root_path, 'benchmark', self.args.dataset, 'rel_{}/val').format(self.args.rel_num)
        self.args.rel_test_file = os.path.join(root_path, 'benchmark', self.args.dataset, 'rel_{}/test').format(self.args.rel_num)
        self.args.rel2id_file = os.path.join(root_path, 'benchmark', self.args.dataset, '{}_rel2id.json'.format(self.args.dataset))

        rel2id = json.load(open(self.args.rel2id_file))

        Mega_sentence_encoder = MEGABERTEntityEncoder(
            max_length=self.args.max_length,
            pretrain_path=self.args.pretrain_path,
            mask_entity=False
        )
        Mega_model = SoftmaxNN(Mega_sentence_encoder, len(rel2id), rel2id)
        Mega_framework = SentenceRE(
            train_path=None,
            train_rel_path=None,
            train_pic_path=None,
            val_path=None,
            val_rel_path=None,
            val_pic_path=None,
            test_path=self.args.test_file,
            test_rel_path=self.args.rel_test_file,
            test_pic_path=self.args.pic_test_file,
            model=Mega_model,
            ckpt=None,
            batch_size=16,
            max_epoch=10,
            lr=0.1,
            opt='sgd'
        )

        MTB_sentence_encoder = BERTEntityEncoder(
            max_length=self.args.max_length,
            pretrain_path=self.args.pretrain_path,
            mask_entity=False
        )
        MTB_model = SoftmaxNN(MTB_sentence_encoder, len(rel2id), rel2id)
        MTB_framework = SentenceRE(
            train_path=None,
            train_rel_path=None,
            train_pic_path=None,
            val_path=None,
            val_rel_path=None,
            val_pic_path=None,
            test_path=None,
            test_rel_path=None,
            test_pic_path=None,
            model=MTB_model,
            ckpt=None,
            batch_size=16,
            max_epoch=10,
            lr=0.1,
            opt='sgd'
        )

        Mega_framework.load_state_dict(torch.load(output_mega_file)['state_dict'])
        MTB_framework.load_state_dict(torch.load(output_mtb_file)['state_dict'])

        result_prob = []

        with open(prob_file) as f:
            f_l = f.readlines()[0][1:-1]
            f_l = f_l.split(",")

            for i in f_l:
                temp = float(i.strip())
                result_prob.append(temp)

        test_dataloder = Mega_framework.test_loader
        all_label_test = []
        all_indexed_tokens_test = []
        all_att_mask_test = []
        all_pos1_test = []
        all_pos2_test = []
        all_pic_feat_test = []
        all_rel_feat_test = []
        all_clip_input_ids_test = []
        all_clip_attention_mask_test = []
        all_clip_pixel_values_test = []
        for i, batch_data in enumerate(test_dataloder):
            label = batch_data[0]  # B
            indexed_tokens = batch_data[1]  # B * L
            att_mask = batch_data[2]  # B * L
            pos1 = batch_data[3]  # B * 1
            pos2 = batch_data[4]  # B * 1
            pic_feat = batch_data[5]  # (B * P_L) * L
            rel_feat = batch_data[6]  # (B * rel_L) * L
            clip_input_ids = batch_data[7]  # B * L
            clip_attention_mask = batch_data[8]  # B * L
            clip_pixel_values = batch_data[9]  # B * 3 * 224 * 224

            all_label_test.append(label)
            all_indexed_tokens_test.append(indexed_tokens)
            all_att_mask_test.append(att_mask)
            all_pos1_test.append(pos1)
            all_pos2_test.append(pos2)
            all_pic_feat_test.append(pic_feat)
            all_rel_feat_test.append(rel_feat)
            all_clip_input_ids_test.append(clip_input_ids)
            all_clip_attention_mask_test.append(clip_attention_mask)
            all_clip_pixel_values_test.append(clip_pixel_values)

        all_label_test = torch.cat(all_label_test, dim=0)  # B*x
        all_indexed_tokens_test = torch.cat(all_indexed_tokens_test, dim=0)  # B * L
        all_att_mask_test = torch.cat(all_att_mask_test, dim=0)  # B * L
        all_pos1_test = torch.cat(all_pos1_test, dim=0)  # B * 1
        all_pos2_test = torch.cat(all_pos2_test, dim=0)  # B * 1

        all_pic_feat_test = torch.cat(all_pic_feat_test, dim=0)  # B*x * L
        all_pic_feat_test = all_pic_feat_test.reshape(-1, 10, 4096)

        all_rel_feat_test = torch.cat(all_rel_feat_test, dim=0)  # B*x * L
        all_rel_feat_test = all_rel_feat_test.reshape(-1, 10, 128)

        all_clip_input_ids_test = torch.cat(all_clip_input_ids_test, dim=0)  # B*x * L
        all_clip_attention_mask_test = torch.cat(all_clip_attention_mask_test, dim=0)  # B*x * L
        all_clip_pixel_values_test = torch.cat(all_clip_pixel_values_test, dim=0)  # B * 3 * 224 * 224

        test_data = TensorDataset(all_label_test, all_indexed_tokens_test, all_att_mask_test, all_pos1_test,
                                  all_pos2_test, all_pic_feat_test, all_rel_feat_test, all_clip_input_ids_test,
                                  all_clip_attention_mask_test, all_clip_pixel_values_test)

        test_dataloader = DataLoader(test_data, shuffle=False,
                                     batch_size=self.args.eval_batch_size)

        logger.info("***** Running Estimator Test evaluation *****")
        logger.info("  Num examples = %d", len(test_data))
        logger.info("  Batch size = %d", self.args.eval_batch_size)



        thredshold = self.args.eval_threshold
        eval_result = torch.tensor(result_prob, dtype=torch.float32)
        sampler = Sampler(eval_result)
        test_result_select_index, test_result_unselect_index = sampler.get_result_thredshold(thredshold)

        logger.info(" Length of test_result_select_index = %d  ", len(test_result_select_index))
        logger.info(" Length of test_result_unselect_index = %d  ", len(test_result_unselect_index))

        test_selected_data = TensorDataset(all_label_test[test_result_select_index],
                                           all_indexed_tokens_test[test_result_select_index],
                                           all_att_mask_test[test_result_select_index],
                                           all_pos1_test[test_result_select_index],
                                           all_pos2_test[test_result_select_index],
                                           all_pic_feat_test[test_result_select_index],
                                           all_rel_feat_test[test_result_select_index])

        test_unselected_data = TensorDataset(all_label_test[test_result_unselect_index],
                                             all_indexed_tokens_test[test_result_unselect_index],
                                             all_att_mask_test[test_result_unselect_index],
                                             all_pos1_test[test_result_unselect_index],
                                             all_pos2_test[test_result_unselect_index],
                                             all_pic_feat_test[test_result_unselect_index],
                                             all_rel_feat_test[test_result_unselect_index])

        test_selected_dataloder = DataLoader(test_selected_data, shuffle=False, batch_size=self.args.eval_batch_size)
        test_unselected_dataloder = DataLoader(test_unselected_data, shuffle=False, batch_size=self.args.eval_batch_size)

        logger.info(" ******** Test - Eval Mega In Multi_data ********")
        mega_true_list, mega_pred_list, mega_in_multi_result = Mega_framework.eval_model(test_selected_dataloder)
        logger.info("Test Mega Model In Multimodal Dataset Accuracy = %s, Micro_p = %s, Micro_r = %s, Micro_f1 = %s  ",
                    str(mega_in_multi_result['acc']), str(mega_in_multi_result['micro_p']),
                    str(mega_in_multi_result['micro_r']),
                    str(mega_in_multi_result['micro_f1']))

        logger.info(" ******** Test - Eval Mega In Text_data ********")
        _, __, mega_in_text_result = Mega_framework.eval_model(test_unselected_dataloder)
        logger.info("Test Mega Model In Text Dataset Accuracy = %s, Micro_p = %s, Micro_r = %s, Micro_f1 = %s  ",
                    str(mega_in_text_result['acc']), str(mega_in_text_result['micro_p']),
                    str(mega_in_text_result['micro_r']),
                    str(mega_in_text_result['micro_f1']))

        logger.info(" ******** Test - Eval MTB In Multi_data ********")
        _, __, mtb_in_multi_result = MTB_framework.eval_model(test_selected_dataloder)
        logger.info("Test MTB Model In Multimodal Dataset Accuracy = %s, Micro_p = %s, Micro_r = %s, Micro_f1 = %s  ",
                    str(mtb_in_multi_result['acc']), str(mtb_in_multi_result['micro_p']),
                    str(mtb_in_multi_result['micro_r']),
                    str(mtb_in_multi_result['micro_f1']))

        logger.info(" ******** Test - Eval MTB In Text_data ********")
        mtb_true_list, mtb_pred_list, mtb_in_text_result = MTB_framework.eval_model(test_unselected_dataloder)
        logger.info("Test MTB Model In Text Dataset Accuracy = %s, Micro_p = %s, Micro_r = %s, Micro_f1 = %s  ",
                    str(mtb_in_text_result['acc']), str(mtb_in_text_result['micro_p']),
                    str(mtb_in_text_result['micro_r']),
                    str(mtb_in_text_result['micro_f1']))

        logger.info(" ********** Union Prediction *********")

        result = self.eval_union(mega_true_list, mega_pred_list, mtb_true_list, mtb_pred_list)

        logger.info(" union_acc:{}  union_micro_p:{}  union_micro_r:{}  union_micro_f1:{} ".format(
            str(result['acc']), str(result['micro_p']), str(result['micro_r']), str(result['micro_f1']))
        )


