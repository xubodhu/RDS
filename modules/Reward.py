from __future__ import absolute_import, division, print_function
import numpy as np
import random

import torch

from my_bert.tokenization import BertTokenizer
from modules.preprocess import *
from modules.Partitioner import *
from transformers.models.clip import CLIPProcessor, CLIPModel
from modules.preprocess import *
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from modules.Eval import  Evaluater


class RL(object):

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.evaluater = Evaluater(logger=self.logger)
        self.eval_umt = self.evaluater.eval_umt
        self.eval_bert = self.evaluater.eval_bert
        self.eval_result = self.evaluater.eval_result

    # Start the entire reinforcement learning training process
    def train(self):
        # Print important hyperparameters
        self.logger.info(" estimator_batch %s ", str(self.args.estimator_batch))
        self.logger.info(" estimator_learning_rate %s ", str(self.args.estimator_learning_rate))
        self.logger.info(" seed %s ", str(self.args.seed))
        self.logger.info(" alpha %s ", str(self.args.alpha))

        alpha = self.args.alpha
        device = self.args.device
        output_config_file = self.args.output_config_file
        output_encoder_file = self.args.output_encoder_file
        output_estimator_file = self.args.output_estimator_file
        output_umt_file = self.args.output_umt_file
        output_bert_file = self.args.output_bert_file

        processor = MNERProcessor()
        label_list = processor.get_labels()
        auxlabel_list = processor.get_auxlabels()
        num_labels = len(label_list) + 1  # label 0 corresponds to padding, label in label_list starts from 1
        auxnum_labels = len(auxlabel_list) + 1  # label 0 corresponds to padding, label in label_list starts from 1

        # ''' initialization of our conversion matrix, in our implementation, it is a 7*12 matrix initialized as follows:
        trans_matrix = np.zeros((auxnum_labels, num_labels), dtype=float)
        trans_matrix[0, 0] = 1  # pad to pad
        trans_matrix[1, 1] = 1  # O to O
        trans_matrix[2, 2] = 0.25  # B to B-MISC
        trans_matrix[2, 4] = 0.25  # B to B-PER
        trans_matrix[2, 6] = 0.25  # B to B-ORG
        trans_matrix[2, 8] = 0.25  # B to B-LOC
        trans_matrix[3, 3] = 0.25  # I to I-MISC
        trans_matrix[3, 5] = 0.25  # I to I-PER
        trans_matrix[3, 7] = 0.25  # I to I-ORG
        trans_matrix[3, 9] = 0.25  # I to I-LOC
        trans_matrix[4, 10] = 1  # X to X
        trans_matrix[5, 11] = 1  # [CLS] to [CLS]
        trans_matrix[6, 12] = 1  # [SEP] to [SEP]

        tokenizer = BertTokenizer.from_pretrained(self.args.bert_model, do_lower_case=self.args.do_lower_case)
        clip_preprocessor = CLIPProcessor.from_pretrained(self.args.clip_dir)

        # Training data set
        train_examples = processor.get_train_examples(self.args.data_dir)

        train_features = convert_mm_examples_to_features(
            train_examples, label_list, auxlabel_list, self.args.max_seq_length, tokenizer, self.args.crop_size,
            self.args.path_image, clip_preprocessor)

        # Training set features
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in train_features])
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in train_features], dtype=torch.long)
        all_clip_input_ids = torch.tensor([f.clip_input_ids for f in train_features], dtype=torch.long)
        all_clip_input_mask = torch.tensor([f.clip_input_mask for f in train_features], dtype=torch.long)
        all_clip_pixel_values = torch.tensor([f.clip_pixel_values for f in train_features], dtype=torch.float32)


        # Validation data set
        eval_examples = processor.get_dev_examples(self.args.data_dir)
        eval_features = convert_mm_examples_to_features(
            eval_examples, label_list, auxlabel_list, self.args.max_seq_length, tokenizer, self.args.crop_size,
            self.args.path_image, clip_preprocessor)

        # Validation set features
        eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        eval_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
        eval_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_img_feats = torch.stack([f.img_feat for f in eval_features])
        eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_auxlabel_ids = torch.tensor([f.auxlabel_id for f in eval_features], dtype=torch.long)
        eval_clip_input_ids = torch.tensor([f.clip_input_ids for f in eval_features], dtype=torch.long)
        eval_clip_input_mask = torch.tensor([f.clip_input_mask for f in eval_features], dtype=torch.long)
        eval_clip_pixel_values = torch.tensor([f.clip_pixel_values for f in eval_features], dtype=torch.float32)

        eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_added_input_mask, \
                                  eval_segment_ids, eval_img_feats, eval_label_ids, eval_auxlabel_ids,
                                  eval_clip_input_ids, eval_clip_input_mask, eval_clip_pixel_values)

        # Test data set
        test_examples = processor.get_test_examples(self.args.data_dir)
        test_features = convert_mm_examples_to_features(
            test_examples, label_list, auxlabel_list, self.args.max_seq_length, tokenizer, self.args.crop_size,
            self.args.path_image, clip_preprocessor)

        # Test set features
        test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        test_added_input_mask = torch.tensor([f.added_input_mask for f in test_features], dtype=torch.long)
        test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        test_img_feats = torch.stack([f.img_feat for f in test_features])
        test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        test_auxlabel_ids = torch.tensor([f.auxlabel_id for f in test_features], dtype=torch.long)
        test_clip_input_ids = torch.tensor([f.clip_input_ids for f in test_features], dtype=torch.long)
        test_clip_input_mask = torch.tensor([f.clip_input_mask for f in test_features], dtype=torch.long)
        test_clip_pixel_values = torch.tensor([f.clip_pixel_values for f in test_features], dtype=torch.float32)

        test_data = TensorDataset(test_input_ids, test_input_mask, test_added_input_mask, \
                                  test_segment_ids, test_img_feats, test_label_ids, test_auxlabel_ids,
                                  test_clip_input_ids, test_clip_input_mask, test_clip_pixel_values)


        clip_model = CLIPModel.from_pretrained(self.args.clip_dir)
        estimator = Estimator(clip_model)
        estimator.to(device)


        optimizer_grouped_parameters = [
            {'params': estimator.classfier.parameters()}
        ]
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args.estimator_learning_rate)


        best_union_f1 = 0

        for i in trange(int(self.args.num_estimator_epochs)):
            torch.cuda.empty_cache()
            # Shuffle the data to get `self.args.estimator_batch` random numbers for randomly selecting the data.
            batch_idx = random.sample([i for i in range(all_input_ids.shape[0])], self.args.estimator_batch)  # batch

            batch_input_ids = all_input_ids[batch_idx]  # batch
            batch_input_mask = all_input_mask[batch_idx]
            batch_added_input_mask = all_added_input_mask[batch_idx]
            batch_segment_ids = all_segment_ids[batch_idx]
            batch_img_feats = all_img_feats[batch_idx]
            batch_label_ids = all_label_ids[batch_idx]
            batch_auxlabel_ids = all_auxlabel_ids[batch_idx]
            batch_clip_input_ids = all_clip_input_ids[batch_idx]
            batch_clip_input_mask = all_clip_input_mask[batch_idx]
            batch_clip_pixel_values = all_clip_pixel_values[batch_idx]
            self.logger.info(" Sample Index Length = %d ", len(batch_idx))
            self.logger.info(" Data Length = %d ", batch_input_ids.shape[0])

            batch_input_ids = batch_input_ids.to(device)
            batch_input_mask = batch_input_mask.to(device)
            batch_added_input_mask = batch_added_input_mask.to(device)
            batch_segment_ids = batch_segment_ids.to(device)
            batch_img_feats = batch_img_feats.to(device)
            batch_label_ids = batch_label_ids.to(device)
            batch_auxlabel_ids = batch_auxlabel_ids.to(device)
            batch_clip_input_ids = batch_clip_input_ids.to(device)
            batch_clip_input_mask = batch_clip_input_mask.to(device)
            batch_clip_pixel_values = batch_clip_pixel_values.to(device)

            # Returns the probability that an image needs to be added to each data: (batch, 1)
            estimator.train()
            probs = estimator(input_ids=batch_clip_input_ids, input_mask=batch_clip_input_mask,
                              pixel_values=batch_clip_pixel_values)  # (batch,1)
            self.logger.info("  Probs Shape =  %s", probs.shape)
            # Define the sampler to select the data to be trained according to the probability of the output, initialize the sampler
            sampler = Sampler(probs)
            thredshold = self.args.train_threshold
            select_idx, unselect_idx, action_holder = sampler.get_index(thredshold)
            self.logger.info(" Length of Select_idx = %d  ", len(select_idx))
            self.logger.info(" Length of Unselect_idx = %d  ", len(unselect_idx))
            self.logger.info(" Length of Action_holder = %d ", len(action_holder))

            select_idx = torch.tensor(select_idx, dtype=torch.long).to(device)
            unselect_idx = torch.tensor(unselect_idx, dtype=torch.long).to(device)

            action_holder = torch.from_numpy(np.array(action_holder)).float()
            action_holder = action_holder.unsqueeze(1).to(device)

            # Clear memory cache
            umt_batch_input_ids = torch.index_select(batch_input_ids, dim=0, index=select_idx).cpu()
            bert_batch_input_ids = torch.index_select(batch_input_ids, dim=0, index=unselect_idx).cpu()
            batch_input_ids = batch_input_ids.cpu()
            torch.cuda.empty_cache()
            umt_batch_input_mask = torch.index_select(batch_input_mask, dim=0,
                                                      index=select_idx).cpu()  # (selected_batch, 128)
            bert_batch_input_mask = torch.index_select(batch_input_mask, dim=0,
                                                       index=unselect_idx).cpu()  # (unselected_batch, 128)
            batch_input_mask = batch_input_mask.cpu()
            torch.cuda.empty_cache()
            umt_batch_added_input_mask = torch.index_select(batch_added_input_mask, dim=0, index=select_idx).cpu()
            bert_batch_added_input_mask = torch.index_select(batch_added_input_mask, dim=0, index=unselect_idx).cpu()
            batch_added_input_mask = batch_added_input_mask.cpu()
            torch.cuda.empty_cache()
            umt_batch_segment_ids = torch.index_select(batch_segment_ids, dim=0, index=select_idx).cpu()
            bert_batch_segment_ids = torch.index_select(batch_segment_ids, dim=0, index=unselect_idx).cpu()
            batch_segment_ids = batch_segment_ids.cpu()
            torch.cuda.empty_cache()
            umt_batch_img_feats = torch.index_select(batch_img_feats, dim=0, index=select_idx).cpu()
            bert_batch_img_feats = torch.index_select(batch_img_feats, dim=0, index=unselect_idx).cpu()
            batch_img_feats = batch_img_feats.cpu()
            torch.cuda.empty_cache()
            umt_batch_label_ids = torch.index_select(batch_label_ids, dim=0, index=select_idx).cpu()
            bert_batch_label_ids = torch.index_select(batch_label_ids, dim=0, index=unselect_idx).cpu()
            batch_label_ids = batch_label_ids.cpu()
            torch.cuda.empty_cache()
            umt_batch_auxlabel_ids = torch.index_select(batch_auxlabel_ids, dim=0, index=select_idx).cpu()
            bert_batch_auxlabel_ids = torch.index_select(batch_auxlabel_ids, dim=0, index=unselect_idx).cpu()
            batch_auxlabel_ids = batch_auxlabel_ids.cpu()
            torch.cuda.empty_cache()

            # Package the data features that are selected to be added to the image [Training]
            umt_train_data = TensorDataset(umt_batch_input_ids, umt_batch_input_mask, umt_batch_added_input_mask, \
                                           umt_batch_segment_ids, umt_batch_img_feats, umt_batch_label_ids,
                                           umt_batch_auxlabel_ids)

            # Packing of data features that are not selected as images that should be added [Training]
            train_data_crf = TensorDataset(bert_batch_input_ids, bert_batch_input_mask, bert_batch_added_input_mask,
                                           bert_batch_segment_ids,
                                           bert_batch_img_feats, bert_batch_label_ids, bert_batch_auxlabel_ids)


            self.logger.info(" ******** Eval UMT In Multimodal Data ********")
            umt_in_umt_f1 = self.eval_umt(self.args, umt_train_data, output_config_file, output_umt_file,
                                          output_encoder_file, num_labels, auxnum_labels, label_list, processor,
                                          trans_matrix)
            self.logger.info(" Training umt_in_multimodal_f1 = %s  ", str(umt_in_umt_f1))


            self.logger.info(" ******** Eval UMT In Unimodal Data ********")
            umt_in_bert_f1 = self.eval_umt(self.args, train_data_crf, output_config_file, output_umt_file,
                                           output_encoder_file, num_labels, auxnum_labels, label_list, processor,
                                           trans_matrix)
            self.logger.info(" Training umt_in_unimodal_f1 = %s  ", str(umt_in_bert_f1))


            self.logger.info(" ******** Eval BERT In Unimodal Data ********")
            bert_in_bert_f1 = self.eval_bert(self.args, train_data_crf, output_config_file, output_bert_file, num_labels,
                                             label_list, processor)
            self.logger.info(" Training bert_in_unimodal_f1 = %s  ", str(bert_in_bert_f1))


            self.logger.info(" ******** Eval BERT In Multimodal Data ********")
            bert_in_umt_f1 = self.eval_bert(self.args, umt_train_data, output_config_file, output_bert_file, num_labels,
                                            label_list, processor)
            self.logger.info(" Training bert_in_multimodal_f1 = %s  ", str(bert_in_umt_f1))

            delta1 = umt_in_umt_f1 - bert_in_umt_f1
            self.logger.info("  delta1 = %s  ", str(delta1))
            delta2 = bert_in_bert_f1 - umt_in_bert_f1
            self.logger.info("  delta2 = %s  ", str(delta2))
            reward = alpha * delta1 + (1 - alpha) * delta2
            self.logger.info("  reward = %s  ", str(reward))

            reward_holder = torch.from_numpy(np.array([reward])).float()
            self.logger.info(" reward_holder = %s", str(reward_holder))
            reward_holder = reward_holder.to(device)
            pi = action_holder * probs + (1.0 - action_holder) * (1.0 - probs)
            loss = -1.0 * torch.sum(torch.log(pi) * reward_holder)

            self.logger.info("   **** loss = %s ****  ", str(loss.tolist()))
            loss.backward()
            optimizer.step()

            # Evaluation of estimator
            estimator.eval()
            self.logger.info("***** Running Estimator Eval evaluation *****")

            eval_result = []
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

            for input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, \
                auxlabel_ids, clip_input_ids, clip_input_mask, clip_pixel_values \
                    in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                img_feats = img_feats.to(device)
                clip_input_ids = clip_input_ids.to(device)
                clip_input_mask = clip_input_mask.to(device)
                clip_pixel_values = clip_pixel_values.to(device)
                with torch.no_grad():
                    predicted_probs = estimator(input_ids=clip_input_ids, input_mask=clip_input_mask,
                                                pixel_values=clip_pixel_values)

                    eval_result.extend(predicted_probs)

            thredshold = self.args.eval_threshold

            eval_result = torch.tensor(eval_result, dtype=torch.float32)
            sampler.probs = eval_result
            eval_result_select_index, eval_result_unselect_index = sampler.get_result_index(thredshold)

            self.logger.info(" Length of eval_result_select_index = %d  ", len(eval_result_select_index))
            self.logger.info(" Length of eval_result_unselect_index = %d  ", len(eval_result_unselect_index))

            eval_selected_data = TensorDataset(eval_input_ids[eval_result_select_index],
                                               eval_input_mask[eval_result_select_index],
                                               eval_added_input_mask[eval_result_select_index],
                                               eval_segment_ids[eval_result_select_index],
                                               eval_img_feats[eval_result_select_index],
                                               eval_label_ids[eval_result_select_index],
                                               eval_auxlabel_ids[eval_result_select_index])

            eval_unselected_data = TensorDataset(eval_input_ids[eval_result_unselect_index],
                                                 eval_input_mask[eval_result_unselect_index],
                                                 eval_added_input_mask[eval_result_unselect_index],
                                                 eval_segment_ids[eval_result_unselect_index],
                                                 eval_img_feats[eval_result_unselect_index],
                                                 eval_label_ids[eval_result_unselect_index],
                                                 eval_auxlabel_ids[eval_result_unselect_index])


            self.logger.info(" ******** Eval UMT In Selected_data ********")
            umt_selected_f1 = self.eval_umt(self.args, eval_selected_data, output_config_file, output_umt_file,
                                            output_encoder_file, num_labels, auxnum_labels, label_list, processor,
                                            trans_matrix)
            self.logger.info(" Evaluation umt_selected_f1 = %s  ", str(umt_selected_f1))

            self.logger.info(" ******** Eval UMT In Unselected_data ********")
            umt_unselected_f1 = self.eval_umt(self.args, eval_unselected_data, output_config_file, output_umt_file,
                                              output_encoder_file, num_labels, auxnum_labels, label_list, processor,
                                              trans_matrix)
            self.logger.info(" Evaluation umt_unselected_f1 = %s  ", str(umt_unselected_f1))

            self.logger.info(" ******** Eval BERT In Selected_data ********")
            bert_selected_f1 = self.eval_bert(self.args, eval_selected_data, output_config_file, output_bert_file,
                                              num_labels, label_list, processor)
            self.logger.info(" Evaluation bert_selected_f1 = %s  ", str(bert_selected_f1))
            self.logger.info(" ******** Eval BERT In Unselected_data ********")
            bert_unselected_f1 = self.eval_bert(self.args, eval_unselected_data, output_config_file, output_bert_file, \
                                                num_labels, label_list, processor)
            self.logger.info(" Evaluation bert_unselected_f1 = %s  ", str(bert_unselected_f1))

            # After training the model, perform joint ground prediction
            self.logger.info(" ********** Union Prediction *********")
            umt_f1, bert_f1, union_f1 = self.eval_result(self.args, eval_selected_data, eval_unselected_data,
                                                         output_config_file, output_umt_file,
                                                         output_bert_file, output_encoder_file, num_labels,
                                                         auxnum_labels, label_list, processor,
                                                         eval_result_select_index, eval_result_unselect_index,
                                                         trans_matrix)

            self.logger.info(" Evaluation umt_selected_f1:{}  bert_selected_f1: {}  ".format(str(umt_selected_f1), str(bert_selected_f1)))
            self.logger.info(" Evaluation umt_f1:{}  bert_f1:{}  union_f1:{}  ".format(str(umt_f1), str(bert_f1), str(union_f1)))

            if union_f1 > best_union_f1:
                best_union_f1 = union_f1
                torch.save(estimator.state_dict(), output_estimator_file)



    def test(self, umt_file, bert_file, prob_file):
        result_prob = []

        with open(prob_file) as f:
            f_l = f.readlines()[0][1:-1]
            f_l = f_l.split(",")


            for i in f_l:
                temp = float(i.strip())
                result_prob.append(temp)

        output_config_file = self.args.output_config_file
        output_encoder_file = self.args.output_encoder_file
        output_umt_file = umt_file
        output_bert_file = bert_file

        processor = MNERProcessor()
        label_list = processor.get_labels()
        auxlabel_list = processor.get_auxlabels()
        num_labels = len(label_list) + 1  # label 0 corresponds to padding, label in label_list starts from 1
        auxnum_labels = len(auxlabel_list) + 1  # label 0 corresponds to padding, label in label_list starts from 1

        # ''' initialization of our conversion matrix, in our implementation, it is a 7*12 matrix initialized as follows:
        trans_matrix = np.zeros((auxnum_labels, num_labels), dtype=float)
        trans_matrix[0, 0] = 1  # pad to pad
        trans_matrix[1, 1] = 1  # O to O
        trans_matrix[2, 2] = 0.25  # B to B-MISC
        trans_matrix[2, 4] = 0.25  # B to B-PER
        trans_matrix[2, 6] = 0.25  # B to B-ORG
        trans_matrix[2, 8] = 0.25  # B to B-LOC
        trans_matrix[3, 3] = 0.25  # I to I-MISC
        trans_matrix[3, 5] = 0.25  # I to I-PER
        trans_matrix[3, 7] = 0.25  # I to I-ORG
        trans_matrix[3, 9] = 0.25  # I to I-LOC
        trans_matrix[4, 10] = 1  # X to X
        trans_matrix[5, 11] = 1  # [CLS] to [CLS]
        trans_matrix[6, 12] = 1  # [SEP] to [SEP]

        tokenizer = BertTokenizer.from_pretrained(self.args.bert_model, do_lower_case=self.args.do_lower_case)
        clip_preprocessor = CLIPProcessor.from_pretrained(self.args.clip_dir)


        test_examples = processor.get_test_examples(self.args.data_dir)
        test_features = convert_mm_examples_to_features(
            test_examples, label_list, auxlabel_list, self.args.max_seq_length, tokenizer, self.args.crop_size,
            self.args.path_image, clip_preprocessor)

        test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        test_added_input_mask = torch.tensor([f.added_input_mask for f in test_features], dtype=torch.long)
        test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        test_img_feats = torch.stack([f.img_feat for f in test_features])
        test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        test_auxlabel_ids = torch.tensor([f.auxlabel_id for f in test_features], dtype=torch.long)
        test_clip_input_ids = torch.tensor([f.clip_input_ids for f in test_features], dtype=torch.long)
        test_clip_input_mask = torch.tensor([f.clip_input_mask for f in test_features], dtype=torch.long)
        test_clip_pixel_values = torch.tensor([f.clip_pixel_values for f in test_features], dtype=torch.float32)

        test_data = TensorDataset(test_input_ids, test_input_mask, test_added_input_mask, \
                                  test_segment_ids, test_img_feats, test_label_ids, test_auxlabel_ids)

        thredshold = self.args.eval_threshold

        eval_result = torch.tensor(result_prob, dtype=torch.float32)

        sampler = Sampler(eval_result)
        test_result_select_index, test_result_unselect_index = sampler.get_result_thredshold(thredshold)

        logger.info(" Length of Eval_result_select_index = %d  ", len(test_result_select_index))
        logger.info(" Length of Eval_result_unselect_index = %d  ", len(test_result_unselect_index))

        test_selected_data = TensorDataset(test_input_ids[test_result_select_index],
                                           test_input_mask[test_result_select_index],
                                           test_added_input_mask[test_result_select_index],
                                           test_segment_ids[test_result_select_index],
                                           test_img_feats[test_result_select_index],
                                           test_label_ids[test_result_select_index],
                                           test_auxlabel_ids[test_result_select_index])

        test_unselected_data = TensorDataset(test_input_ids[test_result_unselect_index],
                                             test_input_mask[test_result_unselect_index],
                                             test_added_input_mask[test_result_unselect_index],
                                             test_segment_ids[test_result_unselect_index],
                                             test_img_feats[test_result_unselect_index],
                                             test_label_ids[test_result_unselect_index],
                                             test_auxlabel_ids[test_result_unselect_index])

        self.logger.info(" ******** Test UMT In Selected_data ********")
        umt_selected_f1 = self.eval_umt(self.args, test_selected_data, output_config_file, output_umt_file,
                                        output_encoder_file, num_labels, auxnum_labels, label_list, processor,
                                        trans_matrix)
        self.logger.info(" umt_selected_f1 = %s  ", str(umt_selected_f1))

        self.logger.info(" ******** Test UMT In Unselected_data ********")
        umt_unselected_f1 = self.eval_umt(self.args, test_unselected_data, output_config_file, output_umt_file,
                                          output_encoder_file, num_labels, auxnum_labels, label_list, processor,
                                          trans_matrix)
        self.logger.info(" umt_unselected_f1 = %s  ", str(umt_unselected_f1))

        self.logger.info(" ******** Test BERT In Selected_data ********")
        bert_selected_f1 = self.eval_bert(self.args, test_selected_data, output_config_file, output_bert_file,
                                          num_labels, label_list, processor)
        self.logger.info(" bert_selected_f1 = %s  ", str(bert_selected_f1))
        self.logger.info(" ******** Test BERT In Unselected_data ********")
        bert_unselected_f1 = self.eval_bert(self.args, test_unselected_data, output_config_file, output_bert_file, \
                                            num_labels, label_list, processor)
        self.logger.info(" bert_unselected_f1 = %s  ", str(bert_unselected_f1))

        self.logger.info(" ********** Union Prediction *********")
        umt_f1, bert_f1, union_f1 = self.eval_result(self.args, test_selected_data, test_unselected_data,
                                                     output_config_file, output_umt_file,
                                                     output_bert_file, output_encoder_file, num_labels,
                                                     auxnum_labels, label_list, processor,
                                                     test_result_select_index, test_result_unselect_index,
                                                     trans_matrix)

        self.logger.info(" umt_selected_f1:{}  bert_selected_f1: {}  ".format(str(umt_selected_f1), str(bert_selected_f1)))
        self.logger.info(" umt_f1:{}  bert_f1:{}  union_f1:{}  ".format(str(umt_f1), str(bert_f1), str(union_f1)))




