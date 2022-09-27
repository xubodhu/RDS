import torch
import os
from my_bert.mner_modeling import BertConfig, MTCCMBertForMMTokenClassificationCRF, BertCRF
import resnet.resnet as resnet
from resnet.resnet_utils import myResnet
from torch.utils.data import DataLoader, SequentialSampler
from seqeval.metrics import classification_report
from ner_evaluate import evaluate_each_class
from ner_evaluate import evaluate
from tqdm import tqdm

class Evaluater(object):
    def __init__(self, logger):
        self.logger = logger
    
    def eval_umt(self, args, eval_data, output_config_file, output_model_file, output_encoder_file, num_labels,
                 auxnum_labels, label_list, processor, trans_matrix):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = BertConfig(output_config_file)
        model = MTCCMBertForMMTokenClassificationCRF(config, layer_num1=1, layer_num2=1,layer_num3=1,
                                                     num_labels=num_labels,auxnum_labels=auxnum_labels)


        net = getattr(resnet, 'resnet152')()
        encoder = myResnet(net, args.fine_tune_cnn, device)
        model.load_state_dict(torch.load(output_model_file))
        model.to(device)
        encoder_state_dict = torch.load(output_encoder_file)
        encoder.load_state_dict(encoder_state_dict)
        encoder.to(device)
        model.eval()
        encoder.eval()

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        y_true = []
        y_pred = []
        y_true_idx = []
        y_pred_idx = []
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        label_map[0] = "PAD"

        for input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids in tqdm(
                eval_dataloader,  desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            added_input_mask = added_input_mask.to(device)
            segment_ids = segment_ids.to(device)
            img_feats = img_feats.to(device)
            label_ids = label_ids.to(device)
            trans_matrix = torch.tensor(trans_matrix).to(device)

            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(img_feats)
                predicted_label_seq_ids = model(input_ids, segment_ids, input_mask, added_input_mask, img_att,
                                                trans_matrix)

            logits = predicted_label_seq_ids
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            for i, mask in enumerate(input_mask):
                temp_1 = []
                temp_2 = []
                tmp1_idx = []
                tmp2_idx = []

                for j, m in enumerate(mask):
                    if j == 0:
                        continue
                    if m:
                        if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "[SEP]":
                            temp_1.append(label_map[label_ids[i][j]])
                            tmp1_idx.append(label_ids[i][j])
                            temp_2.append(label_map[logits[i][j]])
                            tmp2_idx.append(logits[i][j])

                    else:
                        break
                y_true.append(temp_1)
                y_pred.append(temp_2)
                y_true_idx.append(tmp1_idx)
                y_pred_idx.append(tmp2_idx)

        report = classification_report(y_true, y_pred, digits=4)

        sentence_list = []
        test_data, imgs, _ = processor._read_mmtsv(os.path.join(args.data_dir, "test.txt"))

        for i in range(len(y_pred)):
            sentence = test_data[i][0]
            sentence_list.append(sentence)


        reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
        acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)
        self.logger.info("******* UMT Test Eval results ******")
        self.logger.info("\n%s", report)
        self.logger.info("Overall: %s %s %s", p, r, f1)

        return f1



    def eval_bert(self, args, eval_data, output_config_file, output_model_file, num_labels, label_list, processor):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = BertConfig(output_config_file)
        model = BertCRF(config, num_labels=num_labels)

        model.load_state_dict(torch.load(output_model_file))
        model.to(device)
        model.eval()


        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        y_true = []
        y_pred = []
        y_true_idx = []
        y_pred_idx = []
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        label_map[0] = "PAD"

        for input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids in tqdm(
                eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                predicted_label_seq_ids = model(input_ids, segment_ids, input_mask)

            logits = predicted_label_seq_ids
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            for i, mask in enumerate(input_mask):
                temp_1 = []
                temp_2 = []
                tmp1_idx = []
                tmp2_idx = []

                for j, m in enumerate(mask):
                    if j == 0:
                        continue
                    if m:
                        if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "[SEP]":
                            temp_1.append(label_map[label_ids[i][j]])
                            tmp1_idx.append(label_ids[i][j])
                            temp_2.append(label_map[logits[i][j]])
                            tmp2_idx.append(logits[i][j])

                    else:
                        break
                y_true.append(temp_1)
                y_pred.append(temp_2)
                y_true_idx.append(tmp1_idx)
                y_pred_idx.append(tmp2_idx)

        report = classification_report(y_true, y_pred, digits=4)

        sentence_list = []
        test_data, imgs, _ = processor._read_mmtsv(os.path.join(args.data_dir, "test.txt"))

        for i in range(len(y_pred)):
            sentence = test_data[i][0]
            sentence_list.append(sentence)


        reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
        acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)
        self.logger.info("****** BERT Test Eval results ******")
        self.logger.info("Overall: %s %s %s", p, r, f1)


        return f1

    def eval_result(self, args, eval_data_umt, eval_data_bert, output_config_file, output_umt_file, output_bert_file,
                    output_encoder_file, num_labels, auxnum_labels, label_list, processor, eval_selected_index,
                    eval_unselected_index, trans_matrix):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = BertConfig(output_config_file)

        umt = MTCCMBertForMMTokenClassificationCRF(config, layer_num1=1, layer_num2=1, layer_num3=1, num_labels=num_labels, auxnum_labels=auxnum_labels)
        bert_crf = BertCRF(config, num_labels=num_labels)

        net = getattr(resnet, 'resnet152')()

        encoder = myResnet(net, args.fine_tune_cnn, device)
        encoder_state_dict = torch.load(output_encoder_file)
        encoder.load_state_dict(encoder_state_dict)
        encoder.to(device)
        encoder.eval()

        umt.load_state_dict(torch.load(output_umt_file))
        umt.to(device)


        encoder.to(device)
        umt.eval()
        encoder.eval()


        eval_sampler = SequentialSampler(eval_data_umt)
        eval_dataloader = DataLoader(eval_data_umt, sampler=eval_sampler, batch_size=args.eval_batch_size)

        y_true_umt = []
        y_pred_umt = []
        y_true_idx_umt = []
        y_pred_idx_umt = []
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        label_map[0] = "PAD"
        for input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids in tqdm(
                eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            added_input_mask = added_input_mask.to(device)
            segment_ids = segment_ids.to(device)
            img_feats = img_feats.to(device)
            label_ids = label_ids.to(device)
            trans_matrix = torch.tensor(trans_matrix).to(device)

            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(img_feats)
                predicted_label_seq_ids = umt(input_ids, segment_ids, input_mask, added_input_mask, img_att,
                                              trans_matrix)

            logits = predicted_label_seq_ids
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            for i, mask in enumerate(input_mask):
                temp_1 = []
                temp_2 = []
                tmp1_idx = []
                tmp2_idx = []

                for j, m in enumerate(mask):
                    if j == 0:
                        continue
                    if m:
                        if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "[SEP]":
                            temp_1.append(label_map[label_ids[i][j]])
                            tmp1_idx.append(label_ids[i][j])
                            temp_2.append(label_map[logits[i][j]])
                            tmp2_idx.append(logits[i][j])

                    else:
                        break
                y_true_umt.append(temp_1)
                y_pred_umt.append(temp_2)
                y_true_idx_umt.append(tmp1_idx)
                y_pred_idx_umt.append(tmp2_idx)

        report = classification_report(y_true_umt, y_pred_umt, digits=4)
        sentence_list_umt = []
        test_data, imgs, _ = processor._read_mmtsv(os.path.join(args.data_dir, "test.txt"))
        test_data_temp = []
        imgs_temp = []
        for e in eval_selected_index:
            test_data_temp.append(test_data[e])
            imgs_temp.append(imgs[e])


        for i in range(len(y_pred_umt)):
            sentence = test_data[i][0]
            sentence_list_umt.append(sentence)

        reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
        acc, f1, p, r = evaluate(y_pred_idx_umt, y_true_idx_umt, sentence_list_umt, reverse_label_map)
        self.logger.info("****** Union UMT Test Eval results ******")
        self.logger.info("\n%s", report)
        self.logger.info("Overall: %s %s %s", p, r, f1)


        umt_f1 = f1


        umt = umt.cpu()
        encoder = encoder.cpu()

        bert_crf.load_state_dict(torch.load(output_bert_file))
        bert_crf.to(device)
        bert_crf.eval()


        eval_sampler = SequentialSampler(eval_data_bert)
        eval_dataloader = DataLoader(eval_data_bert, sampler=eval_sampler, batch_size=args.eval_batch_size)

        y_true_bert = []
        y_pred_bert = []
        y_true_idx_bert = []
        y_pred_idx_bert = []
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        label_map[0] = "PAD"
        for input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids in tqdm(
                eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            trans_matrix = torch.tensor(trans_matrix).to(device)

            with torch.no_grad():
                predicted_label_seq_ids = bert_crf(input_ids, segment_ids, input_mask)

            logits = predicted_label_seq_ids
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            for i, mask in enumerate(input_mask):
                temp_1 = []
                temp_2 = []
                tmp1_idx = []
                tmp2_idx = []

                for j, m in enumerate(mask):
                    if j == 0:
                        continue
                    if m:
                        if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "[SEP]":
                            temp_1.append(label_map[label_ids[i][j]])
                            tmp1_idx.append(label_ids[i][j])
                            temp_2.append(label_map[logits[i][j]])
                            tmp2_idx.append(logits[i][j])

                    else:
                        break
                y_true_bert.append(temp_1)
                y_pred_bert.append(temp_2)
                y_true_idx_bert.append(tmp1_idx)
                y_pred_idx_bert.append(tmp2_idx)

        report = classification_report(y_true_bert, y_pred_bert, digits=4)
        sentence_list_bert = []
        test_data, imgs, _ = processor._read_mmtsv(os.path.join(args.data_dir, "test.txt"))
        test_data_temp = []
        imgs_temp = []
        for e in eval_unselected_index:
            test_data_temp.append(test_data[e])
            imgs_temp.append(imgs[e])


        for i in range(len(y_pred_bert)):
            sentence = test_data[i][0]
            sentence_list_bert.append(sentence)


        reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
        acc, f1, p, r = evaluate(y_pred_idx_bert, y_true_idx_bert, sentence_list_bert, reverse_label_map)
        self.logger.info("****** Union BERT Test Eval results ******")
        self.logger.info("\n%s", report)
        self.logger.info("Overall: %s %s %s", p, r, f1)


        bert_f1 = f1


        y_true_umt.extend(y_true_bert)
        y_pred_umt.extend(y_pred_bert)
        y_pred_idx_umt.extend(y_pred_idx_bert)
        y_true_idx_umt.extend(y_true_idx_bert)
        sentence_list_umt.extend(sentence_list_bert)


        report = classification_report(y_true_umt, y_pred_umt, digits=4)
        acc, f1, p, r = evaluate(y_pred_idx_umt, y_true_idx_umt, sentence_list_umt, reverse_label_map)
        self.logger.info("***** Union F1 Test Eval results  *****")
        self.logger.info("\n%s", report)
        self.logger.info("Overall: %s %s %s", p, r, f1)

        union_f1 = f1

        return umt_f1, bert_f1, union_f1


