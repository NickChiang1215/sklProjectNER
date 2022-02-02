# -*- coding: utf-8 -*-
import inspect
import os
import sys

import _jsonnet

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import json
import torch.nn as nn
import torch
import datetime
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from processors.ner_seq_joint import collate_fn
from transformers import BertConfig, BertTokenizer
from processors.ner_seq_joint import ner_processors as processors
from processors.ner_seq_joint import InputExample
from models.bert_for_joint import BertCrfSoftmaxForJoint
from processors.utils_ner import get_entities
from processors.ner_seq_joint import convert_examples_to_features


# class NER_post_processor():
#     def __init__(self):
#
#     def post_processing(self, predict_results):
#         new_result = []
#         for json_d in predict_results:
#             shift_cnt = 0
#             json_d['new_sentence'] = json_d['orig_sentence'].replace(' ', '，')
#             json_d['new_tag_seq'] = json_d['model_tag_seq'].split(' ')
#             json_d['new_tag_seq'] = json_d['new_tag_seq'][:len(json_d['new_sentence'])]
#             json_d['new_entities'] = []
#
#             for entity_info in json_d['model_entities']:
#                 entity_name, start_idx, end_idx = entity_info
#                 start_idx += shift_cnt
#                 end_idx += shift_cnt
#                 end_idx += 1
#                 entity_info = [entity_name, start_idx, end_idx]
#
#                 if entity_name in ['金額', '_數量_', '年度']:
#                     json_d, shift_cnt, entity_info = self.get_decode_result(json_d, shift_cnt, entity_info,
#                                                                             self.num_decorder)
#                     if entity_name == '年度':
#                         json_d, shift_cnt, entity_info = self.get_regularized_year(json_d, shift_cnt, entity_info)
#                 elif entity_name in self.entity_link_decorder.keys():
#                     json_d, shift_cnt, entity_info = self.get_decode_result(json_d, shift_cnt, entity_info,
#                                                                             self.entity_link_decorder[entity_name])
#                 entity_name, start_idx, end_idx = entity_info # 年度正規化時，可能修正model預測錯誤的entity_name
#                 json_d['new_entities'].append([entity_name, start_idx, end_idx - 1])
#
#             shift_cnt = 0
#             entities_cnt = 1
#             json_d['new_sentence_ratsql'] = json_d['new_sentence']
#             json_d['new_tag_seq_ratsql'] = json_d['new_tag_seq'].copy()
#             json_d['new_entities_ratsql'] = []
#
#             for entity_info in json_d['new_entities']:
#                 entity_name, start_idx, end_idx = entity_info
#                 start_idx += shift_cnt
#                 end_idx += shift_cnt
#                 end_idx += 1
#                 entity_info = [entity_name, start_idx, end_idx]
#
#                 if entity_name in self.entity_add_suffix_tag:
#                     json_d, shift_cnt, entity_info = self.add_suffix_tag(json_d, shift_cnt, entity_info, entities_cnt)
#                     entities_cnt += 1
#                     entity_name, start_idx, end_idx = entity_info
#
#                 json_d['new_entities_ratsql'].append([entity_name, start_idx, end_idx - 1])
#
#             assert len(json_d['new_tag_seq']) == len(json_d['new_sentence'])
#             assert len(json_d['new_tag_seq_ratsql']) == len(json_d['new_sentence_ratsql'])
#
#             json_d['new_tag_seq'] = ' '.join(json_d['new_tag_seq'])
#             json_d['new_tag_seq_ratsql'] = ' '.join(json_d['new_tag_seq_ratsql'])
#             del json_d['id']
#             new_result.append(json_d)
#         return new_result


class Model:
    def __init__(self, exp_config_path):
        exp_config = json.loads(_jsonnet.evaluate_file(exp_config_path))
        # task_name = exp_config["task_name"]
        # WEIGHTS_NAME = exp_config["WEIGHTS_NAME"]
        checkpoint = exp_config["checkpoint"]
        model_name_or_path = exp_config["model_name_or_path"]
        task_name = exp_config["task_name"]
        self.device = exp_config["device"]
        os.environ["CUDA_VISIBLE_DEVICES"] = self.device
        self.markup = exp_config["markup"]
        self.eval_max_seq_length = exp_config["eval_max_seq_length"]
        self.model_type = exp_config["model_type"]
        do_lower_case = exp_config["do_lower_case"]
        self.batch_size = exp_config["batch_size"]

        self.relative_year = {'今年': 0, '去年': -1, '前年': -2, '大前年': -3}
        # self.post_processor = NER_post_processor()

        task_name = task_name.lower()
        if task_name not in processors:
            raise ValueError("Task not found: %s" % (exp_config["task_name"]))
        processor = processors[task_name]()
        self.label_list = processor.get_labels()
        self.visit_label_list = processor.get_visit_labels()
        self.ride_label_list = processor.get_ride_labels()
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.visitid2label = {i: label for i, label in enumerate(self.visit_label_list)}
        self.rideid2label = {i: label for i, label in enumerate(self.ride_label_list)}
        num_labels = len(self.label_list)

        config_class, model_class, tokenizer_class = BertConfig, BertCrfSoftmaxForJoint, BertTokenizer
        config = config_class.from_pretrained(model_name_or_path)
        config.update({"ner_num_labels": num_labels,
                       "visit_num_labels": len(self.visit_label_list),
                       "ride_num_labels": len(self.ride_label_list)})
        self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
                                                         do_lower_case=do_lower_case, )

        self.model = model_class.from_pretrained(checkpoint, config=config)
        self.model.to(self.device)
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module

    def process_input_data(self, input_cases):
        examples = []
        for i, case in enumerate(input_cases):
            char = []
            # 醫囑
            char += list(case["doc_comment"].replace(' ', '，'))
            char += ["[SEP]"]
            # 病名
            char += list(case["disease_name"].replace(' ', '，'))

            examples.append(
                InputExample(guid=f'predict_{i}', text_a=[token for token in char], labels=['O'] * len(char),
                             visit_labels=[], ride_label=[]))

        # features = convert_examples_to_features(examples=examples,
        #                                         tokenizer=self.tokenizer,
        #                                         label_list=self.label_list,
        #                                         max_seq_length=self.eval_max_seq_length,
        #                                         cls_token_at_end=bool(self.model_type in ["xlnet"]),
        #                                         pad_on_left=bool(self.model_type in ['xlnet']),
        #                                         cls_token=self.tokenizer.cls_token,
        #                                         cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
        #                                         sep_token=self.tokenizer.sep_token,
        #                                         # pad on the left for xlnet
        #                                         pad_token=
        #                                         self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
        #                                         pad_token_segment_id=4 if self.model_type in ['xlnet'] else 0,
        #                                         )
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=self.tokenizer,
                                                label_list=self.label_list,
                                                visit_label_list=self.visit_label_list,
                                                ride_label_list=self.ride_label_list,
                                                max_seq_length=self.eval_max_seq_length,
                                                cls_token_at_end=bool(self.model_type in ["xlnet"]),
                                                pad_on_left=bool(self.model_type in ['xlnet']),
                                                cls_token=self.tokenizer.cls_token,
                                                cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
                                                sep_token=self.tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                pad_token=
                                                self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if self.model_type in ['xlnet'] else 0,
                                                )
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        # dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)


        all_visit_label_ids = torch.tensor([f.visit_label_ids for f in features], dtype=torch.float)
        all_ride_label_id = torch.tensor([f.ride_label_id for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids,
                                all_visit_label_ids, all_ride_label_id)
        return dataset

    def inference(self, input_cases):
        print("input_cases", input_cases)
        dataset = self.process_input_data(input_cases)
        predict_sampler = SequentialSampler(dataset)
        predict_dataloader = DataLoader(dataset, sampler=predict_sampler, batch_size=self.batch_size,
                                        collate_fn=collate_fn)
        results = []

        for step, batch in enumerate(predict_dataloader):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None,
                          "visit_labels": batch[5], "ride_labels": batch[6]}
                if self.model_type != "distilbert":
                    # XLM and RoBERTa don"t use segment_ids
                    inputs["token_type_ids"] = (batch[2] if self.model_type in ["bert", "xlnet"] else None)
                outputs = self.model(**inputs)
                logits = outputs[0]
                visit_probs = nn.Sigmoid()(outputs[1])
                ride_logits = nn.Softmax(dim=1)(outputs[2])
                tags = self.model.crf.decode(logits, inputs['attention_mask'])
                tags = tags.squeeze(0).cpu().numpy().tolist()
            input_len = batch[4].cpu().numpy().tolist()

            for batch_idx, (tag, len_, visit, ride) in enumerate(zip(tags, input_len, visit_probs, ride_logits)):
                pred = tag[1:len_ - 1]  # [CLS]XXXX[SEP]
                label_entities = get_entities(pred, self.id2label, self.markup)
                json_d = {}
                json_d['id'] = f'step{step}_batchid{batch_idx}'
                # json_d['model_tag_seq'] = " ".join([self.id2label[x] for x in pred if self.id2label[x] != 'X'])
                ner_result = {}
                original_sentence = input_cases[step * self.batch_size + batch_idx]["doc_comment"]
                year = datetime.datetime.now().year
                for label, startIdx, endIdx in label_entities:
                    if label not in ner_result:
                        ner_result[label] = []
                    if endIdx - startIdx == 6:
                        date = original_sentence[startIdx:endIdx + 1]
                        year = original_sentence[startIdx:startIdx + 3]
                        mon = original_sentence[startIdx + 3:startIdx + 5]
                        day = original_sentence[startIdx + 5:startIdx + 7]
                    else:
                        date = str(year) + original_sentence[startIdx:endIdx + 1]
                        mon = original_sentence[startIdx:startIdx + 2]
                        day = original_sentence[startIdx + 2:startIdx + 4]
                    ner_result[label].append(date)
                json_d['visit_type_result'] = [self.visitid2label[idx] for idx, prob in enumerate(visit) if
                                               prob > 0.5]  # TODO: find max f1 score threshold
                json_d['ride_type_result'] = self.rideid2label[torch.argmax(ride).item()]
                json_d['ner_result'] = ner_result

                # json_d['model_entities'] = label_entities
                results.append(json_d)

        # processed_result = self.post_processor.post_processing(results)
        processed_result = results
        return processed_result


if __name__ == "__main__":
    exp_config_path = "./localtest.jsonnet"
    model = Model(exp_config_path)
    print(model.inference([{"doc_comment": '起1091023至1100305共9次', "disease_name": '多處挫傷右上臂右食指左膝及下肢瘀腫痛'},
                           {"doc_comment": '1100804接受胸腔鏡右中及右下肺葉切除手術合併淋巴擴清術,1100727 11:51:36至1100811 '
                                           '10:04:20住院檢查及治療共16日0804轉加護病房,0806轉普通病房1100825接受胸腔鏡肋膜剝脫手'
                                           '術及胸壁清創手術,1100823 17:04:42至1100823 18:49:58急診室檢查及治療共1日,1100823 '
                                           '18:32:28至1100824 14:50:11觀察室檢查及治療共2日,1100824 14:53:07至1100903 '
                                           '11:34:31住院檢查及治療共11日依病歷記錄,患者接受於1100817、1100907、1100914、'
                                           '1100928之本院門診追蹤治療,共計4次', "disease_name": '右中及右下肺癌;右側肋膜膿胸'},
                           {"doc_comment": '於1101124 10:09,經緊急醫療救護車送至本院急診求診,經診治後,於同日11:41出'
                                           '院宜持續門診追蹤治療', "disease_name": '臀部挫傷'}]))
