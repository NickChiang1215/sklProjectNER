# -*- coding: utf-8 -*-
import _jsonnet
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import json
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertConfig, BertTokenizer

from processors.utils_ner import bert_extract_item
from processors.utils_ner import get_entities
from models.bert_for_ner import BertSpanForNer
from processors.ner_span import collate_fn
from processors.ner_span import convert_examples_to_features
from processors.ner_span import ner_processors as processors
from processors.ner_span import InputExample
from processors.aigo_ner_decoder import ChineseNumDecoder, Entitylink_Decoder_pinyin_Levenshtein


class NER_post_processor():
    def __init__(self, entity_link_dict=None, entity_add_suffix_tag=[]):
        self.num_decorder = ChineseNumDecoder()
        self.entity_add_suffix_tag = entity_add_suffix_tag
        if entity_link_dict:
            self.entity_link_decorder = {}
            for entity_name, entity_link in entity_link_dict.items():
                self.entity_link_decorder[entity_name] = Entitylink_Decoder_pinyin_Levenshtein(entity_link)

    def get_decode_result(self, json_d, shift_cnt, entity_info, decoder):
        entity_name, start_idx, end_idx = entity_info

        before_transform_string = json_d['new_sentence']
        new_entity = decoder.transform(before_transform_string[start_idx:end_idx])
        json_d['new_sentence'] = before_transform_string[:start_idx] + new_entity + before_transform_string[end_idx:]
        before_transform_tag_seq = json_d['new_tag_seq']
        json_d['new_tag_seq'] = before_transform_tag_seq[:start_idx] + \
                                [f'B-{entity_name}'] + [f'I-{entity_name}'] * (len(new_entity) - 1) + \
                                before_transform_tag_seq[end_idx:]
        entity_len_diff = len(new_entity) - (end_idx - start_idx)
        shift_cnt += entity_len_diff
        end_idx += entity_len_diff

        return json_d, shift_cnt, [entity_name, start_idx, end_idx]

    def add_suffix_tag(self, json_d, shift_cnt, entity_info, entiries_cnt):
        entity_name, start_idx, end_idx = entity_info
        suffix = f'#{entiries_cnt}#'
        len_suffix = len(suffix)
        suffix_label = [f'I-{entity_name}'] * len_suffix
        json_d['new_sentence'] = json_d['new_sentence'][:end_idx] + suffix + json_d['new_sentence'][end_idx:]
        json_d['new_tag_seq'] = json_d['new_tag_seq'][:end_idx] + suffix_label + json_d['new_tag_seq'][end_idx:]

        shift_cnt += len_suffix
        end_idx += len_suffix
        return json_d, shift_cnt, [entity_name, start_idx, end_idx]

    def get_regularized_year(self, json_d, shift_cnt, entity_info):
        entity_name, start_idx, end_idx = entity_info

        before_transform_string = json_d['new_sentence']
        # 西元年轉民國年
        int_str = ''
        for c in before_transform_string[start_idx:end_idx + 1]:
            if c.isdigit():
                int_str += c
            else:
                break

        # 正規化年度
        if int(int_str) > 1911:
            new_entity = str(int(int_str) - 1911) + '年'
        else:
            new_entity = str(int(int_str)) + '年'

        json_d['new_sentence'] = before_transform_string[:start_idx] + new_entity + before_transform_string[end_idx:]
        before_transform_tag_seq = json_d['new_tag_seq']
        json_d['new_tag_seq'] = before_transform_tag_seq[:start_idx] + \
                                [f'B-{entity_name}'] + [f'I-{entity_name}'] * (len(new_entity) - 1) + \
                                before_transform_tag_seq[end_idx:]
        entity_len_diff = len(new_entity) - (end_idx - start_idx)
        shift_cnt += entity_len_diff
        end_idx += entity_len_diff

        return json_d, shift_cnt, [entity_name, start_idx, end_idx]

    def post_processing(self, predict_results):
        new_result = []
        for json_d in predict_results:
            shift_cnt = 0
            entities_cnt = 1
            json_d['new_sentence'] = json_d['orig_sentence'].replace(' ', '')
            json_d['new_tag_seq'] = json_d['model_tag_seq'].split(' ')
            json_d['new_tag_seq'] = json_d['new_tag_seq'][:len(json_d['new_sentence'])]
            json_d['new_entities'] = []

            for entity_info in json_d['model_entities']:
                entity_name, start_idx, end_idx = entity_info
                start_idx += shift_cnt
                end_idx += shift_cnt
                end_idx += 1
                entity_info = [entity_name, start_idx, end_idx]

                if entity_name in ['金額', '_數量_', '年度']:
                    json_d, shift_cnt, entity_info = self.get_decode_result(json_d, shift_cnt, entity_info,
                                                                            self.num_decorder)
                    if entity_name == '年度':
                        json_d, shift_cnt, entity_info = self.get_regularized_year(json_d, shift_cnt, entity_info)
                elif entity_name in self.entity_link_decorder.keys():
                    json_d, shift_cnt, entity_info = self.get_decode_result(json_d, shift_cnt, entity_info,
                                                                            self.entity_link_decorder[entity_name])

                if entity_name in self.entity_add_suffix_tag:
                    json_d, shift_cnt, entity_info = self.add_suffix_tag(json_d, shift_cnt, entity_info, entities_cnt)
                    entities_cnt += 1
                    entity_name, start_idx, end_idx = entity_info
                    json_d['new_entities'].append([entity_name, start_idx, end_idx - 1])

            assert len(json_d['new_tag_seq']) == len(json_d['new_sentence'])
            json_d['new_tag_seq'] = ' '.join(json_d['new_tag_seq'])
            del json_d['id']
            new_result.append(json_d)
        return new_result


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
        # train_max_seq_length = exp_config["train_max_seq_length"]
        self.eval_max_seq_length = exp_config["eval_max_seq_length"]
        self.model_type = exp_config["model_type"]
        do_lower_case = exp_config["do_lower_case"]
        self.batch_size = exp_config["batch_size"]
        entity_link_path = exp_config["entity_link_path"]
        entity_add_suffix_tag = exp_config["entity_add_suffix_tag"]

        with open(entity_link_path) as f:
            entity_link = json.load(f)
        self.post_processor = NER_post_processor(entity_link, entity_add_suffix_tag)

        task_name = task_name.lower()
        if task_name not in processors:
            raise ValueError("Task not found: %s" % (exp_config["task_name"]))
        processor = processors[task_name]()
        self.label_list = processor.get_labels()
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        num_labels = len(self.label_list)

        # config_class, model_class, tokenizer_class = BertConfig, BertCrfForNer, BertTokenizer
        config_class, model_class, tokenizer_class = BertConfig, BertSpanForNer, BertTokenizer

        config = config_class.from_pretrained(model_name_or_path, num_labels=num_labels)
        config.soft_label = True
        config.loss_type = 'focal'

        self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
                                                         do_lower_case=do_lower_case)
        self.model = model_class.from_pretrained(checkpoint, config=config)
        self.model.to(self.device)
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module

    # all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids, all_lens
    def process_input_data(self, input_sentences):
        examples = []
        for i, sentence in enumerate(input_sentences):
            sentence = sentence.replace(' ', '')
            subject = get_entities(['O'] * len(sentence), id2label=None, markup='bio')
            examples.append(
                InputExample(guid=f'predict_{i}', text_a=[token for token in sentence], subject=subject))

        features = convert_examples_to_features(examples=examples,
                                                label_list=self.label_list,
                                                tokenizer=self.tokenizer,
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
        all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
        all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
        all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids, all_end_ids,
                                all_input_lens)
        return dataset

    def bert_extract_item(self, start_logits, end_logits):
        result = []

        for start_logit, end_logit in zip(start_logits, end_logits):
            S = []

            start_pred = torch.argmax(start_logit, -1).cpu().numpy()[1:-1]
            end_pred = torch.argmax(end_logit, -1).cpu().numpy()[1:-1]

            for i, s_l in enumerate(start_pred):
                if s_l == 0:
                    continue
                for j, e_l in enumerate(end_pred[i:]):
                    if s_l == e_l:
                        S.append((s_l, i, i + j))
                        break

            result.append(S)
        return result

    def inference(self, input_sentences):
        dataset = self.process_input_data(input_sentences)
        predict_sampler = SequentialSampler(dataset)
        predict_dataloader = DataLoader(dataset, sampler=predict_sampler, batch_size=self.batch_size,
                                        collate_fn=collate_fn)
        results = []
        for step, batch in enumerate(predict_dataloader):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                # SPAN
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "start_positions": None,
                          "end_positions": None}
                if self.model_type != "distilbert":
                    # XLM and RoBERTa don"t use segment_ids
                    inputs["token_type_ids"] = (batch[2] if self.model_type in ["bert", "xlnet"] else None)
                outputs = self.model(**inputs)
            start_logits, end_logits = outputs[:2]
            extract_results = self.bert_extract_item(start_logits, end_logits)

            for batch_idx, R in enumerate(extract_results):

                if R:
                    model_entities = [[self.id2label[x[0]], x[1], x[2]] for x in R]
                else:
                    model_entities = []

                json_d = {}
                json_d['id'] = f'step{step}_batchid{batch_idx}'
                json_d['orig_sentence'] = input_sentences[step * self.batch_size + batch_idx].replace(' ', '')
                model_tag_seq = ['O']*len(json_d['orig_sentence'])

                for model_entity in model_entities:
                    entity_name, start_idx, end_idx = model_entity
                    if start_idx >= len(model_tag_seq) or end_idx >= len(model_tag_seq):
                        break
                    model_tag_seq[start_idx] = f'B-{entity_name}'
                    for idx in range(start_idx+1, end_idx+1):
                        model_tag_seq[idx] = f'I-{entity_name}'

                json_d['model_tag_seq'] = " ".join(model_tag_seq)
                json_d['model_entities'] = [model_entity for model_entity in model_entities if not(model_entity[1] >= len(model_tag_seq) or model_entity[2] >= len(model_tag_seq))]
                results.append(json_d)

        processed_result = self.post_processor.post_processing(results)
        # processed_result = results
        return processed_result


if __name__ == "__main__":
    exp_config_path = "./custom.jsonnet"
    model = Model(exp_config_path)
    print(model.inference(['衛服部109年預算的歲出機關計畫預算分支計畫名稱和預算數是什麼？', '列出106年及107年預算是1540149千元的歲出機關預算的科目編號。']))
