#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __author__=u"Frank Jing"

import torch
from transformers import *

model = BertModel.from_pretrained('../input/bert-pytorch-model/bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# input_ids:D_bat*D_sen*D_hid
# tensor([[  101,  7592,  1010,  2026,  3899,  2003, 10140,  1012,   102,  2748, 2009,  2003,  1012,   102]])
input_ids = torch.tensor(tokenizer.encode(u"[CLS] Hello, my dog is cute. [SEP] yes it is. [SEP]")).unsqueeze(0)  # Batch size 1
# tensor([[True, True, True, True, True, True, True, True, True, True, True, True, True, True]])
attention_mask = input_ids > 0
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
token_type_id = [0 if i <= input_ids[0].tolist().index(102) else 1 for i in range(input_ids.shape[1])]
token_type_ids = torch.tensor(token_type_id).unsqueeze(0)
# last_hid:D_bat*D_sen*D_hid
# pooled:D_bat*D_hid
last_hid, pooled, *_ = model(input_ids, attention_mask, token_type_ids)  # The last hidden-state is the first element of the output tuple
