import json
import numpy as np

import torch
from torch.utils.data import Dataset


class LLMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = json.loads(self.data[idx])
        text = "<s>" + line["text"] + "</s>"
        inp_ids = self.tokenizer.encode(text)
        text_len = len(inp_ids)

        if text_len > self.max_seq_len:
            inp_ids = inp_ids[: self.max_seq_len]
        else:
            inp_ids += [0] * (self.max_seq_len - text_len)

        inp_ids = np.array(inp_ids)
        x = np.array(inp_ids[:-1]).astype(np.int64)
        y = np.array(inp_ids[1:]).astype(np.int64)

        return {
            "inp_ids": torch.from_numpy(x),
            "labels": torch.from_numpy(y),
        }


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = json.loads(self.data[idx])
        inst_text = line["instruction"]
        inp_text = line["input"]
        out_text = line["output"]
        hist = line["history"]
        query = inst_text + " " + inp_text
        answer = out_text + self.tokenizer.eos_token

        msg = []
        if hist:
            for h in hist:
                msg.append({"role": "user", "content": h[0]})
                msg.append({"role": "assistant", "content": h[1]})

        msg.append({"role": "user", "content": query})

        prompt = self.tokenizer.apply_chat_template(msg, tokenize=False)
        prompt_inp_ids = self.tokenizer.encode(prompt)
        answer_inp_ids = self.tokenizer.encode(answer)

        inp_ids = prompt_inp_ids + answer_inp_ids
        labels = [0] * len(prompt_inp_ids) + answer_inp_ids

        text_len = len(inp_ids)
        if text_len > self.max_seq_len:
            inp_ids = inp_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]
        else:
            inp_ids += [0] * (self.max_seq_len - text_len)
            labels = labels + [0] * (self.max_seq_len - text_len)

        inp_ids = inp_ids[:-1]
        labels = labels[1:]
        return {
            "inp_ids": torch.tensor(inp_ids),
            "labels": torch.tensor(labels),
        }


# class DPODataset(Dataset):
#     def __init__(self, data_path, tokenizer):
#         super().__init__()
#         self.data_path = data_path
#         self.tokenizer = tokenizer
#         self.eos_id = tokenizer.eos_token_id

#         with open(data_path, "r", encoding="utf-8") as f:
#             self.data = json.load(f)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         prompt = sample["prompt"]
#         chosen = sample["chosen"]
#         rejected = sample["rejected"]
#         msg = [{"role": "user", "content": prompt}]

#         text = self.tokenizer.apply_chat_template(
#             msg,
#             tokenize=False,
#             add_generation_prompt=True,
#         )

#         prompt_inp = self.tokenizer(text)["input_ids"]
#         rejected_inp = self.tokenizer(rejected)["input_ids"] + [self.eos_id]
#         chosen_inp = self.tokenizer(chosen)["input_ids"] + [self.eos_id]

#         return [prompt_inp, chosen_inp, rejected_inp]


# class DPODataCollator:
#     def __init__(self, tokenizer, max_seq_len):
#         self.tokenizer = tokenizer
#         self.max_seq_len = max_seq_len

#     def process(self, inp_ids, labels):
#         inp_ids = [inp_id[: self.max_seq_len] for inp_id in inp_ids]
#         labels = [label[: self.max_seq_len] for label in labels]
#         max_len = max(len(inp_id) for inp_id in inp_ids)
#         batch_inp_ids = []
#         batch_labels = []

#         for inp_id, label in zip(inp_ids, labels):
#             if len(inp_id) <= max_len:
#                 inp_id = inp_id + [0] * (max_len - len(inp_id))
#                 label = label + [0] * (max_len - len(label))
#                 batch_inp_ids.append(inp_id[:-1])
#                 batch_labels.append(label[1:])

#         return batch_inp_ids, batch_labels

#     def __call__(self, features):
#         inp_ids = []
#         labels = []

#         for fea in features:
#             inp_ids.append(fea[0] + fea[1])
#             labels.append([0] * len(fea[0]) + fea[1])

#         for fea in features:
#             inp_ids.append(fea[0] + fea[2])
#             labels.append([0] * len(fea[0]) + fea[2])

#         inp_ids, labels = self.process(inp_ids, labels)

#         return {
#             "inp_ids": torch.tensor(inp_ids),
#             "labels": torch.tensor(labels),
#         }
