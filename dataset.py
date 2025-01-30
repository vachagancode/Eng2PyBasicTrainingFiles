import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict

from pathlib import Path
import pandas as pd
from transformers import RobertaTokenizer


class DSet(Dataset):
	def __init__(self, annotations_file, tokenizer, max_length : int = 200):
		super().__init__()
		self.annotations_file = annotations_file
		self.df = pd.read_json(self.annotations_file)
		self.features = self.df["instruction"]
		self.labels = self.df["output"]

		self.max_length = max_length
		self.tokenizer = tokenizer

	def __len__(self):
		return len(self.features)

	def __getitem__(self, index):
		text = self.features[index]
		code = self.labels[index]

		text_encoding = self.tokenizer(
			text,
			max_length=self.max_length,
			padding='max_length',
			truncation=True,
			return_tensors=None,
			return_attention_mask=True
		)
		
		code_encoding = self.tokenizer(
			code,
			max_length=self.max_length,
			padding='max_length',
			truncation=True,
			return_tensors=None,
			return_attention_mask=True
		)

		decoder_input_ids = [self.tokenizer.bos_token_id]
		decoder_input_ids.extend(code_encoding["input_ids"][:-1])

		if len(decoder_input_ids) < self.max_length:
			decoder_input_ids.extend([self.tokenizer.pad_token_id] * (self.max_length - len(decoder_input_ids)))

		return {
			"input_ids" : torch.tensor(text_encoding["input_ids"], dtype=torch.long),
			"attention_mask" : torch.tensor(text_encoding["attention_mask"], dtype=torch.long),
			"labels" : torch.tensor(code_encoding["input_ids"], dtype=torch.long),
			"decoder_input_ids" : torch.tensor(decoder_input_ids, dtype=torch.long),
			"decoder_attention_mask" : torch.tensor(code_encoding["attention_mask"], dtype=torch.long)
		}


# Load the dataset
def load_dataset(split_paths : Dict[str, str] = {
		'train' : 'data/code_generation_EN.json',
		'test' : 'CoNaLa/raw/valid.csv',
		'validation' : 'CoNaLa/raw/test.csv'
	}):
	# Initialize the tokenizer
	tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")

	# define the special tokens
	special_tokens = {
		"additional_special_tokens" : ["<START>", "<END>", "<PAD>"]
	}
	tokenizer.add_special_tokens(special_tokens)

	train_data = DSet(
		annotations_file=split_paths["train"],
		tokenizer=tokenizer,
		max_length=150
	)

	# test_data = DSet(
	# 	annotations_file=split_paths["test"],
	# 	tokenizer=tokenizer,
	# 	max_length=150
	# )

	# validation_data = DSet(
	# 	annotations_file=split_paths["validation"],
	# 	tokenizer=tokenizer,
	# 	max_length=150
	# )

	return train_data

def create_dataloaders(train_data):
	train_dataloader = DataLoader(
		dataset=train_data,
		batch_size=32,
		shuffle=True
	)

	# if test_data is not None:
	# 	test_dataloader = DataLoader(
	# 		dataset=test_data,
	# 		batch_size=32,
	# 		shuffle=True
	# 	)

	# if test_data is not None:
	# 	valid_dataloader = DataLoader(
	# 		dataset=validation_data,
	# 		batch_size=32,
	# 		shuffle=True
	# 	)

	return train_dataloader

if __name__ == "__main__":
	split_paths = {
		'train' : 'data/code_generation_EN.json',
		'test' : None,
		'validation' : None
	}
	train_data = load_dataset(split_paths=split_paths)
	print(len(train_data))
	train_dataloader = create_dataloaders(
		train_data
	)
	print(next(iter(train_dataloader))["input_ids"])