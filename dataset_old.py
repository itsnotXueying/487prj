from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import torch
import random
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
import string
import gensim.downloader
import isodate

punks = string.punctuation
punks = punks+ '``'+ "''"
stopword_list = stopwords.words('english')

np.random.seed(88)
torch.manual_seed(88)
random.seed(88)

embed = gensim.downloader.load("glove-wiki-gigaword-200")

def get_train_val_test_loaders(batch_size):
    tr = Source("train")
    va = Source("val")
    te = Source("test")


    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=2)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=2)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=2)

    return tr_loader,va_loader, te_loader

class Source(Dataset):
    def __init__(self, task):
        self.base_folder = "splits"
        self.task = task

        self._load_metadata()

    def _load_metadata(self):
        self.data = pd.read_csv('useRecipesData.csv')
        #grab only rows we care about for a train/val/test split
        self.data = self.data.loc[self.data['split'] == self.task]
        self.data.reset_index(inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        time = row.PrepTime
        instructions = row.RecipeInstructions
        review_embedding = np.zeros((200,))
        clean_tokens = []
        sent_tokens = sent_tokenize(instructions)
        for sent in sent_tokens:
            tokenized = word_tokenize(instructions)
            for token in tokenized:
                if token not in punks and token not in stopword_list:
                    clean_tokens.append(token.lower())
        for token in clean_tokens:
            if token not in embed:
                continue
            review_embedding += embed[token]
        
        review_embedding = torch.tensor(review_embedding, dtype=torch.float32)
        
        return review_embedding, time