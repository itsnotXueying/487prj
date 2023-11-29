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
import sys

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


    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=1)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=1)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=1)

    return tr_loader,va_loader, te_loader

class Source(Dataset):
    def __init__(self, task):
        self.base_folder = "splits"
        self.task = task

        self._load_metadata()

    def _load_metadata(self):
        self.data = pd.read_csv('useReviews.csv')
        #grab only rows we care about for a train/val/test split
        self.data = self.data.loc[self.data['split'] == self.task]
        self.data.reset_index(inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        y = row[['1','2','3','4','5']].astype(float).values
        y = y.astype(np.float32)
        y = torch.tensor(y)
        review = row.Review
        review_embedding = np.zeros((200,))
        clean_tokens = []
        if type(review) != str:
            print(review)
            print(row['index'])
            sys.exit()
        sent_tokens = sent_tokenize(review)
        for sent in sent_tokens:
            words = word_tokenize(sent)
            for w in words:
                if w not in punks and w not in stopword_list:
                    clean_tokens.append(w.lower())
        for token in clean_tokens:
            if token not in embed:
                continue
            review_embedding += embed[token]
        
        review_embedding = torch.tensor(review_embedding, dtype=torch.float32)

        
        return review_embedding, y
