import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import numpy as np
import re
import spacy
import tqdm
import time
import sys
import csv
from sklearn import preprocessing
from joblib import Parallel, delayed
from tensorflow.keras.utils import to_categorical

def lower_and_split_special_chars(data):
    data = re.split('(\d+\.*\,*\'*\-*\d+|[a-zA-Z]+)',data.lower())
    return " ".join([token.strip() for token in data if not token.isspace() and len(token)>0])

def check_header(file_path):
    expected_header = ['drug_name', 'quantity']
    with open(file_path, 'r') as file:
        header = file.readline().strip().split('\t')
        file.close()
    assert header == expected_header, f"The file header must be '{expected_header}' separated by a tabulation."
        
class Dataset():
    
    def __init__(self, path=None):
        self.path = path
        if path:
            self.get_from_path(path)
            
    def clear_data(self):
        self.text = self.text[self.text.columns.tolist()].apply(
            lambda row: lower_and_split_special_chars(' , '.join(str(val) for val in row.values if pd.notna(val))), axis=1
        )
            
    def get_from_path(self, path):
        check_header(path)
        self.path = path
        self.text = pd.read_csv(self.path, sep='\t', header=0)
        self.clear_data()
        
    def BIO_Tagger(self):
        
        sample = pd.DataFrame([(m.group(0), m.start(), m.end(), 'O', 'S-'+str(idx)) 
                               for idx,sentence in enumerate(self.text) 
                               for m in re.finditer(r'\S+', sentence)], 
                              columns=['Token', 'Start', 'End', 'Tag', 'Sentence'])
        self.BIO_format = sample
        
    def Generate_Input(self, cfg, offsets, n_jobs):
        self.inputs = create_inputs(self.BIO_format, self.tokenizer, cfg, offsets, n_jobs)
        
    
    def CBBTransform(self, tokenizer, cfg, offsets, n_jobs):
        self.tokenizer = tokenizer
        #self.get_from_path(path)
        self.BIO_Tagger()
        self.Generate_Input(cfg, offsets, n_jobs)
        
    
def process_dataframe2(df):
    
    enc_tag = preprocessing.LabelEncoder()
    df.loc[:, "Old_Tag"] = df.Tag
    tags = ['B-DRUG', 'B-STRENGTH', 'I-STRENGTH', 'B-FORM', 'O', 'B-DOSAGE', 'I-DOSAGE', 'I-DRUG', 'B-ROUTE', 'I-FORM', 'I-ROUTE']
    enc_tag.fit(tags)
    df.loc[:, "Tag"] = enc_tag.transform(df["Tag"])
    sentences = df.groupby("Sentence")["Token"].apply(list).values
    sents_name = df.groupby("Sentence")['Sentence'].apply(np.unique).values
    tag = df.groupby("Sentence")["Tag"].apply(list).values
    tag_name = df.groupby("Sentence")["Old_Tag"].apply(list).values
    start = df.groupby("Sentence")["Start"].apply(list).values
    end = df.groupby("Sentence")["End"].apply(list).values
    return sents_name, sentences, tag, tag_name, start, end, enc_tag

def process_dataframe(df):
    
    enc_tag = preprocessing.LabelEncoder()
    df['Index'] = df.Sentence.apply(lambda x: int(x[2:]))
    df.loc[:, "Old_Tag"] = df.Tag
    tags = ['B-DRUG', 'B-STRENGTH', 'I-STRENGTH', 'B-FORM', 'O', 'B-DOSAGE', 'I-DOSAGE', 'I-DRUG', 'B-ROUTE', 'I-FORM', 'I-ROUTE']
    enc_tag.fit(tags)
    df.loc[:, "Tag"] = enc_tag.transform(df["Tag"])
    sentences = df.groupby("Index")["Token"].apply(list).values
    sents_name = df.groupby("Index")['Sentence'].apply(np.unique).values
    tag = df.groupby("Index")["Tag"].apply(list).values
    tag_name = df.groupby("Index")["Old_Tag"].apply(list).values
    start = df.groupby("Index")["Start"].apply(list).values
    end = df.groupby("Index")["End"].apply(list).values
    return sents_name, sentences, tag, tag_name, start, end, enc_tag

    
def create_inputs(dataframe, tokenizer, cfg, offsets=False, n_jobs=-1, from_BRAT=False):
    
    if from_BRAT:
        sents_name, sentences, tags, tag_names, start, end, tag_encoder = process_dataframe2(dataframe)
    else:
        sents_name, sentences, tags, tag_names, start, end, tag_encoder = process_dataframe(dataframe)
    
    def parallel_fn(sent_name, sentence, tag, tag_name, start, end):
        
        input_ids = []
        target_tags = []
        start_offset = []
        end_offset = []
        for idx, word in enumerate(sentence):
            ids = tokenizer.encode(word, add_special_tokens=False)
            input_ids.extend(ids)
            num_tokens = len(ids)
            
            if tag_name[idx].split('-')[0] == 'B':
                name = 'I-' + tag_name[idx][2:]
                target_tags.extend([tag[idx]] + ([tag_encoder.transform([name])[0]] * (num_tokens-1)))
            else:    
                target_tags.extend([tag[idx]] * num_tokens)   
            if offsets:
                start_offset.extend([start[idx]] * num_tokens)
                end_offset.extend([end[idx]] * num_tokens)

        input_ids = [101] + input_ids + [102]
        target_tags = [tag_encoder.transform(['O'])[0]] + target_tags + [tag_encoder.transform(['O'])[0]] #Probar con cls y sep como pad id
        attention_mask = [1] * len(input_ids)
        if offsets:
            start_offset = [-1] + start_offset + [-1]
            end_offset = [-1] + end_offset + [-1]
        padding_len = cfg['max_seq_len'] - len(input_ids)
        
        input_ids = input_ids + ([0] * padding_len)
        attention_mask = attention_mask + ([0] * padding_len)
        target_tags = target_tags + ([cfg['num_tags']] * padding_len)
        if offsets:
            start_offset = start_offset + ([-1] * padding_len)
            end_offset = end_offset + ([-1] * padding_len)

        assert len(target_tags) == cfg['max_seq_len'], f'{len(input_ids)}, {len(target_tags)}'
        
        return sent_name, input_ids, attention_mask, target_tags, padding_len, start_offset, end_offset
    
    data_folds = Parallel(n_jobs=n_jobs)(delayed(parallel_fn)(sent_name, sentence, tag, tag_name, start, end) \
                                         for sent_name, sentence, tag, tag_name, start, end \
                                         in tqdm.tqdm(zip(sents_name, sentences, tags, tag_names, start, end)))
    data_folds = list(zip(*data_folds))
    sent = np.reshape(data_folds[0], newshape=[-1])
    x = [np.array(data_folds[1]), np.array(data_folds[2])]
    y = np.array(data_folds[3])
    pad_len = np.array(data_folds[4])
    offsets = [np.array(data_folds[5]),np.array(data_folds[6])]

    return sent, x, to_categorical(y), tag_encoder, pad_len, offsets
