import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
import re
from Utils.val_script import main as UKB_eval
from Utils.data import create_inputs

label_names = ['DRUG', 'ROUTE', 'STRENGTH', 'FORM', 'DOSAGE']
df_columns = ['TEXT','DRUG','DOSAGE','FORM','ROUTE','STRENGTH','TEXT_drug_name','TEXT_quantity']

"""
CBB
"""

def get_results_cbb(ds, y_pred, n_jobs=-1):
    
    sent, X_test, y_test, tag_encoder, pad_len, offsets = ds.inputs
    untrtext = pd.read_csv(ds.path, sep='\t', header=0)
    
    data, span = list(zip(*Parallel(n_jobs)(
        delayed(ents_dic_format_cbb)(untrtext.iloc[i], ds.text.iloc[i], X_test[0][i], y_pred[i], offsets[0][i], offsets[1][i], 
                                 pad_len[i], label_names, sent[i], ds.tokenizer, tag_encoder) 
        for i in tqdm.tqdm(range(len(y_pred))))))
    pd_UKB_NER = pd.DataFrame(data)
    pd_UKB_NER['SENT'] = list(map(lambda x : int(x.split('-')[1]),sent))
    pd_UKB_NER = pd_UKB_NER.sort_values('SENT')
    result = pd_UKB_NER[df_columns].reset_index(drop=True)
    result.iloc[:,1:-2] = result.iloc[:,1:-2].applymap(lambda x : list(set(x)))
    return result, span


def Model2BRAT(txt, prd, start, end, doc_idx):
    pr = list(zip(txt, prd, start, end))
    pr = sorted(pr, key = lambda x: x[2])
    idx = 1
    while idx < len(pr):
        if pr[idx-1][2]==pr[idx][2] and pr[idx-1][3]==pr[idx][3]:
            pr[idx-1] = (pr[idx-1][0]+pr[idx][0].replace('##',''), *pr[idx-1][1:])
            pr = pr[:idx]+pr[idx+1:]
            idx-=1
        idx+=1
    idx = len(pr)-1
    while idx >= 0:
        label = pr[idx][1].split('-')
        if label[0] == 'I':
            wnd = 1
            while idx-wnd>=0 and pr[idx-wnd][1].split('-')[0]!='B':
                wnd+=1
            if pr[idx-wnd][1].split('-')[0]=='B':
                pr[idx-wnd] = (' '.join([e[0] for e in pr[idx-wnd:idx+1]]), pr[idx-wnd][1], pr[idx-wnd][2], pr[idx][3])
                pr = pr[:idx-wnd+1]+pr[idx+1:]
                idx-=wnd
            else: pr[idx] = (pr[idx][0], 'B-'+pr[idx][1].split('-')[1], *pr[idx][2:])
        idx-=1
    return pr

def get_lists_of_results(X_test, y_pred, start, end, pad_len, tokenizer, tag_encoder):
    txt = np.array([tokenizer.decode([t]) for t in X_test[1:-pad_len-1]])
    prd = tag_encoder.inverse_transform(y_pred.argmax(axis=-1)[1:-pad_len-1])
    start = start[1:-pad_len-1]
    end = end[1:-pad_len-1]
    return txt, prd, start, end

def ents_dic_format_cbb(untrtext, text, X, y_pred, start, end, pad_len, keys, sent, tokenizer, tag_encoder):
    doc_dic = {k:[] for k in keys}
    doc_dic['TEXT'] = text#tokenizer.decode(X[1:-pad_len-1])
    doc_dic['TEXT_drug_name'] = untrtext.drug_name
    doc_dic['TEXT_quantity'] = untrtext.quantity

    lofr = get_lists_of_results(X, y_pred, start, end, pad_len, tokenizer, tag_encoder)
    df = pd.DataFrame(Model2BRAT(*lofr, sent), columns=['Token', 'Tag', 'Start', 'End'])
    df.Tag = df.Tag.apply(lambda x : x[2:])

    values = df[df.Tag!=''].values
    for text, ent in values[:,:-2]:
        doc_dic[ent].append(text)
    return doc_dic, df[df.Tag!='']



"""
MED7
"""

def ents_dic_format_med7(untrtext, text, ents, keys, lemmatizer=None):
    doc_dic = {k:[] for k in keys}
    doc_dic[keys[0]] = text
    doc_dic[keys[-2]] = untrtext.drug_name
    doc_dic[keys[-1]] = untrtext.quantity
    span = []
    for ent in ents:
        if ent.label_ in label_names:
            doc_dic[ent.label_].append(ent.text)
            doc_dic[ent.label_] = remove_duplicates(doc_dic[ent.label_], lemmatizer)
            span.append((ent.text, ent.label_, ent.start_char, ent.end_char))
    return doc_dic, pd.DataFrame(span, columns=["Token","Tag","Start","End"])

def span_dic_format(ents):
    return [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in ents]

def remove_duplicates(values, lemmatizer=None):
    if lemmatizer:
        values = [" ".join([token.lemma_ for token in doc]) for doc in lemmatizer.pipe(values)]
    values = sorted(values, key = len)
    return [j for i, j in enumerate(values) if all(j.replace(" ", "") not in k.replace(" ", "") for k in values[i + 1:])]

def get_results_med7(path, docs):
    untrtext = pd.read_csv(path, sep='\t', header=0)
    result, span = list(zip(*[ents_dic_format_med7(untrtext.iloc[idx], d.text, d.ents, df_columns) for idx,d in enumerate(docs)]))
    return pd.DataFrame(result), span


"""
MERGE
"""

def span_merge(span1, span2):
    span = span2.merge(span1, how='left', indicator=True)
    span = span[span._merge == 'left_only'].iloc[:,:-1]
    return pd.concat([span1, span], ignore_index=True)

def result_merge(result1, result2):
    result = result1.copy()
    result.iloc[:,1:-2] = result.iloc[:,1:-2]+result2.iloc[:,1:-2]
    result.iloc[:,1:-2] = result.iloc[:,1:-2].applymap(lambda x : list(set(x)))
    return result

def merge(results, spans):
    return result_merge(*results), [span_merge(*list(zip(*spans))[idx]) for idx in range(len(spans[0]))]

def merge_parallel(results, spans, n_jobs):
    return result_merge(*results), Parallel(n_jobs=n_jobs)(delayed(span_merge)(*s) for s in tqdm.tqdm(list(zip(*spans))))
    
    
"""
METRIC EVALUATION
"""   

class BRAT2BIO():
    
    def __init__(self, label_names, data_path, sentence_max_len, sentence_separator=None, files_list=None):
        self.label_names = label_names
        self.data_path = data_path
        self.sentence_tagger_path = 'en_core_web_sm'
        self.sentence_max_len = sentence_max_len
        self.sentence_separator = sentence_separator
        self.files_list = files_list

    def get_file_list(self):
        return [file.split('.')[0] for file in os.listdir(self.data_path)if file.endswith(".ann")]

    def format_offset_col(self, row):
        tokens = re.split(';|\s+',row)
        tokens = [tokens[0]]+list(map(int, tokens[1:]))
        if tokens[0] in self.label_names:
            tokens[0] = 'B-'+tokens[0]
            if len(tokens)>3:
                start = [tokens[i] for i in range(1,len(tokens)-1,2)]
                end = [tokens[i] for i in range(2,len(tokens),2)]
                if end[0]<start[1]-1:
                    #print("There're list objects")
                    tokens[1], tokens[2] = start[0], end[0]
                else: 
                    tokens[1], tokens[2] = start[0], end[1]
            return tokens[:3]
        return None

    def load_ann_data(self, path):
        df = pd.read_csv(path+".ann", sep='\t', names=['Task','Entity','Text'], header=None, quoting=csv.QUOTE_NONE)
        df = df[df.Task.str.contains(r'^T+')]
        df.Entity = df.Entity.apply(lambda x: self.format_offset_col(x))
        df = df[df.Entity.notnull()]
        if len(df)==0: return pd.DataFrame(columns=['Text','Tag','Start','End'])
        df[['Tag','Start','End']] = pd.DataFrame(df.Entity.tolist(), index=df.index)
        return df[['Text','Tag','Start','End']].sort_values('Start').reset_index(drop=True)

    def load_text_data(self, path):
        f = open(path+".txt", "rt")
        data = f.read()#.lower()
        sample = pd.DataFrame([(m.group(0), m.start(), m.end()) for m in re.finditer(r'\S+', data)], columns=['Token', 'Start', 'End'])
        return data, sample

    def load_data(self, path):
        self.document_name = path.split('/')[-1]
        return *self.load_text_data(path), self.load_ann_data(path)

    def get_matrix_token_ann(self, df_text, df_ann):
        matrix = df_text.Start.apply(lambda x : x<=df_ann.End)*df_text.End.apply(lambda x : x>=df_ann.Start)
        df_matrix = pd.DataFrame([list(df_ann[matrix.iloc[i]][['Tag','Start','End']].values[0]) if any(matrix.iloc[i]) else ['O', 0, 0] 
                                  for i in df_text.index], columns=['Tag','S_pos','E_pos'])
        return df_matrix

    def clear_r_margin(self, df):
        prb = df[(df.Tag!='O')*(df.End-1 == df.E_pos)]
        for i,idx in enumerate(prb.index):
            elem = list(df.iloc[idx+i])
            prev, next = elem.copy(),elem.copy()
            prev[0], prev[2] =elem[0][:-1], elem[-2]
            next[0], next[1], next[-4:] = elem[0][-1], elem[2]-1, ['O', 0, 0, elem[-1]]
            df.iloc[idx+i] = prev
            df = pd.concat([df.iloc[:idx+i+1],
                            pd.DataFrame([next],columns=df.columns),
                            df.iloc[idx+i+1:]]).reset_index(drop=True)
        return df

    def clear_l_margin(self, df):
        prb = df[(df.Tag!='O')*(df.Start+1 == df.S_pos)]
        for i,idx in enumerate(prb.index):
            elem = list(df.iloc[idx+i])
            prev, next = elem.copy(),elem.copy()
            prev[0], prev[2], prev[-4:] = elem[0][0], elem[1]+1, ['O', 0, 0, elem[-1]]
            next[0], next[1] = elem[0][1:], elem[1]+1
            df.iloc[idx+i] = prev
            df = pd.concat([df.iloc[:idx+i+1],
                            pd.DataFrame([next],columns=df.columns),
                            df.iloc[idx+i+1:]]).reset_index(drop=True)
        return df

    def clear_margins(self, df):
        return self.clear_r_margin(self.clear_l_margin(df))

    def BIO_extends(self, df):
        for idx in range(len(df)-1):
            cols = ['S_pos','E_pos']
            if all(df[cols].iloc[idx] == df[cols].iloc[idx+1]) and (df.Tag.iloc[idx]!='O'):
                df.loc[idx+1, 'Tag'] = 'I-'+df.Tag.iloc[idx][2:]
        return df
    
    def sentence_tagging_nltk(self, raw, df):
        sents = nltk.sent_tokenize(raw)
        len_per_sentence = [len(list(re.finditer(r'\S+',sent))) for sent in sents]
        #return len_per_sentence
        start_idx = 0
        for pos in range(len(len_per_sentence)):
            end_idx = start_idx+len_per_sentence[pos]
            df.loc[start_idx:end_idx,'Sentence'] = "S-"+str(self.document_name)+"-"+str(pos)
            start_idx+=len_per_sentence[pos]
        return df

    def sentence_tagging(self, raw, df):
        #spacy.prefer_gpu()
        def repl(m):
            return 'A'* len(m.group())
        raw = re.sub(r'\[\*\*(.*?)\*\*\]',repl, raw)
        
        nlp = spacy.load(self.sentence_tagger_path)
        doc = nlp(raw)
        len_per_sentence = [sent.end_char for sent in doc.sents if len(sent.text.strip())>0]
        start_idx = 0
        for pos in range(len(len_per_sentence)):
            end_idx = df.index[df.Start>=len_per_sentence[pos]]
            if len(end_idx)>1: end_idx = end_idx[0]
            else: end_idx = len(df)-1
            df.loc[start_idx:end_idx,'Sentence'] = "S-"+str(self.document_name)+"-"+str(pos)
            start_idx = end_idx
        return df
    
    def sentence_tagging_by_char(self, raw, df):
        sents = raw.split(self.sentence_separator)
        len_per_sentence = [len(sent)+1 for sent in sents]
        start_idx = 0
        for pos in range(len(len_per_sentence)):
            end_idx = df.index[df.Start>=sum(len_per_sentence[:pos+1])]
            if len(end_idx)>1: end_idx = end_idx[0]
            else: end_idx = len(df)-1
            df.loc[start_idx:end_idx,'Sentence'] = "S-"+str(self.document_name)+"-"+str(pos)
            start_idx = end_idx
        return df
    
    def split_sentences_max_len(self, df):
        lg_sentences = df.groupby('Sentence').Token.count()
        sents_id = lg_sentences[lg_sentences>self.sentence_max_len].index
        for sent_id in sents_id:
            sub_id = 1
            sentence_len = len(df[df.Sentence == sent_id])
            while self.sentence_max_len < sentence_len:
                df.loc[df[df.Sentence == sent_id][self.sentence_max_len:self.sentence_max_len*2].index, 'Sentence'] = sent_id+'-'+str(sub_id)
                sentence_len = len(df[df.Sentence == sent_id])
                sub_id+=1
        return df
    
    def get_BIO_dataframe(self, raw, text, ann):
        df = pd.concat([text,self.get_matrix_token_ann(text, ann)], axis=1)
        if self.sentence_separator: fn_sep = self.sentence_tagging_by_char
        else: fn_sep = self.sentence_tagging
        return self.split_sentences_max_len(self.BIO_extends(self.clear_margins(fn_sep(raw, df))))

    def get_all_BIO_texts(self):
        if self.files_list is None:
            self.files_list = self.get_file_list()
        return [self.get_BIO_dataframe(*self.load_data(self.data_path+file)) for file in tqdm.tqdm(self.files_list, leave=False)]
    
def BRAT_dict2df(dictObj):
    df_BRAT = pd.DataFrame(dictObj, columns=['TEXT','TAG','START','END'])
    df_BRAT = df_BRAT[df_BRAT.TAG!='O']
    df_BRAT.TAG = df_BRAT.TAG.apply(lambda x : x[2:])
    df_BRAT['ENT']=df_BRAT.TAG+" "+list(df_BRAT.START.apply(str))+" "+list(df_BRAT.END.apply(str))
    df_BRAT = df_BRAT.reset_index(drop=True)
    df_BRAT['TERM'] = list(df_BRAT.index +1)
    df_BRAT['TERM'] = df_BRAT['TERM'].apply(lambda x : 'T'+str(x))
    return df_BRAT[['TERM','ENT','TEXT']]
    
def get_set_files(path, test_split):
    all_files = np.array([file.split('.')[0] for file in os.listdir(path) if file.endswith(".ann")])
    return all_files[:-int(len(all_files)*test_split)], all_files[-int(len(all_files)*test_split):]

def get_df_from_files(label_names, data_path, files):
    tagger = BRAT2BIO(label_names, data_path, sentence_max_len=500, sentence_separator='\n', files_list=files)
    df = tagger.get_all_BIO_texts()
    return pd.concat(df).reset_index()


def generate_ann_files(test, df_test, offsets, pad_len, model, tokenizer, tag_encoder, ann_files_save_path, y_pred=None):
    if y_pred is None:
        y_pred = model.predict(test)
    
    df_test['Document'] = df_test.Sentence.apply(lambda x : x.split('-')[1])
    document = df_test.groupby("Sentence")['Document'].apply(lambda x: list(x)[0])
    dictObj = { key:[] for key in document.unique()}
    for idx in tqdm.tqdm(range(len(y_pred)), leave=False):
        dictObj[document.iloc[idx]] = dictObj[document.iloc[idx]] + list(zip(*get_lists_of_results(
            test[0][idx], y_pred[idx], offsets[0][idx], offsets[1][idx], pad_len[idx], tokenizer, tag_encoder)))
    for doc in tqdm.tqdm(dictObj, leave=False):
        dictObj[doc] = Model2BRAT(*list(map(list, zip(*dictObj[doc]))), doc) 
    df_BRAT = {doc: BRAT_dict2df(dictObj[doc]) for doc in tqdm.tqdm(dictObj, leave=False)}
    for file in tqdm.tqdm(df_BRAT, leave=False):
        df_BRAT[file].to_csv(ann_files_save_path+'/'+file+'.ann', sep='\t', columns=None, header=False, index=False)
        
    
def UKB_mean_eval(results, folds):
    results_mean = {}
    def mean_std(values):
        if isinstance(values[0], float):
            return np.mean(values), np.std(values)
        micro = [v['micro'] for v in values]
        macro = [v['macro'] for v in values]
        return {'micro':(np.mean(micro), np.std(micro)), 'macro':(np.mean(macro), np.std(macro))}
    for mode in ['strict', 'lenient']:
        results_mean[mode] = {}
        for tag in results[0]:
            if tag!='Overall':
                results_mean[mode][tag] = {metric:mean_std([results[i][tag][mode][metric] for i in range(folds)]) 
                                     for metric in ['precision', 'recall', 'f1']}
            else:
                results_mean[mode][tag] = {metric:mean_std([{measure: 
                                      results[i][tag][mode][measure][metric]
                                      for measure in ['micro', 'macro']} for i in range(folds)]) 
                                     for metric in ['precision', 'recall', 'f1']}
    return results_mean
    
def get_metrics_mean_results(results, verbose=True):
    results_mean = UKB_mean_eval(results, 1)
    if verbose:
        print('=' * 100)
        print('micro F1 (strict)(mean, std): ',{metric:results_mean['strict']['Overall'][metric]['micro'] 
                                                for metric in ['precision', 'recall', 'f1']})
        print('micro F1 (lenient)(mean, std): ',{metric:results_mean['lenient']['Overall'][metric]['micro'] 
                                                 for metric in ['precision', 'recall', 'f1']})
        print('macro F1 (strict)(mean, std): ',{metric:results_mean['strict']['Overall'][metric]['macro'] 
                                                for metric in ['precision', 'recall', 'f1']})
        print('macro F1 (lenient)(mean, std): ',{metric:results_mean['lenient']['Overall'][metric]['macro'] 
                                                 for metric in ['precision', 'recall', 'f1']})
    return results_mean

def get_metrics_result_from_BRAT(data_path, result_path, ner, verbose=False, split=0, n_jobs=-1):
    train_files, test_files = get_set_files(data_path, split)
    df_test = get_df_from_files(label_names, data_path, test_files)
    
    sent, X_test, y_test, tag_encoder, pad_len, offsets = create_inputs(df_test, ner.tokenizer, ner.cfg, offsets=True, n_jobs=n_jobs, from_BRAT=True)
    generate_ann_files(X_test, df_test, offsets, pad_len, 
                   ner.model, ner.tokenizer, tag_encoder, 
                   ann_files_save_path=result_path)

    mean_results = get_metrics_mean_results([UKB_eval(data_path,result_path,2,False)], False)
    strict_metrics_result=pd.concat([
        pd.DataFrame(mean_results['strict']).swapaxes("index", "columns").iloc[:-1].applymap(lambda x: ["{:.3f}".format(e) for e in x]),
        pd.DataFrame(mean_results['strict']['Overall']).applymap(lambda x: ["{:.3f}".format(e) for e in x])
    ])
    lenient_metrics_result=pd.concat([
        pd.DataFrame(mean_results['lenient']).swapaxes("index", "columns").iloc[:-1].applymap(lambda x: ["{:.3f}".format(e) for e in x]),
        pd.DataFrame(mean_results['lenient']['Overall']).applymap(lambda x: ["{:.3f}".format(e) for e in x])
    ])
    if verbose:
        print('-------Strict-------')
        s = strict_metrics_result.to_string().replace('\nmicro', '\n'+'_'*60+'\nmicro')
        print(s)
        print('-------Lenient-------')
        s = lenient_metrics_result.to_string().replace('\nmicro', '\n'+'_'*60+'\nmicro')
        print(s)
    return strict_metrics_result, lenient_metrics_result

