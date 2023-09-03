import pickle
import pandas as pd
import numpy as np
import json
import spacy
import urllib.request, json , math
from ast import literal_eval
import itertools
import os, inspect
import re
import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
pd.set_option('display.max_colwidth', None)
df_columns = ['TEXT','DRUG','DOSAGE','FORM','ROUTE','STRENGTH','TEXT_drug_name','TEXT_quantity']

def get_chembl_json_from_url(offset=0, count_elements=False):
    path = "https://www.ebi.ac.uk"
    link = path+"/chembl/api/data/molecule.json?limit=1000&offset="+str(offset)+"&max_phase=4"
    with urllib.request.urlopen(link) as url:
        if count_elements:
            return json.loads(url.read().decode())['page_meta']['total_count']
        return json.loads(url.read().decode())['molecules']

def get_atc_json_from_url(offset=0, count_elements=False):
    path = "https://www.ebi.ac.uk"
    link = path+"/chembl/api/data/atc_class.json?limit=1000&offset="+str(offset)
    with urllib.request.urlopen(link) as url:
        if count_elements:
            return json.loads(url.read().decode())['page_meta']['total_count']
        return json.loads(url.read().decode())['atc']

def extract_drugs(atc=False):
    if atc: extract_fn = get_atc_json_from_url 
    else: extract_fn = get_chembl_json_from_url
    total_elements = extract_fn(count_elements=True)
    return sum([extract_fn(offset=i) for i in range(0, total_elements, 1000)], [])

def pref_names_dictionary(list_of_chembl_pages):
    dictionary = {k:[] for k in ['term', 'chemblid']}
    for drug in list_of_chembl_pages:
        dictionary['term'].append(drug['pref_name'].lower())
        dictionary['chemblid'].append(drug['molecule_chembl_id'])
    return pd.DataFrame(dictionary)

def atc_codes_from_chembl(list_of_chembl_pages):
    atc_codes_dict = [({'atc_code':atc}|{'chemblid':drug['molecule_chembl_id']}) for drug in list_of_chembl_pages for atc in drug['atc_classifications']]
    atc_codes_dict.append({'atc_code':'J070000', 'chemblid':'CHEMBLVAC'})
    return pd.DataFrame(atc_codes_dict)

def pref_atc_dictionary(list_of_atc_pages):
    def set_vaccines_atc_code(dataframe):
        pd_vacciones = pd.DataFrame([["J","ANTIINFECTIVES FOR SYSTEMIC USE","J07","VACCINES_added","J070","VACCINES_added","J0700","VACCINES_added","J070000","vaccine_added_to_dict"]],columns=dataframe.columns)
        pd_atc = pd.concat([dataframe, pd_vacciones], ignore_index=True)
        pd_atc['atc_code'] = pd_atc['level5']
        return pd_atc
    return set_vaccines_atc_code(pd.DataFrame(list_of_atc_pages))

def synonyms_dictionary(list_of_chembl_pages):
    def get_synonyms(elem):
        if len(elem["molecule_synonyms"])>0:
            id = elem["molecule_chembl_id"]
            return pd.DataFrame(elem["molecule_synonyms"]).assign(chemblid=lambda x: id)
    def clean_syns(syns_list):
        df = pd.DataFrame()
        df = syns_list.assign(total=syns_list.groupby('syn_type')['synonyms'].transform('size')).sort_values(by='total',ascending=False)
        df = df.assign(term=df.molecule_synonym.apply(lambda x: x.lower()))
        df = df.assign(occurrences=df.groupby(['term','chemblid']).cumcount()+1)
        df = df[(df.occurrences == 1) & (df.term.apply(lambda x : len(x) > 3))]
        df = df.loc[:,['term', 'syn_type', 'chemblid']]
        return {syn_type : df[df.syn_type==syn_type].loc[:, ['term', 'chemblid']].reset_index(drop=True) for syn_type in pd.unique(df.syn_type)}
    return clean_syns(pd.concat([get_synonyms(elem) for elem in list_of_chembl_pages], ignore_index=True))

def drop_duplicates(list_of_synonym_dicts, dictonary_pref_names):
    return {dict:list_of_synonym_dicts[dict][list_of_synonym_dicts[dict].term.apply(lambda x : x not in list(dictonary_pref_names.term))].reset_index(drop=True) for dict in list_of_synonym_dicts}
     
# def extract_chembl_parents(list_of_chembl_pages):
#     pd_chembl_parents = pd.DataFrame([{'pref_name':drug['pref_name'].lower()} | drug['molecule_hierarchy'] for drug in list_of_chembl_pages if drug['molecule_hierarchy'] and len(drug['molecule_hierarchy']) == 2])
#     pd_chembl_parents.columns = ["pref_name_child","child_chemblid", "parent_chemblid"]
#     return pd_chembl_parents

def extract_chembl_parents(list_of_chembl_pages):
    pd_chembl_parents = pd.DataFrame([{'pref_name':drug['pref_name'].lower()} | drug['molecule_hierarchy'] for drug in list_of_chembl_pages if drug['molecule_hierarchy'] and len(drug['molecule_hierarchy']) >= 2])
    pd_chembl_parents = pd_chembl_parents.rename(columns={"pref_name":"pref_name_child","molecule_chembl_id":"child_chemblid", "parent_chembl_id":"parent_chemblid"})
    return pd_chembl_parents

def extract_parent_atcs(parents_chembl, pd_chembl, atc_dictionary):
    pd_chembl.columns = ["pref_name_parent", "parent_chemblid"]
    parents_chembl = parents_chembl.merge(pd_chembl, on='parent_chemblid')
    atc_dictionary.columns = ["atc_code", "parent_chemblid"]
    atc_to_parents = parents_chembl.merge(atc_dictionary, on='parent_chemblid')
    atc_dictionary.columns = ["atc_code", "child_chemblid"]
    atc_to_child = parents_chembl.merge(atc_dictionary, on='child_chemblid')
    atc_to_parents = atc_to_parents[["atc_code", "parent_chemblid"]]
    atc_to_parents.columns = ["atc_code", "chemblid"]
    atc_to_child = atc_to_child[["atc_code", "child_chemblid"]]
    atc_to_child.columns = ["atc_code", "chemblid"]
    atc_dictionary.columns = ["atc_code", "chemblid"]
    pd_chembl.columns = ["term", "chemblid"]
    return pd.concat([atc_to_parents,atc_to_child,atc_dictionary]).drop_duplicates()

def chembl_pcss(chembl_json, atc_json):
    """
    returns chemble drugs dictionary and atc dict
    """
    pd_chembl = pref_names_dictionary(chembl_json)
    atc_dictionary = atc_codes_from_chembl(chembl_json)
    pd_atc = pref_atc_dictionary(atc_json)
    parents_chembl = extract_chembl_parents(chembl_json)
    new_atc_dict = extract_parent_atcs(parents_chembl, pd_chembl, atc_dictionary)
    syns_list = synonyms_dictionary(chembl_json)
    drugs_list = drop_duplicates(syns_list, pd_chembl)
    drugs_list['PREF_NAME'] = pd_chembl
    drugs = pd.DataFrame()
    for dataframe in drugs_list:
        drugs_list[dataframe]['dict_type'] = dataframe
        drugs = pd.concat([drugs,drugs_list[dataframe]])    
    vaccine = pd.DataFrame([{'term':term, 'chemblid':'CHEMBLVAC', 'dict_type':'VACCINE'} for term in ['vaccine','vaccines', 'vacc']])
    drugs_w_vacc = pd.concat([drugs,pd.DataFrame(vaccine)], ignore_index=True)
    output = drugs_w_vacc.merge(new_atc_dict, on='chemblid')
    
    return output.reset_index(drop=True), new_atc_dict

def load_ehr(path):
    return pd.read_table(path, encoding = "ISO-8859-1")

def load_and_process_ehr(path, column='drug_name'):
    return pd.read_table(path, encoding = "ISO-8859-1")[column].str.lower() 

def load_ehr(path):
    return pd.read_table(path, encoding = "ISO-8859-1")

def load_and_process_ehr(path, column='drug_name'):
    return pd.read_table(path, encoding = "ISO-8859-1")[column].str.lower()

def create_matcher(chembl_drugs_dict):
    dict_names = chembl_drugs_dict['dict_type'].unique()
    nlp_blank = spacy.blank('en')
    matcher = spacy.matcher.PhraseMatcher(nlp_blank.vocab)
    for dict_type in dict_names:
        terms = [nlp_blank(ent) for ent in list(chembl_drugs_dict[chembl_drugs_dict.dict_type==dict_type].term)]
        matcher.add(dict_type, None, *terms)
    return matcher, nlp_blank

def extract_drug_names2(untrtext, data, matcher, nlp_blanck):    
    def extract_drug_names_from_text(text):
        doc = nlp_blanck(text)
        entities = matcher(doc)
        return list(map(list,zip(*[[doc[ent[1]:ent[2]].text,doc[ent[1]:ent[2]].start_char,doc[ent[1]:ent[2]].end_char] 
                                   for ent in entities])))
    def return_df(dp):
        if len(dp)>0: dp = [dp]
        span = pd.DataFrame(dp, columns=['Token','Start','End']).apply(pd.Series.explode).reset_index(drop=True)
        span.insert(1,'Tag','DRUG', True)
        return span
    
    ents_dict = [{df_columns[0]:dp,
                  df_columns[1]:extract_drug_names_from_text(dp),
                  df_columns[-2]: untrtext.iloc[idx].drug_name,
                  df_columns[-1]: untrtext.iloc[idx].quantity} 
                 for idx,dp in enumerate(data)]
    
    df = pd.DataFrame(ents_dict)
    span = [return_df(dp) for dp in df.DRUG]
    df.DRUG = df.DRUG.apply(lambda x: x[0] if len(x) > 0 else x)
    df[[df_columns[2:6]]]=None
    df.iloc[:,-4:] = df.iloc[:,-4:].applymap(lambda x : [])
    
    return df[df_columns], span

def extract_drug_names(untrtext, data, matcher, nlp_blanck, offsets=False, comm=False):
    def extract_drug_names_from_text(text, offsets=False):
        doc = nlp_blanck(text)
        entities = matcher(doc)
        if offsets:
            return [[doc[ent[1]:ent[2]].text,doc[ent[1]:ent[2]].start_char,doc[ent[1]:ent[2]].end_char] for ent in entities]
        return [doc[ent[1]:ent[2]].text for ent in entities]
    
    def span_generator(text, offsets=True):
        span = pd.DataFrame(extract_drug_names_from_text(text, offsets), columns=['Token','Start','End'])
        span.insert(1,'Tag','DRUG', True)
        return span
    
    ents_dict = [{df_columns[0]:dp,df_columns[1]:extract_drug_names_from_text(dp, offsets),
                 df_columns[-2]: untrtext.iloc[idx].drug_name,
                 df_columns[-1]: untrtext.iloc[idx].quantity} for idx,dp in enumerate(data)]
    df = pd.DataFrame(ents_dict)
    if offsets:
        return df
    if comm: 
        span = [span_generator(dp, offsets=True) for dp in data]    
        df[[df_columns[2:6]]]=None
        df.iloc[:,-4:] = df.iloc[:,-4:].applymap(lambda x : [])
        
        return df[df_columns], span
    return df

def get_matched_names(dataframe):
    indxs = dataframe.DRUG.apply(lambda x : len(x)>0)
    return dataframe[indxs], dataframe[~indxs]

def get_results_chembl(path, data, offsets=False, comm=True):
    untrtext = pd.read_csv(path, sep='\t', header=0)
    caller_path = os.path.dirname(os.path.abspath((inspect.stack()[1])[1]))
    try:
        print("Getting ChEMBL data online.")
        chembl_drugs, _ = chembl_pcss(extract_drugs(), extract_drugs(atc=True))
        manual = pd.read_csv(caller_path+'/Resources/manual_dict_atc_codes.txt', sep='\t')
        chembl_drugs = pd.concat([chembl_drugs,manual], axis=0).reset_index(drop=True)
        #chembl_drugs.to_json(caller_path+'/Resources/chembl_atc_results.json')
    except:
        print("Failed to get ChEMBL data online. Getting ChEMBL data locally.")
        chembl_drugs = pd.read_json(caller_path+'/Resources/chembl_atc_results.json')    
    
    matcher, nlp_blanck = create_matcher(chembl_drugs)
    result, span = extract_drug_names2(untrtext, data, matcher, nlp_blanck)
    return result, span
