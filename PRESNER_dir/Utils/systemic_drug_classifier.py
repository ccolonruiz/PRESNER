import pandas as pd
import numpy as np
import os, inspect
pd.options.mode.chained_assignment = None

def load_manually_created_resources(path):
    files = ['systemic_route_terms.txt', 'non_systemic_route_terms.txt', 'systemic_form_terms.txt', 
             'non_systemic_form_terms.txt', 'nonsys_atc_codes.txt']
    return [set(pd.read_csv(path + file_name, sep='\t').iloc[:,0]) for file_name in files]

def load_and_process_chembl_results(chembl_result: list, dicts_path: str, combinations_path: str) -> pd.DataFrame:
    chembl_drugs = pd.read_json(dicts_path)
    dict_data = pd.concat({name: chembl_drugs[chembl_drugs['dict_type'] == name] for name in chembl_drugs['dict_type'].unique()})
    chembl_result = pd.concat(chembl_result)[['TEXT','DRUG']]
    
    chembl_result['fk'] = chembl_result.index
    chembl_result = chembl_result[chembl_result['DRUG'].apply(lambda x:len(x)>0)].explode('DRUG', ignore_index=True)
    
    chembl_result = pd.merge(chembl_result, dict_data, left_on='DRUG', right_on='term', how='inner').drop(['DRUG'], axis=1)
    
    combinations = pd.read_csv(combinations_path, usecols=[0], sep='\t')
    comb_atc_codes = combinations.iloc[:, 0].unique()
    return chembl_result[~chembl_result['atc_code'].isin(comb_atc_codes)]

def filter_mapped_dps(ner_output: pd.DataFrame, mapping_data: pd.DataFrame) -> pd.DataFrame:
    """
    Read the NER output for BCB-CRF or MED7 and filter drug products with a mapped and cleaned ATC code.
    """
    cols_to_use = mapping_data.columns[mapping_data.columns != 'TEXT']

    output = ner_output.merge(mapping_data[cols_to_use], left_index=True, right_on='fk', how='right').reset_index(drop=True)
    output[['FORM','ROUTE']] = output[['FORM','ROUTE']].applymap(set)
    return output

def filter_by_atc_code(data: pd.DataFrame, code: str, level: str) -> pd.DataFrame:
    """
    Select drug products by ATC codes - at any level
    """
    levels_length = {
        'level5': 7,
        'level4': 5,
        'level3': 4,
        'level2': 3,
        'level1': 1,
        }
    
    def find_length(level):
        return levels_length.get(level, 'not available')
    
    n = find_length(level)
    return data[data.atc_code.str[0:n]==code].reset_index(drop=True)

def systemic_classifiation_MOD(df, manual_resources_path):
    
    sys_route, non_sys_route, sys_form, non_sys_form, non_sys_atc = load_manually_created_resources(manual_resources_path)

    sistemic_matrix = pd.DataFrame(df.apply(lambda row: (
            bool(row['ROUTE'] & non_sys_route),
            bool(row['FORM'] & non_sys_form),
            bool(row['ROUTE'] & sys_route),
            bool(row['FORM'] & sys_form),
        ), axis=1).tolist(), columns=['NSDPRoute', 'NSDPForm', 'SDPRoute', 'SDPForm'])
    
    df = df[df.columns.difference(['dict_type'], sort=False)]
    df[['class', 'preferred']] = None, False
    
    result = branch(df, sistemic_matrix, non_sys_atc)
    
    nsdp_preferred = (result['class']=='NSDP') & (result['atc_code'].isin(non_sys_atc))
    sdp_preferred = (result['class']=='SDP') & (~result['atc_code'].isin(non_sys_atc))
    
    result.loc[nsdp_preferred | sdp_preferred, 'preferred'] = True
    result.insert(0, 'fk', result.pop('fk'))
    return result

def branch(data, mtx, non_sys_atc):
    
    special_char = data['TEXT'].str.contains("%")
    atc_systemic_exceptions = set(['N02CC03','N02CC01','N02BA10','N02AB03','L02AE01','J070000'])
       
    SDP_route, no_SDP_route = mtx['SDPRoute'], ~mtx['SDPRoute']
    NSDP_route, no_NSDP_route = no_SDP_route & mtx['NSDPRoute'], no_SDP_route & ~mtx['NSDPRoute']
    SDP_form, no_SDP_form = no_NSDP_route & mtx['SDPForm'], no_NSDP_route & ~mtx['SDPForm']
    NSDP_form, no_NSDP_form = no_SDP_form & mtx['NSDPForm'], no_SDP_form & ~mtx['NSDPForm']
    
    sdp = SDP_route | SDP_form
    nsdp = NSDP_route | NSDP_form
    psdp = no_NSDP_form
    
    branch_a = data['atc_code'].isin(non_sys_atc) & ~data['atc_code'].isin(atc_systemic_exceptions)
    contains_special = branch_a & psdp & special_char
    exceptions = data['atc_code'].isin(atc_systemic_exceptions)
    
    data.loc[:, 'class'] = np.select([exceptions, contains_special, sdp, nsdp, psdp], 
                                     ['SDP', 'NSDP', 'SDP', 'NSDP', 'PSDP'], default=None)
    
    return data

def select_systemic(classi_results, preferred=False):
    if preferred:
        return classi_results.loc[(classi_results['class']!='NSDP') & (classi_results['preferred']),
                                 classi_results.columns.difference(['preferred'], sort=False)]
    return classi_results.loc[classi_results['class']!='NSDP']

def categorize_drugs(ner_output, chembl_result):
    caller_path = os.path.dirname(os.path.abspath((inspect.stack()[1])[1]))
    dicts_path = caller_path+'/Resources/chembl_atc_results.json'
    manual_resources_path = caller_path+'/Resources/'
    combinations_path = manual_resources_path+'combination_atc_codes.txt'
    
    chembl_cleaned = load_and_process_chembl_results(chembl_result, dicts_path, combinations_path)
    result = filter_mapped_dps(ner_output, chembl_cleaned)
    return systemic_classifiation_MOD(result,manual_resources_path)