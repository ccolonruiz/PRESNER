#!/usr/local/bin/python

"""
To run this file, use:

python PRESNER.py -i <Input data file path> -o <Output folder> -m <Method 1> ... <Method n> [-s]

# Methods can be: 'cbb', 'med7_trf', 'med7_lg' and/or 'chembl'.

# Use the optional "-s" argument to obtain ATC codes and classify drugs into systemic and non-systemic categories.

e.g.: python -i data.txt -o output_folder -m cbb chembl -s

Please note that you must use Python 3 to get the correct results with this script.


"""

import pickle
import inspect
import os
import sys
import argparse
import pandas as pd
from Utils.data import Dataset
from Utils.systemic_drug_classifier import categorize_drugs
from Utils.evaluate import get_results_cbb, get_results_med7, merge_parallel
from Models.CHEMBL.retrieve import get_results_chembl
from joblib import Parallel, delayed
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
CURRENT_DIR = os.path.dirname(os.path.abspath((inspect.stack()[0])[1]))
MODEL_CFG_FILE = CURRENT_DIR+"/Config/NER_CBBERT.cfg"
METHODS = ['cbb','med7_trf','med7_lg','chembl']

GPU=False
METHODS_N_JOBS = 1    #Number of jobs for parallel execution of the methods.
N_JOBS = 8      #Number of jobs for parallel execution.
CBB_BATCH_SIZE = 128   #Data batch size to be inferred by the CBB model.

def recursive_merge(results,spans):
    """Returns the union of the results provided by different methods."""
    print("Merging results...")
    if len(results)==2:
        return merge_parallel([results[0], results[1]], [spans[0], spans[1]], N_JOBS)
    else:
        results_l, spans_l = recursive_merge(results[1:], spans[1:])
        return merge_parallel([results[0], results_l], [spans[0], spans_l], N_JOBS)
    
def get_all_results_parallel(methods, ds, get_atc_systemic):
    """Returns the results provided by the different methods. Each method is executed in parallel. Warning: Each method executes sections of its code in parallel at the process level."""
    if len(methods)>1:
        results, spans, chembl_result = list(zip(*Parallel(METHODS_N_JOBS)(delayed(globals()[method])(ds,get_atc_systemic) for method in methods)))
        return *recursive_merge(results, spans), chembl_result    
    return globals()[methods[0]](ds, get_atc_systemic)

def get_all_results(methods, ds, get_atc_systemic):
    """Returns the results provided by the different methods. It is executed sequentially."""
    if len(methods)>1:
        results, spans, chembl_result = list(zip(*(globals()[method](ds,get_atc_systemic) for method in methods)))
        return *recursive_merge(results, spans), chembl_result    
    return globals()[methods[0]](ds,get_atc_systemic)

def cbb(ds,get_atc_systemic=False):
    """Returns the results provided by the CBB model"""
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    if GPU:
        gpu = tf.config.experimental.list_physical_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(gpu, True)
    from Models.CBB.CBB import NER_CBBERT

    print("CBB: Loading predictive model...")
    ner = NER_CBBERT(MODEL_CFG_FILE)
    print("CBB: Formatting input data...")
    ds.CBBTransform(ner.tokenizer, ner.cfg, offsets=True, n_jobs=N_JOBS)
    print("CBB: Inferring entities...")
    sents, X_test, y_test, tag_encoder, pad_len, offsets = ds.inputs
    y_pred = ner.model.predict(X_test, batch_size=CBB_BATCH_SIZE)
    print("CBB: Obtaining results...")
 
    return *get_results_cbb(ds, y_pred, n_jobs=N_JOBS), None

def med7_trf(ds,get_atc_systemic=False):
    """Returns the results provided by the MED7_TRF model."""
    if GPU:
        import spacy
        spacy.prefer_gpu()
    import en_core_med7_trf
    print("MED7_TRF: Loading predictive model...")
    med7 = en_core_med7_trf.load()
    med7.add_pipe("doc_cleaner")
    print("MED7_TRF: Formatting input data...")
    print("MED7_TRF: Inferring entities...")
    docs = list(med7.pipe(ds.text))
    print("MED7_TRF: Obtaining results...")
    return *get_results_med7(ds.path, docs), None

def med7_lg(ds,get_atc_systemic=False):
    """Returns the results provided by the MED7_LG model."""
    if GPU:
        import spacy
        spacy.prefer_gpu()
    import en_core_med7_lg
    print("MED7_LG: Loading predictive model...")
    med7 = en_core_med7_lg.load()
    med7.add_pipe("doc_cleaner")
    print("MED7_LG: Formatting input data...")
    print("MED7_LG: Inferring entities...")
    docs = list(med7.pipe(ds.text))
    print("MED7_LG: Obtaining results...")    
    return *get_results_med7(ds.path, docs), None

def chembl(ds,get_atc_systemic=False):
    """Returns the results provided by CHEMBL dictionaries."""
    print("CHEMBL: Obtaining results...")
    result, span = get_results_chembl(ds.path, ds.text)
    if get_atc_systemic:
        return result, span, result
    return result, span, None
  

def predict(data_path, cbb=True, med7_trf=False, med7_lg=False, chembl=True, get_atc_systemic=False):
    """Returns the results according to the methods given as arguments."""
    sig = inspect.signature(predict).bind(data_path, cbb, med7_trf, med7_lg, chembl).arguments
    methods = [k for k,v in sig.items() if v==True]
    
    if get_atc_systemic and ("chembl" not in methods or len(set(methods) - {'chembl'}) == 0):
        raise ValueError("If get_atc_systemic is True, must include 'chembl' and at least one of the other options.")
    
    ds = Dataset(data_path)
    results, spans, chembl_result = get_all_results(methods, ds, get_atc_systemic)
    
    if get_atc_systemic:
        print("Classifying drugs into systemic and non-systemic...")
        results = categorize_drugs(results, chembl_result)

    return results, spans
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument("-i", "--data_path", type=str, required=True, help="Data path (must be a system path, absolute or relative)")
    parser.add_argument("-o", "--result_path", type=str, required=True, help="Output path (must be a system path, absolute or relative)")
    parser.add_argument("-s", "--get_atc_systemic", action="store_true", help="Flag to indicate if you want to get ATC and systemic data")
    parser.add_argument("-m", "--methods", nargs="+", required=True, choices=["cbb", "med7_trf", "med7_lg", "chembl"], help="Methods to use")
    parser.add_argument("-gpu", "--use_gpu", action="store_true", help="Flag to indicate if you want to use GPU acceleration")
    parser.add_argument("-n_jobs", "--n_jobs", type=int, help="Number of jobs for parallel execution", default=N_JOBS)
    parser.add_argument("-bs", "--batch_size", type=int, help="Data batch size to be inferred", default=CBB_BATCH_SIZE)

    args = parser.parse_args()
    
    if args.get_atc_systemic and ("chembl" not in args.methods or len(set(args.methods) - {'chembl'}) == 0):
        parser.error("If --get_atc_systemic is True, 'methods' must include 'chembl' and at least one of the other options.")

    data_path = args.data_path
    result_path = args.result_path
    get_atc_systemic = args.get_atc_systemic
    methods = args.methods
    GPU = args.use_gpu
    N_JOBS = args.n_jobs
    CBB_BATCH_SIZE = args.batch_size
    
    ds = Dataset(data_path)

    results, spans, chembl_result = get_all_results(methods, ds, get_atc_systemic)
    
    if get_atc_systemic:
        print("Classifying drugs into systemic and non-systemic...")
        results = categorize_drugs(results, chembl_result)
    
    results.to_json(result_path+'/result.json')
    pickle.dump(spans,open(result_path+'/span.pkl', "wb"))
    
    print("Results saved in folder "+str(result_path))
