# PRESNER

A name entity recognition (NER) and processing pipeline for Electronic Health Records (EHR) prescription data tailored to the needs of the human genetics research community working with UK Biobank (UKB). \
We anticipate that the pipeline will also be used to process other prescription datasets, such as CPRD data or hospital EHR datasets.


PRESNER combines a deep learning model trained with manually annoted clinical notes and UKB prescription entries, drug dictionaries extracted from the ChEMBL database and manually created resources and rules.

### Usage

To run the PRESNER pipeline, use:

`python PRESNER.py <Input data file path> <Output folder> <Method 1> ... <Method n>`

Methods can be: 'cbb', 'med7_trf', 'med7_lg' and/or 'chembl'.

e.g.: `python data.txt output_folder cbb chembl`

#### Configuration settings:

The file PRESNER.py contains the following values:

- GPU: Default value "False". To run using GPU set to "True".
- METHODS_N_JOBS: Number of jobs for parallel execution of the methods. Use function get_all_results_parallel instead of get_all_results.
- MERGE_N_JOBS: Number of jobs for parallel execution of the results union.
- CBB_N_JOBS: Number of jobs for parallel execution of data processing for the CBB model.
- CBB_BATH_SIZE: Data batch size to be inferred by the CBB model.

#### Input data format:

#### Results:

The following results are stored in the specified folder `<Output folder>`:

- result.json: Shows the processed texts, the matched entities and the original texts from the columns "drug_name" and "quantity".
- span.pkl: Shows the entities, entity type and their offsets in the processed texts of a given index.
  
![alt text](https://github.com/mariaheza/CLINICAL_DRUGS_NER/blob/main/PRESNER/images/Results.png?raw=true)

#### Run from jupyter notebook:

PRESNER.py can be run directly from jupyter notebook. The results are stored in `<output_folder>`.

e.g.: `!python "<data.txt>" "<output_folder>" "cbb" "chembl"`

Moreover, PRESNER can display results in memory as follows:
  
![alt text](https://github.com/mariaheza/CLINICAL_DRUGS_NER/blob/main/PRESNER/images/Beauty.png?raw=true)

Warning: beauty_display shows all rows indicated in "result". If result contains a large number of rows, it is preferable to display small subsets.

### Requirements:

[Python](https://www.python.org/) (PRESNER was tested with Python version >=3.9.7) 

#### Relevant software libraries employed:

- [Tensorflow](https://www.tensorflow.org/?hl=es-419) (version 2.7.0)
- [Tensorflow-addons](https://www.tensorflow.org/addons?hl=es-419) (version 0.16.1)
- [Spacy](https://spacy.io/) (version 3.1.5)
- [Transformers](https://pypi.org/project/transformers/) (version 4.20.1)
- [en_core_med7_trf](https://github.com/kormilitzin/med7) (version 3.1.3.1)
- [en_core_med7_lg](https://github.com/kormilitzin/med7) (version 3.1.3.1)
- [Pandas](https://pandas.pydata.org/docs/index.html) (version 1.4.1)
- [Numpy](https://numpy.org/) (version 1.21.2)
- [Sklearn](https://scikit-learn.org/stable/#) (version 1.0.2)
- [Joblib](https://joblib.readthedocs.io/en/latest/) (version 1.1.0)
- [Srsly](https://pypi.org/project/srsly/) (version 2.4.1)
- [Tqdm](https://pypi.org/project/tqdm/) (version 4.63.0)
- [Yaml](https://pypi.org/project/PyYAML/) (version 6.0)

If you use jupyter notebook to visualise the results with beauty_display:
- [IPython](https://ipython.org/) (tested with version 8.1.1)
