# PRESNER

A name entity recognition (NER) and processing pipeline for Electronic Health Records (EHR) prescription data tailored to the needs of the human genetics research community working with UK Biobank (UKB). \
We anticipate that the pipeline will also be used to process other prescription datasets, such as CPRD data or hospital EHR datasets.

PRESNER combines a deep learning model trained with manually annoted clinical notes and UKB prescription entries, drug dictionaries extracted from the ChEMBL database and manually created resources and rules.

## Usage

To use the PRESNER pipeline, you can choose one of the following options:

1) Clone the repository and use the requirements.txt file to install all dependencies (recommended to use a virtual environment).
2) Use singularity to work with a container:

   `singularity pull library://ccolonruiz/embl-ebi/presner:11.2.2-ubuntu20.04`

   This software contains source code provided by NVIDIA Corporation.

## Running PRESNER:
In case 1, use:

`python PRESNER.py -i <Input data file path> -o <Output folder> -m <Method 1> ... <Method n> [-s] [-gpu] [-n_jobs] <Jobs for parallel execution> [-bs] <Data batch size>`

In case 2, use:

`singularity run [--nv] PRESNER.sif -i <Input data file path> -o <Output folder> -m <Method 1> ... <Method n> [-s] [-gpu] [-n_jobs] <Jobs for parallel execution> [-bs] <Data batch size>`

Arguments between `[]` are optional, and their default values are as follows:

- --nv: Flag that enables access to GPUs within a Singularity container; if not used, the default value will be False.
- -s, --get_atc_systemic: `False`. Flag to indicate if you want to get ATC and systemic data classification.
- -gpu, --use_gpu: `False`. Flag to indicate if you want to use GPU acceleration; if not used, the default value will be False.
- -n_jobs, --n_jobs: `8`. Number of jobs for parallel execution.
- -bs, --batch_size: `128`. Data batch size to be inferred

Methods from -m can be: 'cbb', 'med7_trf', 'med7_lg' and/or 'chembl'.

e.g. (case 1): `python PRESNER.py -i data.txt -o output_folder -m cbb chembl -s -gpu -n_jobs 4 -bs 32`

e.g. (case 2): `singularity run --nv PRESNER.sif -i data.txt -o output_folder -m cbb chembl -s -gpu -n_jobs 4 -bs 32`

## Input Data Description:

The input data are in a text file (extension `.txt`) and follow the following format:

- The text file has two columns as a header, separated by a tabulation (`\t`):
	- "drug_name": This column might contain the text with the name of the prescribed medicinal product.
	- "quantity": This column might contain information on the quantity of the prescribed medicine.

- The rest of the rows contain the texts of the prescriptions, where each row represents a different prescription.
- In case a prescription has additional information on the drug's quantity, this information can be included in an extra second column, just like the header, using the tabulation (`\t`) as a separator.

An example of the format of the input data is shown below:

|drug_name|quantity|
|-----------|-----------|
|Amoxicillin 500 mg capsules|21|
|Erythromycin 250 mg gastro - resistant tablets|56 tablets -250 mg|
|Clotrimazole 1% cream|20 grams - 1 %|

## Results:

The following results are stored in the specified folder `<Output folder>`:

- result.json: Shows the processed texts, the matched entities and the original texts from the columns "drug_name" and "quantity".
- span.pkl: Shows the entities, entity type and their offsets in the processed texts of a given index.
  
![alt text](https://github.com/mariaheza/CLINICAL_DRUGS_NER/blob/main/PRESNER/images/Results.png?raw=true)

## Run from Jupyter Notebook:

When using PRESNER in both options, whether it's within a virtual environment or a Singularity container, Jupyter Notebook can effectively display results by highlighting text entities through the beauty_display class, which you can import with: `from display import beauty_display`.

In the container environment, you have the added convenience of running Jupyter Notebook with the following command, for example:

`singularity run PRESNER.sif notebook --ip 0.0.0.0 --port 8887`

Furthermore, the beauty_display class facilitates the exploration of results by enabling the use of slices and indices to access different dataframe rows. Additionally, it empowers you to perform masked searches on indexes, as exemplified here `beauty_result[beauty_result.df.apply(lambda row: 'amoxicillin' in row['DRUG'], axis=1)]`:



In the case of running PRESNER with the -s argument, you can use the "select_systemic" function to filter out the systemic ones and observe the allocation of ATC codes. 
Note that different ATC codes are assigned to each drug, indicating those with higher confidence in the "preferred" column:

