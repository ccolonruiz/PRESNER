# PRESNER

This repository contains the first version of [Automated Extraction and Classification of Drug Prescriptions in Electronic Health Records: Introducing the PRESNER Pipeline](https://doi.org/10.1101/2023.10.04.23296481)

PRESNER is a name entity recognition (NER) and processing pipeline for Electronic Health Records (EHR) prescription data tailored to the needs of the human genetics research community working with UK Biobank (UKB). 
We anticipate that the pipeline will also be used to process other prescription datasets, such as CPRD data or hospital EHR datasets.

PRESNER combines a deep learning model trained with manually annoted clinical notes and UKB prescription entries, drug dictionaries extracted from the ChEMBL database and manually created resources and rules.

## This work has been conducted by:

Colón-Ruiz, Cristóbal<sup>1</sup>; 
Fitzgerald, Tomas<sup>2</sup>; 
Segura-Bedmar, Isabel<sup>1</sup>; 
Birney, Ewan<sup>2</sup>;
Herrero-Zazo, Maria<sup>2</sup>

<sup>1</sup>Computer Science Department, University Carlos III of Madrid, Avenida de la Universidad 30, 28911, Leganés, Madrid, Spain. \
<sup>2</sup>European Molecular Biology Laboratory, European Bioinformatics Institute (EMBL-EBI), Wellcome Genome Campus, Hinxton, Cambridgeshire, UK.

## Usage

To use the PRESNER pipeline, you can choose one of the following options:

1) Clone the repository and use the requirements.txt file to install all dependencies (recommended to use a virtual environment).
   
   ```
   git clone https://github.com/ccolonruiz/PRESNER.git
   cd ./PRESNER/PRESNER_dir
   conda create -y -n PRESNER python=3.9
   conda activate PRESNER
   pip install --no-cache-dir -r ../requirements.txt -f https://download.pytorch.org/whl/cu111/torch_stable.html
   ```
   Optional: If you want to use GPU acceleration, you must install the appropriate version of [CUDA Toolkit 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive) for your system.
   
2) Use singularity to work with a container:

   ```
   singularity pull library://ccolonruiz/embl-ebi/presner:11.2.2-ubuntu20.04
   mv ./presner:11.2.2-ubuntu20.04.sif ./PRESNER.sif
   ```
   This software contains source code provided by NVIDIA Corporation. ([NVIDIA Deep learning Container License](https://developer.download.nvidia.com/licenses/NVIDIA_Deep_Learning_Container_License.pdf?t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLyJ9)).

## Running PRESNER:
In case 1, use:

```
python PRESNER.py \
	-i <Input data file path> \
	-o <Output folder> \
	-m <Method 1> ... <Method n> \
	[-s] [-gpu] [-n_jobs] <Jobs for parallel execution> [-bs] <Data batch size to be inferred>
```

In case 2, use:

```
singularity run [--nv] PRESNER.sif \
	-i <Input data file path> \
	-o <Output folder> -m <Method 1> ... <Method n> \
	[-s] [-gpu] [-n_jobs] <Jobs for parallel execution> [-bs] <Data batch size to be inferred>
```

Arguments between `[]` are optional, and their default values are as follows:

- --nv: `False`. Flag that enables access to GPUs within a Singularity container; if not used, the default value will be False.
- -s, --get_atc_systemic: `False`. Flag to indicate if you want to get ATC codes and systemic data classification.
- -gpu, --use_gpu: `False`. Flag to indicate if you want to use GPU acceleration; if not used, the default value will be False.
- -n_jobs, --n_jobs: `8`. Number of jobs for parallel execution.
- -bs, --batch_size: `128`. Data batch size to be inferred

Methods from -m can be: 'cbb', 'med7_trf', 'med7_lg' and/or 'chembl'.

e.g. (case 1): 
```bash
python PRESNER.py -i data.txt -o output_folder -m cbb chembl -s -gpu -n_jobs 4 -bs 32
```

e.g. (case 2): 
```bash
singularity run --nv PRESNER.sif -i data.txt -o output_folder -m cbb chembl -s -gpu -n_jobs 4 -bs 32
```

## Input Data Description:

The input data are in a text file (extension `.txt`) and follow the following format:

- The text file has two columns as a header, separated by a tabulation (`\t`):
	- "drug_name": This column might contain the text with the name of the prescribed medicinal product.
	- "quantity": This column might contain information on the quantity of the prescribed medicine.

- The rest of the rows contain the texts of the prescriptions, where each row represents a different prescription.
- In case a prescription has additional information on the drug's quantity, this information can be included in an extra second column, just like the header, using the tabulation (`\t`) as a separator.

An example of the data input format is shown below, and examples can also be found in the [datasets](https://github.com/ccolonruiz/PRESNER/tree/main/datasets) folder:

|drug_name|quantity|
|-----------|-----------|
|Amoxicillin 500 mg capsules|21|
|Erythromycin 250 mg gastro - resistant tablets|56 tablets -250 mg|
|Clotrimazole 1% cream|20 grams - 1 %|

## Results:

The following results are stored in the specified folder `<Output folder>`:

- result.json: Shows the processed texts, the matched entities and the original texts from the columns "drug_name" and "quantity".
- span.pkl: Shows the entities, entity type and their offsets in the processed texts of a given index.
  
![alt text](https://github.com/ccolonruiz/PRESNER/blob/main/images/df_ns_no_beauty.png?raw=true)

## Run from Jupyter Notebook:

When PRESNER is used in both options within a virtual environment or a Singularity container, Jupyter Notebook can effectively display the results by highlighting the text entities using the `beauty_display` class of the [display.py](https://github.com/ccolonruiz/PRESNER/blob/main/PRESNER_dir/Utils/display.py) script. Using the Singularity container, you can run Jupyter Notebook using the `notebook` argument. In addition, you can specify the rest of the arguments that you would normally specify when starting Jupyter Notebook. For example:

```bash
singularity run PRESNER.sif notebook --ip 0.0.0.0 --port 8887
```
Once inside the notebook, to display the highlighted texts, you must read the files described in the result section and use them to create an object of the beauty_display class:

```python
import pandas as pd
from display import beauty_display
result = pd.read_json("/path/to/result.json")
span = pd.read_pickle("/path/to/span.pkl")
beauty = beauty_display(result, span, "TEXT", n_jobs=8)
beauty[:10]
```
The above code will display a table like the following:

![alt text](https://github.com/ccolonruiz/PRESNER/blob/main/images/df_ns.png?raw=true)

## ATC codes and systemic data classification:

In the case of running PRESNER with the `[-s]` argument ('methods' must include 'chembl' and at least one of the other options), you can use the "select_systemic" function of the [systemic_drug_classifier.py](https://github.com/ccolonruiz/PRESNER/blob/main/PRESNER_dir/Utils/systemic_drug_classifier.py) script to filter out the systemic drugs and observe the allocation of ATC codes: 

```python
from systemic_drug_classifier import select_systemic
systemic_result = select_systemic(result, preferred=False)
beauty = beauty_display(systemic_result, span, "TEXT", n_jobs=8)
beauty[:10]
```

If you use `preferred=False`, note that different ATC codes are assigned to each drug, indicating those with higher confidence in the "preferred" column:

![alt text](https://github.com/ccolonruiz/PRESNER/blob/main/images/df_s_ssf.png?raw=true)

If you use `preferred=True`, only the ATC codes assigned with the highest confidence will be displayed:

![alt text](https://github.com/ccolonruiz/PRESNER/blob/main/images/df_s_sst.png?raw=true)

## Use of beauty_display objects:

The beauty_display objects consist mainly of a pandas dataframe of result.json enriched with HTML and CSS. As such, slices or indices can access the different rows. For example:

```python
beauty[:10] # Displays the first 10 rows
beauty[5] # Displays the fifth row
beauty[beauty.df['atc_code'] == 'J01CA04'] # Displays all rows whose prescriptions contain drugs with the ATC code J01CA04
```

## Citing:
If you found this work useful, please cite it as:
```
@article {ColnRuiz2023presner,
	title = {Automated Extraction and Classification of Drug Prescriptions in Electronic Health Records: Introducing the PRESNER Pipeline},
	author = {Crist{\'o}bal Col{\'o}n-Ruiz and Tomas W Fitzgerald and Isabel Segura-Bedmar and Ewan Birney and Maria Herrero-Zazo},
	doi = {10.1101/2023.10.04.23296481},
	journal = {medRxiv},
	year = {2023}
}
```
