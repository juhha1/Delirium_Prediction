# Delirium_Prediction

## install requirements
pip install -r requirements.txt

## make input data
### SMC data
python code/data_extraction_smc.py
### MIMIC data
python code/data_extraction_mimic.py

## Make model from SMC data
python code/make_model_smc.py

## MIMIC external validation
python code/tpot_code_generate.py

## TPOT
python code/tpot_code_generate.py

*** Note that all codes are executable ONLY if data exist.