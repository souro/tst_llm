#!/bin/bash

source ../../../env_llm/bin/activate

python eval_gpt4_toxicity.py --model_name gpt-4-turbo --input_csv_file human_eval_toxicity_en.csv --language English
python eval_gpt4_toxicity.py --model_name gpt-4-turbo --input_csv_file human_eval_toxicity_hi.csv --language Hindi