#!/bin/bash

source ../../../env_llm/bin/activate

# python eval_gpt4_sentiment.py --model_name gpt-4-turbo --input_csv_file human_eval_sentiment_en.csv --language English
# python eval_gpt4_sentiment.py --model_name gpt-4-turbo --input_csv_file human_eval_sentiment_hi.csv --language Hindi
python eval_gpt4_sentiment.py --model_name gpt-4-turbo --input_csv_file human_eval_sentiment_bn.csv --language Bengali