#!/bin/bash

source ../../../env_llm/bin/activate

# languages=("English" "Hindi" "Bengali")
# codes=("en" "hi" "bn")

languages=("Bengali")
codes=("bn")

tasks=("pos_to_neg" "neg_to_pos")
# tasks=("pos_to_neg")

model_name="gpt-3.5-turbo"
methodology="gpt35_FS"

for i in "${!languages[@]}"; do
  language="${languages[$i]}"
  code="${codes[$i]}"
  
  prompt_input_csv_file="../data/sentiment/${code}_train.csv"
  input_csv_file="../data/sentiment/${code}_test.csv"

  for task in "${tasks[@]}"; do
    if [ "$task" == "pos_to_neg" ]; then
      src="POSITIVE"
      trg="NEGATIVE"
    else
      src="NEGATIVE"
      trg="POSITIVE"
    fi
    
    output_csv_file="../output/sentiment/gpt35/${task}-${code}-${methodology}.csv"
    
    python llm_gpt.py --prompt_input_csv_file "$prompt_input_csv_file" --input_csv_file "$input_csv_file" --output_csv_file "$output_csv_file" --language "$language" --lang "$code" --src "$src" --trg "$trg" --task "$task" --model_name "$model_name" --methodology "$methodology"
  done
done