#!/bin/bash

# if [ "$#" -ne 3 ]; then
#     echo "Usage: $0 <model_name> <methodology> <prompt_type>"
#     exit 1
# fi

model_name="$1"
methodology="$2"
prompt_type="$3"

if [ "$4" == "quantize" ]; then
    quantize_flag="--quantize"
else
    quantize_flag=""
fi

languages=("English" "Hindi" "Bengali")
codes=("en" "hi" "bn")
tasks=("pos_to_neg" "neg_to_pos")


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
    
    output_csv_file="../output/sentiment/${task}-${code}-${methodology}_${prompt_type}.csv"

    echo "Language: $language"
    echo "Task: $task"
    echo "Model Name: $model_name"
    echo "Prompt Type: $prompt_type"
    # echo "Methodology: $methodology"
    
    python llms.py \
    --prompt_input_csv_file "$prompt_input_csv_file" \
    --input_csv_file "$input_csv_file" \
    --output_csv_file "$output_csv_file" \
    --language "$language" \
    --lang "$code" \
    --src "$src" \
    --trg "$trg" \
    --task "$task" \
    --model_name "$model_name" \
    --prompt_type "$prompt_type" \
    --methodology "${methodology}_${prompt_type}" \
    --log_file_name "${methodology}_${prompt_type}" \
    $quantize_flag
  done
done