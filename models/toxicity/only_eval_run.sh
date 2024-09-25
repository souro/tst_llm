#!/bin/bash

source ../../../env_llm/bin/activate

dir="../output/toxicity/temp/"

for filepath in "$dir"*.csv; do
    filename=$(basename -- "$filepath")
    
    base_filename="${filename%.*}"
    
    IFS='-' read -r lang methodology <<< "$base_filename"
    
    output_file="$dir$filename"
    
    python only_automatic_eval.py --lang "$lang" --output_file "$output_file" --methodology "$methodology"
done