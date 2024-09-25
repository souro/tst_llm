#!/bin/bash

dir="../../multilingual_tst/output/bn/"
# dir="../output/sentiment/gpt35/"
# dir="../output/sentiment/sotas/"

for filepath in "$dir"*.csv; do
    filename=$(basename -- "$filepath")
    
    base_filename="${filename%.*}"
    
    IFS='-' read -r task lang methodology <<< "$base_filename"
    
    output_file="$dir$filename"
    
    # if [[ "$methodology" == *"finetune"* ]]; then
    python only_automatic_eval.py --task "$task" --lang "$lang" --output_file "$output_file" --methodology "$methodology"
    # fi
done