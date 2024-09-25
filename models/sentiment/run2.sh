source ../../../env_llm/bin/activate

./llms.sh "meta-llama/Llama-2-13b-hf" "Llama2_13b" "zero_shot" "quantize"
./llms.sh "meta-llama/Llama-2-13b-hf" "Llama2_13b" "few_shot" "quantize"

# ./llms.sh "THUDM/chatglm-6b" "chatglm_6b" "zero_shot"
# ./llms.sh "THUDM/chatglm-6b" "chatglm_6b" "few_shot"

# ./llms.sh "THUDM/chatglm2-6b" "chatglm2_6b" "zero_shot"
# ./llms.sh "THUDM/chatglm2-6b" "chatglm2_6b" "few_shot"

# ./llms.sh "bigscience/bloomz-3b" "bloomz_3b" "zero_shot"
# ./llms.sh "bigscience/bloomz-3b" "bloomz_3b" "few_shot"

# ./llms.sh "bigscience/bloomz-7b1" "bloomz_7b" "zero_shot"
# ./llms.sh "bigscience/bloomz-7b1" "bloomz_7b" "few_shot"