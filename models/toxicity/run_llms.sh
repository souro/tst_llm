source ../../../env_llm/bin/activate

########################################################################################

./llms.sh "/lnet/express/work/people/limisiewicz/hf_llama/llama_7B" "Llama_7b" "zero_shot"
./llms.sh "/lnet/express/work/people/limisiewicz/hf_llama/llama_7B" "Llama_7b" "few_shot"

./llms.sh "/lnet/express/work/people/limisiewicz/hf_llama/llama_13B" "Llama_13b" "zero_shot" "quantize"
./llms.sh "/lnet/express/work/people/limisiewicz/hf_llama/llama_13B" "Llama_13b" "few_shot" "quantize"

./llms.sh "/lnet/express/work/people/limisiewicz/hf_llama/llama_30B" "Llama_30b" "zero_shot" "quantize"
./llms.sh "/lnet/express/work/people/limisiewicz/hf_llama/llama_30B" "Llama_30b" "few_shot" "quantize"

#######################################################################################

./llms.sh "meta-llama/Llama-2-7b-hf" "Llama2_7b" "zero_shot"
./llms.sh "meta-llama/Llama-2-7b-hf" "Llama2_7b" "few_shot"

./llms.sh "meta-llama/Llama-2-13b-hf" "Llama2_13b" "zero_shot" "quantize"
./llms.sh "meta-llama/Llama-2-13b-hf" "Llama2_13b" "few_shot" "quantize"

########################################################################################

./llms.sh "meta-llama/Llama-2-7b-chat-hf" "Llama2_7b_chat" "zero_shot"
./llms.sh "meta-llama/Llama-2-7b-chat-hf" "Llama2_7b_chat" "few_shot"

./llms.sh "meta-llama/Llama-2-13b-chat-hf" "Llama2_13b_chat" "zero_shot" "quantize"
./llms.sh "meta-llama/Llama-2-13b-chat-hf" "Llama2_13b_chat" "few_shot" "quantize"

###################################################################################

./llms.sh "meta-llama/Meta-Llama-3-8B" "Llama3_8b" "zero_shot"
./llms.sh "meta-llama/Meta-Llama-3-8B" "Llama3_8b" "few_shot"

##############################################################################
# Need to change the python file corresponding to chatglm in llms.sh and venv too

./llms.sh "THUDM/chatglm-6b" "chatglm_6b" "zero_shot"
./llms.sh "THUDM/chatglm-6b" "chatglm_6b" "few_shot"

./llms.sh "THUDM/chatglm2-6b" "chatglm2_6b" "zero_shot"
./llms.sh "THUDM/chatglm2-6b" "chatglm2_6b" "few_shot"

#########################################################################

./llms.sh "facebook/opt-1.3b" "opt_1b" "zero_shot"
./llms.sh "facebook/opt-1.3b" "opt_1b" "few_shot"

./llms.sh "facebook/opt-2.7b" "opt_2b" "zero_shot"
./llms.sh "facebook/opt-2.7b" "opt_2b" "few_shot"

./llms.sh "facebook/opt-6.7b" "opt_6b" "zero_shot"
./llms.sh "facebook/opt-6.7b" "opt_6b" "few_shot"

./llms.sh "facebook/opt-13b" "opt_13b" "zero_shot" "quantize"
./llms.sh "facebook/opt-13b" "opt_13b" "few_shot" "quantize"

./llms.sh "facebook/opt-30b" "opt_30b" "zero_shot" "quantize"
./llms.sh "facebook/opt-30b" "opt_30b" "few_shot" "quantize"

#########################################################################

./llms.sh "tiiuae/falcon-7b" "falcon_7b" "zero_shot"
./llms.sh "tiiuae/falcon-7b" "falcon_7b" "few_shot"

###########################################################################

./llms.sh "bigscience/bloom-560m" "bloom_560m" "zero_shot"
./llms.sh "bigscience/bloom-560m" "bloom_560m" "few_shot"

./llms.sh "bigscience/bloom-1b7" "bloom_1b" "zero_shot"
./llms.sh "bigscience/bloom-1b7" "bloom_1b" "few_shot"

./llms.sh "bigscience/bloom-3b" "bloom_3b" "zero_shot"
./llms.sh "bigscience/bloom-3b" "bloom_3b" "few_shot"

./llms.sh "bigscience/bloom-7b1" "bloom_7b" "zero_shot"
./llms.sh "bigscience/bloom-7b1" "bloom_7b" "few_shot"

#############################################################################

./llms.sh "bigscience/bloomz-560m" "bloomz_560m" "zero_shot"
./llms.sh "bigscience/bloomz-560m" "bloomz_560m" "few_shot"

./llms.sh "bigscience/bloomz-1b7" "bloomz_1b" "zero_shot"
./llms.sh "bigscience/bloomz-1b7" "bloomz_1b" "few_shot"

./llms.sh "bigscience/bloomz-3b" "bloomz_3b" "zero_shot"
./llms.sh "bigscience/bloomz-3b" "bloomz_3b" "few_shot"

./llms.sh "bigscience/bloomz-7b1" "bloomz_7b" "zero_shot"
./llms.sh "bigscience/bloomz-7b1" "bloomz_7b" "few_shot"

