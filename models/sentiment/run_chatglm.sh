source ../../../venv_llm_fntn_temp/bin/activate

# Change the py file in llms.sh from llms.py -> llms_chatglm

./llms.sh "THUDM/chatglm-6b" "chatglm_6b" "zero_shot"
./llms.sh "THUDM/chatglm-6b" "chatglm_6b" "few_shot"

./llms.sh "THUDM/chatglm2-6b" "chatglm2_6b" "zero_shot"
./llms.sh "THUDM/chatglm2-6b" "chatglm2_6b" "few_shot"