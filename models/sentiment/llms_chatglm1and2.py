 # import logger
import argparse
import time
import socket
import torch
import random
import traceback
import re
import numpy as np
import pandas as pd
import pandas as pd
from my_logger import setup_logger
# from openai import OpenAI
# from automatic_eval import TSTEvaluator
# import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModel #ForCausalLM

def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
def generate_prompt(logger, args, examples, src_sentence):
    if args.task == 'pos_to_neg':
        task_text = "positive to negative"
    elif args.task == 'neg_to_pos':
        task_text = "negative to positive"
    
    task_definition = f"Sentiment transfer changes the sentiment of a sentence while keeping the rest of the content unchanged."
    instruction = f"Now change the sentiment of the following {args.language} sentence."
    input = f"Task: {task_text}\nInput: {src_sentence}\nOutput: "
    
    prompt = None
    if args.prompt_type == 'zero_shot':
        prompt = (f"{task_definition}\n"
                  f"{instruction}\n"
                  f"{input}")
    elif args.prompt_type == 'few_shot':
        example_prompts = "\n".join([
            f"Task: {example['task']}\nInput: {example['input']}\nOutput: {example['output']}\n" 
            for example in examples
        ])
        
        prompt = (f"{task_definition}\n"
                  f"\nExamples:\n\n{example_prompts}\n"
                  f"{instruction}\n"
                  f"{input}")
    
    # logger.info(f"Generated prompt: {prompt}")
    
    return prompt

def clean_text(text):
    patterns = [
        r'\(negative\)',
        r'\(positive\)',
        r'\(negative to positive\)',
        r'\(positive to negative\)',
        r'\(',
        r'\)'
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    
    text = text.strip()
    
    if (text.startswith('“') and text.endswith('”')) or (text.startswith('‘') and text.endswith('’')):
        text = text[1:-1]
    elif text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    return text.strip()


def extract_output_text(output):
    section_start = "Now change the sentiment of the following"
    if section_start in output:
        relevant_part = output.split(section_start, 1)[1]
        
        lines = relevant_part.split('\n')
        
        for line in lines:
            if line.startswith('Output:'):
                return clean_text(line[len('Output:'):].strip())
    
    return ""

def perform_sentiment_transfer(logger, tokenizer, model, args, examples, src_sentence):
    prompt = generate_prompt(logger, args, examples, src_sentence)
    # logger.info("Generated prompt for sentiment transfer.")

    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # generate_ids = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
    # output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    output = model.chat(tokenizer, prompt, history=[])
    # logger.info(f"Output Text: {output}")
    
    # return extract_output_text(output)
    return output[0]

def process_csv_file(logger, args, examples):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device {device}')    

    if torch.cuda.is_available():
        logger.info(f'GPU machine name: {torch.cuda.get_device_name(0)}')
        logger.info(f'SSH machine name: {socket.gethostname()}')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # if args.quantize:
    #     model = AutoModelForCausalLM.from_pretrained(
    #         args.model_name,
    #         load_in_8bit=True,  # Quantize the model to 8-bit
    #         device_map='auto',
    #     )
    # else:
        # model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).half().to(device)

    # logger.info(f"Model: {args.model_name} Loaded successfully")
    
    # logger.info(f"Processing input CSV file: {input_csv_file}")
    df = pd.read_csv(args.input_csv_file)
    # df = df.iloc[0:3]
    
    new_rows = []
    for index, row in df.iterrows():
        src_sentence = row[args.src]
        trg_sentence = row[args.trg]

        pred_sentence = None
        try:
            pred_sentence = perform_sentiment_transfer(logger, tokenizer, model, args, examples, src_sentence)
            # logger.info(f"Clean Predicted Output: {pred_sentence}")
        
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())
            # pred_sentence = ''
        
        new_rows.append({'src': src_sentence, 'trg': trg_sentence, 'pred': pred_sentence})

    output_df = pd.DataFrame(new_rows)
    
    # logger.info(f"Writing output to CSV file: {output_csv_file}")
    output_df.to_csv(args.output_csv_file, index=False, encoding='utf-8')

    # tst_evaluator = TSTEvaluator(
    #     task=args.task,
    #     lang=args.lang,
    #     output_file=args.output_csv_file,
    #     methodology=args.methodology
    # )

    # logger.info('Starting evaluation process.')
    # tst_evaluator.set_seed(53)
    # accuracy, similarity_score, bleu = tst_evaluator.evaluate()
    
    # accuracy_text = f", Sentiment Accuracy: {accuracy:.2f}" if accuracy is not None else ", Sentiment Accuracy: None"
    # similarity_score_text = f", Similarity: {similarity_score:.2f}" if similarity_score is not None else ", Similarity: None"
    # bleu_text = f", Bleu Score: {bleu:.2f}" if bleu is not None else ", Bleu Score: None"
    
    # logger.info(f"Language: {args.lang}, methodology: {args.methodology}, task: {args.task} {accuracy_text}{similarity_score_text}{bleu_text}")
    

def read_examples(logger, args):
    df = pd.read_csv(args.prompt_input_csv_file)
    examples = []
    for i in range(4):
        if i % 2 == 0:
            examples.append({'task': 'positive to negative', 'input': df.at[i, 'POSITIVE'], 'output': df.at[i, 'NEGATIVE']})
        else:
            examples.append({'task': 'negative to positive', 'input': df.at[i, 'NEGATIVE'], 'output': df.at[i, 'POSITIVE']})
    return examples

def main(args):
    logger = setup_logger(args.log_file_name)
    
    logger.info("Command-line arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    set_seed(53)
    
    start_time = time.time()
    
    examples = read_examples(logger, args)
    process_csv_file(logger, args, examples)
    
    end_time = time.time()
    
    time_taken_seconds = end_time - start_time
    time_taken_formatted = time.strftime('%H:%M:%S', time.gmtime(time_taken_seconds))

    logger.info(f"Time taken for language: {args.lang}, methodology: {args.methodology}, task: {args.task} - {time_taken_formatted}")

    logger.info('=' * 50 + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Transfer using LLM ChatGPT")
    
    parser.add_argument('--log_file_name', type=str, default='tst_llm', help='Log File name')
    parser.add_argument('--model_name', type=str, required=True, help='Pretrained LLM name')
    parser.add_argument('--prompt_input_csv_file', required=True, help="Path to the prompt input CSV file")
    parser.add_argument('--src', type=str, required=True, help='')
    parser.add_argument('--trg', type=str, required=True, help='')
    parser.add_argument('--task', type=str, required=True, help='Task (positive to negative or negative to positive')
    parser.add_argument('--input_csv_file', required=True, help="Path to the input CSV file")
    parser.add_argument('--output_csv_file', required=True, help="Path to the output CSV file")
    parser.add_argument('--lang', required=True, help="Language code of the input sentences")
    parser.add_argument('--language', required=True, help="Language of the input sentences")
    parser.add_argument('--methodology', type=str, required=True, help='Methodology LLM')
    parser.add_argument('--prompt_type', type=str, required=True, help='Prompt Type (zero_shot or few_shot)')
    parser.add_argument("--quantize", action="store_true", help="Set this flag for model quantization")
    
    args = parser.parse_args()
    
    main(args)