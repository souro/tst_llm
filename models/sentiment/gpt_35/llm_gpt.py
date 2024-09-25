# import logger
import argparse
import time
import socket
import torch
import random
import traceback
import numpy as np
import pandas as pd
import pandas as pd
from my_logger import logger
from openai import OpenAI
from automatic_eval import TSTEvaluator

# logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
def generate_prompt(input_sentence, examples, language, task):
    example_prompts = "\n".join([
        f"Task: {example['task']}\nInput: {example['input']}\nOutput: {example['output']}\n" 
        for example in examples
    ])

    if task == 'pos_to_neg':
        task_text = "positive to negative"
    elif task == 'neg_to_pos':
        task_text = "negative to positive"
    
    prompt = (f"Sentiment transfer change the sentiment of a sentence while keeping the rest of the content unchanged.\n"
              f"Examples:\n\n{example_prompts}\n"
              f"Now change the sentiment of the following {language} sentence.\n"
              f"Task: {task_text}\nInput: {input_sentence}\nOutput: ")
    
    # logger.info(f"Generated prompt: {prompt}")
    return prompt

def perform_sentiment_transfer(client, model_name, input_sentence, examples, language, task):
    content = generate_prompt(input_sentence, examples, language, task)
    logger.info("Generated prompt for sentiment transfer.")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": content}
        ],
        seed=53,
        max_tokens=128,
        temperature=0
    )
    # logger.info(f"Response: {response}")

    output_text = response.choices[0].message.content.strip()
    
    # logger.info(f"Output Text: {output_text}")
    
    return output_text

def process_csv_file(client, args, examples):
    input_csv_file = args.input_csv_file
    output_csv_file = args.output_csv_file
    language = args.language
    src = args.src
    trg = args.trg
    task = args.task
    model_name = args.model_name
    
    logger.info(f"Processing input CSV file: {input_csv_file}")
    df = pd.read_csv(input_csv_file)
    
    # For debuging purpose
    # df = df.iloc[0:2]
    
    new_rows = []
    for index, row in df.iterrows():
        src_sentence = row[src]
        trg_sentence = row[trg]
        
        try:
            pred_sentence = perform_sentiment_transfer(client, model_name, src_sentence, examples, language, task)
        except Exception as e:
            pred_sentence = ""
            logger.error(f"Failed to generate pred sentence for - src_sentence: {src_sentence}, trg_sentence: {trg_sentence}")
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())
        
        new_rows.append({'src': src_sentence, 'trg': trg_sentence, 'pred': pred_sentence})

    output_df = pd.DataFrame(new_rows)
    
    logger.info(f"Writing output to CSV file: {output_csv_file}")
    output_df.to_csv(output_csv_file, index=False, encoding='utf-8')

    tst_evaluator = TSTEvaluator(
        task=args.task,
        lang=args.lang,
        output_file=args.output_csv_file,
        methodology=args.methodology
    )

    logger.info('Starting evaluation process.')
    tst_evaluator.set_seed(53)
    accuracy, similarity_score, bleu, fluency = tst_evaluator.evaluate()
    
    accuracy_text = f", Sentiment Accuracy: {accuracy:.4f}" if accuracy is not None else ", Sentiment Accuracy: None"
    similarity_score_text = f", Similarity: {similarity_score:.4f}" if similarity_score is not None else ", Similarity: None"
    bleu_text = f", Bleu Score: {bleu:.4f}" if bleu is not None else ", Bleu Score: None"
    fluency_text = f", Fluency: {fluency:.4f}" if fluency is not None else ", Fluency: None"
    
    logger.info(f"Language: {args.lang}, methodology: {args.methodology}, task: {args.task} {accuracy_text}{similarity_score_text}{bleu_text},{fluency_text}")    

def read_examples(prompt_input_csv_file):
    logger.info(f"Reading prompt input CSV file: {prompt_input_csv_file}")
    df = pd.read_csv(prompt_input_csv_file)
    examples = []
    for i in range(4):
        if i % 2 == 0:
            examples.append({'task': 'positive to negative', 'input': df.at[i, 'POSITIVE'], 'output': df.at[i, 'NEGATIVE']})
        else:
            examples.append({'task': 'negative to positive', 'input': df.at[i, 'NEGATIVE'], 'output': df.at[i, 'POSITIVE']})
    return examples

def main(args):    
    client = OpenAI(api_key='')
    examples = read_examples(args.prompt_input_csv_file)
    process_csv_file(client, args, examples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Transfer using LLM ChatGPT")
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
    args = parser.parse_args()

    start_time = time.time()
    logger.info("Command-line arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device {device}')    

    if torch.cuda.is_available():
        logger.info(f'GPU machine name: {torch.cuda.get_device_name(0)}')
        logger.info(f'SSH machine name: {socket.gethostname()}')
    
    set_seed(53)
    
    main(args)

    end_time = time.time()
    time_taken_seconds = end_time - start_time
    time_taken_formatted = time.strftime('%H:%M:%S', time.gmtime(time_taken_seconds))

    logger.info(f"Time taken for language: {args.lang}, methodology: {args.methodology}, task: {args.task} - {time_taken_formatted}")

    logger.info('=' * 50 + '\n')