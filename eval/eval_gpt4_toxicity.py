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

# logger.basicConfig(level=logger.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def generate_acc_prompt(df_row, language):
    task_definition = ('Text detoxification changes a toxic sentence to a clean sentence while keeping the rest of the content unchanged.\n')
    
    instruction = (
        f"Please rate the accuracy of the detoxification process between the following {language} "
        "toxic sentence S1 and the resulting clean sentence S2. Use a scale of 1 to 5, where 1 indicates that "
        "S2 is as toxic as S1, and 5 indicates that S2 has been completely transformed into a clean sentence.\n"
        f"S1: {df_row['src']}\n"
        f"S2: {df_row['pred']}\n"
        "Only provide the rating as a number between 1 to 5, with no text or explanation, just the number.\n"
        "Detoxification accuracy rating (on a scale of 1 to 5) = "
    )
    
    prompt = task_definition + instruction
    # logger.info(f"Generated prompt: {prompt}")
    return prompt

def generate_cp_prompt(df_row, language):
    task_definition = ('Text detoxification changes a toxic sentence to a clean sentence while keeping the rest of the content unchanged.\n')

    instruction = (
        f"Please rate the preservation of content between the following {language} toxic source sentence S1 and resulting clean sentence S2. "
        "Use a scale of 1 to 5, where 1 indicates very low content preservation and 5 indicates very high content preservation. "
        "When assessing text detoxification, focus solely on the information presented in the sentences, "
        "disregarding any alterations in toxicity resulting from the detoxification process.\n"
        f"S1: {df_row['src']}\n"
        f"S2: {df_row['pred']}\n"
        "Please provide only the rating as a number between 1 to 5, without any text or explanation, just the number.\n"
        "Content Preservation rating (on a scale of 1 to 5) = "
    )
    
    prompt = task_definition + instruction
    # logger.info(f"Generated prompt: {prompt}")
    return prompt

def generate_fluency_prompt(df_row, language):

    instruction = (
        f"Please rate the fluency of the following {language} sentence S on a scale of 1 to 5, where 1 represents poor fluency,\n"
        "and 5 represents excellent fluency.\n"
        f"S: {df_row['pred']}\n"
        "Only provide the rating as a number between 1 to 5, with no text or explanation, just the number.\n"
        "Fluency rating (on a scale of 1 to 5) = "
    )
    
    prompt = instruction
    # logger.info(f"Generated prompt: {prompt}")
    return prompt

def execute_gpt4(client, model_name, content):
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

    rating = response.choices[0].message.content.strip()
    
    # logger.info(f"Rating: {rating}")
        
    return rating

def process_csv_file(client, args):
    input_csv_file = args.input_csv_file
    language = args.language
    model_name = args.model_name
    
    logger.info(f"Processing input CSV file: {input_csv_file}")
    df = pd.read_csv(input_csv_file)
    
    # For debuging purpose
    # df = df.iloc[0:1]
    
    new_rows = []
    for index, row in df.iterrows():
        acc_content = generate_acc_prompt(row, language)
        # logger.info(f"Generated ACC prompt: {acc_content}")
        cp_content = generate_cp_prompt(row, language)
        # logger.info(f"Generated CP prompt: {cp_content}")
        fluency_content = generate_fluency_prompt(row, language)
        # logger.info(f"Generated Fluency prompt: {fluency_content}")
        
        try:
            acc_rating = execute_gpt4(client, model_name, acc_content)
        except Exception as e:
            acc_rating = 0
            logger.error(f"Failed to rate style transfer accuracy for: {row}")
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())

        try:
            cp_rating = execute_gpt4(client, model_name, cp_content)
        except Exception as e:
            cp_rating = 0
            logger.error(f"Failed to rate content preservation for: {row}")
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())

        try:
            fluency_rating = execute_gpt4(client, model_name, fluency_content)
        except Exception as e:
            fluency_rating = 0
            logger.error(f"Failed to rate fluency for: {row}")
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())

        new_row = row.copy()
        new_row['style_accuracy'] = acc_rating
        new_row['content_preservation'] = cp_rating
        new_row['fluency'] = fluency_rating
        new_rows.append(new_row)

    output_df = pd.DataFrame(new_rows)
    
    output_csv_file = input_csv_file.replace('human', 'gpt4')
    logger.info(f"Writing output to CSV file: {output_csv_file}")
    output_df.to_csv(output_csv_file, index=False, encoding='utf-8')

def main(args):    
    client = OpenAI(api_key='')
    process_csv_file(client, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style Transfer Evaluation using GPT-4")
    parser.add_argument('--model_name', type=str, required=True, help='Pretrained LLM name')
    parser.add_argument('--input_csv_file', required=True, help="Path to the input CSV file")
    parser.add_argument('--language', required=True, help="Language of the input sentences")
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

    logger.info(f"Time taken: {time_taken_formatted}")

    logger.info('=' * 50 + '\n')