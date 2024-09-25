import argparse
import torch
import random
import pandas as pd
import numpy as np
import os
import traceback
import socket
import evaluate
# import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel

#logging.basicConfig(level=#logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TSTEvaluator:
    def __init__(self, lang, output_file, methodology):
        self.lang = lang
        self.output_file = output_file
        self.methodology = methodology
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_seed(self, seed_value):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    def style_accuracy(self, text, trg_label, lang):
        try:
            cls_model_name = "xlm-roberta-base"
            cls_tokenizer = AutoTokenizer.from_pretrained(cls_model_name)
            cls_model = AutoModelForSequenceClassification.from_pretrained(cls_model_name,
                                                                           num_labels=2,
                                                                           output_attentions=False,
                                                                           output_hidden_states=False)
            cls_model.to(self.device)
            cls_model.load_state_dict(torch.load(f'../classifier/models/toxicity/{lang}_best.model'))

            inputs = cls_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = cls_model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            predicted_labels = logits.argmax(dim=1).cpu().numpy()

            accuracy = accuracy_score(trg_label, predicted_labels)

            return accuracy
        except Exception as e:
            #logging.error(f"Error in style_accuracy: {e}")
            #logging.error(traceback.format_exc())
            return None

    def similarity(self, text1, text2, lang):
        try:
            sim_model = SentenceTransformer('sentence-transformers/LaBSE')
            sim_model.to(self.device)

            sim_scores = list()
            for idx, _ in enumerate(text1):
                sentence_embedding1 = sim_model.encode(text1[idx])
                sentence_embedding2 = sim_model.encode(text2[idx])
                sim_score = cosine_similarity([sentence_embedding1], [sentence_embedding2])
                sim_scores.append(sim_score[0][0])

            return sum(sim_scores) / len(sim_scores)
        except Exception as e:
            #logging.error(f"Error in similarity: {e}")
            #logging.error(traceback.format_exc())
            return None

    def bleu_score(self, pred, ref):
        try:
            bleu = evaluate.load("bleu")
            ref_bleu = list()
            for idx, text in enumerate(ref):
                ref_bleu.append([text])
            return bleu.compute(predictions=pred, references=ref_bleu, max_order=4)['bleu']
        except Exception as e:
            #logging.error(f"Error in bleu_score: {e}")
            #logging.error(traceback.format_exc())
            return None

    def evaluate(self):
        try:
            output_df = pd.read_csv(self.output_file)

            output_df.fillna('', inplace=True)
            # output_df['pred'].fillna(output_df['src'], inplace=True)
            # output_df.loc[output_df['pred'].str.isspace(), 'pred'] = output_df['src']
            
            target_label = 1

            accuracy = self.style_accuracy(output_df['pred'].to_list(), [target_label] * len(output_df), self.lang)
            similarity_score = self.similarity(output_df['pred'].to_list(), output_df['trg'].to_list(), self.lang)
            bleu = self.bleu_score(output_df['pred'].to_list(), output_df['trg'].to_list())

            return accuracy, similarity_score, bleu
        except Exception as e:
            #logging.error(f"Error in evaluate: {e}")
            return None, None, None
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TST Evaluator")
    parser.add_argument("--lang", type=str, required=True, help="Language")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--methodology", type=str, required=True, help="Methodology")
    args = parser.parse_args()

    #logging.info("Command-line arguments:")
    # for arg, value in vars(args).items():
        #logging.info(f"{arg}: {value}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #logging.info(f'Device {device}')    

    # if torch.cuda.is_available():
        #logging.info(f'GPU machine name: {torch.cuda.get_device_name(0)}')
        #logging.info(f'SSH machine name: {socket.gethostname()}')
    
    tst_evaluator = TSTEvaluator(args.lang, args.output_file, args.methodology)
    tst_evaluator.set_seed(53)
    accuracy, similarity_score, bleu = tst_evaluator.evaluate()

    # accuracy_text = f", Style Accuracy: {accuracy:.4f}" if accuracy is not None else ", Style Accuracy: None"
    # similarity_score_text = f", Similarity: {similarity_score:.4f}" if similarity_score is not None else ", Similarity: None"
    # bleu_text = f", Bleu Score: {bleu:.4f}" if bleu is not None else ", Bleu Score: None"
    
    #logging.info(f"Language: {args.lang}, methodology: {args.methodology} {accuracy_text}{similarity_score_text}{bleu_text}")

    #logging.info('=' * 50 + '\n')