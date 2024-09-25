import argparse
import torch
import random
import pandas as pd
import numpy as np
import os
import traceback
import socket
import evaluate
from my_logger import logger
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TSTEvaluator:
    def __init__(self, task, lang, output_file, methodology):
        self.task = task
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

    def sentiment_accuracy(self, text, trg_label, lang):
        try:
            cls_model_name = "xlm-roberta-base"
            cls_tokenizer = AutoTokenizer.from_pretrained(cls_model_name)
            cls_model = AutoModelForSequenceClassification.from_pretrained(cls_model_name,
                                                                           num_labels=2,
                                                                           output_attentions=False,
                                                                           output_hidden_states=False)
            cls_model.to(self.device)
            cls_model.load_state_dict(torch.load(f'../classifier/models/{lang}_best.model'))

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
            logger.error(f"Error in sentiment_accuracy: {e}")
            logger.error(traceback.format_exc())
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
            logger.error(f"Error in similarity: {e}")
            logger.error(traceback.format_exc())
            return None

    def bleu_score(self, pred, ref):
        try:
            bleu = evaluate.load("bleu")
            ref_bleu = list()
            for idx, text in enumerate(ref):
                ref_bleu.append([text])
            return bleu.compute(predictions=pred, references=ref_bleu, max_order=4)['bleu']
        except Exception as e:
            logger.error(f"Error in bleu_score: {e}")
            logger.error(traceback.format_exc())
            return None

    def calculate_mgpt_fluency(self, sentence, tokenizer, model):
        tokenize_input = tokenizer.encode(sentence)
        tensor_input = torch.tensor([tokenize_input]).to(self.device)

        with torch.no_grad():
            loss = model(tensor_input, labels=tensor_input)[0]
            ppl = np.exp(loss.detach().cpu().numpy())
            return ppl

    def mgpt_fluency(self, src_sentences, trg_sentences, pred_sentences):
        try:
            model_name = 'ai-forever/mGPT'
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
            model.to(self.device)
            model.eval()

            pred_ppl = []
            pred_num_skipped_sentences = 0

            for idx, (src_sent, trg_sent, pred_sent) in enumerate(zip(src_sentences, trg_sentences, pred_sentences)):
                try:
                    pred_ppl.append(self.calculate_mgpt_fluency(pred_sent, tokenizer, model))
                except Exception as e:
                    logger.error(f"Failed in mgpt fluency calculation for pred sentence. Index: {idx}, src: {src_sent}, trg: {trg_sent}, pred: {pred_sent}, Error: {e}")
                    pred_num_skipped_sentences += 1

            if pred_num_skipped_sentences > 0:
                logger.error(f"{pred_num_skipped_sentences} pred sentences failed in fluency calculation.")

            pred_fluency = sum(pred_ppl) / len(pred_ppl) if pred_ppl else None

            return pred_fluency
        except Exception as e:
            logger.error(f"Unexpected error in mgpt fluency: {e}")
            return None

    def evaluate(self):
        try:
            output_df = pd.read_csv(self.output_file)

            # output_df.fillna('', inplace=True)
            output_df.fillna(' ', inplace=True)
            # output_df['pred'].fillna(output_df['src'], inplace=True)
            # output_df.loc[output_df['pred'].str.isspace(), 'pred'] = output_df['src']
            
            if self.task == 'pos_to_neg':
                target_label = 0
            else:
                target_label = 1

            accuracy = self.sentiment_accuracy(output_df['pred'].to_list(), [target_label] * len(output_df), self.lang)
            similarity_score = self.similarity(output_df['pred'].to_list(), output_df['trg'].to_list(), self.lang)
            bleu = self.bleu_score(output_df['pred'].to_list(), output_df['trg'].to_list())
            fluency = self.mgpt_fluency(output_df['src'].to_list(), output_df['trg'].to_list(), output_df['pred'].to_list())

            return accuracy, similarity_score, bleu, fluency
        except Exception as e:
            logger.error(f"Error in evaluate: {e}")
            return None, None, None, None
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TST Evaluator")
    parser.add_argument("--task", type=str, required=True, help="Task (pos_to_neg or neg_to_pos)")
    parser.add_argument("--lang", type=str, required=True, help="Language")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--methodology", type=str, required=True, help="Methodology")
    args = parser.parse_args()

    logger.info("Command-line arguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device {device}')    

    if torch.cuda.is_available():
        logger.info(f'GPU machine name: {torch.cuda.get_device_name(0)}')
        logger.info(f'SSH machine name: {socket.gethostname()}')
    
    tst_evaluator = TSTEvaluator(args.task, args.lang, args.output_file, args.methodology)
    tst_evaluator.set_seed(53)
    accuracy, similarity_score, bleu, fluency = tst_evaluator.evaluate()

    accuracy_text = f", Sentiment Accuracy: {accuracy:.2f}" if accuracy is not None else ", Sentiment Accuracy: None"
    similarity_score_text = f", Similarity: {similarity_score:.2f}" if similarity_score is not None else ", Similarity: None"
    bleu_text = f", Bleu Score: {bleu:.2f}" if bleu is not None else ", Bleu Score: None"
    fluency_text = f", Fluency: {fluency:.2f}" if fluency is not None else ", Fluency: None"
    
    logger.info(f"Language: {args.lang}, methodology: {args.methodology}, task: {args.task} {accuracy_text}{similarity_score_text}{bleu_text},{fluency_text}")

    logger.info('=' * 50 + '\n')