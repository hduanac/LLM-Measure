import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import re
import json
import rdata
import torch
import random
import pickle
import datasets
import argparse
import numpy as np
import transformers
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from typing import Optional
import statsmodels.api as sm
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from  datasets import load_dataset
from readability import Readability
from sklearn.decomposition import PCA
from nltk.tokenize import sent_tokenize
from scipy.stats.mstats import winsorize
from dataclasses import dataclass, field
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments


def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('-D','--rephrase_definition', type=bool, default=False, help = 'Rephrase definitions?')
    parser.add_argument('-I','--rephrase_instruction', type=bool, default=False, help = 'Rephrase the instruction?')

    args = vars(parser.parse_args())
    rephrase_definition = args['rephrase_definition']
    rephrase_instruction = args['rephrase_instruction']
    
    print("Rephrase definition: ", rephrase_definition)
    print("Rephrase instruction: ", rephrase_instruction)
    
    
    
    with open('./data/review_total.pkl', 'rb') as file: 
        total_df = pickle.load(file) 
    all_texts = total_df['review'].tolist()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", token="")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", token="",
                                                 device_map="auto",
                                                 torch_dtype=torch.float16,
                                                )
    model.eval()



    pos_preference = "complex"
    neg_preference = "concise"
    desc_dim = "content complexity"
    role = "linguist"
    
    if rephrase_definition:
        pos_desc = "extensive and intricate assessments that cover a broad spectrum of elements, incorporate varied subjects, and offer subtle perspectives on the hotel stay"
        neg_desc = "straightforward and surface-level evaluations that lack profundity, concentrate on specific subjects, and provide restricted analysis of the hotel stay"
    else:
        pos_desc = "comprehensive and detailed reviews that encompass a wide range of aspects, incorporate diverse topics, and provide nuanced insights into the hotel experience."
        neg_desc = "simplistic and superficial reviews that lack depth, focus on narrow topics, and offer limited analysis of the hotel experience."
    

    output_string_all = []
    for n, input_text_single in enumerate(all_texts):
        
        if rephrase_instruction:
            system_message = f"Imagine you are a {role} tasked with assessing the {desc_dim} of a hotel customer review. Complex reviews are characterized by {pos_desc}, whereas concise reivews are defined as {neg_desc}"
            prompt_augmented = f" Hotel review: ```{input_text_single}```"
            cue = f"How {pos_preference} the review is? Assign a score between 0 and 100, where 0 indicates very concise and 100 indicates very complex. The higher the score, the greater the complexity of the review. Ouput only the score without any explanation. ###SCORE:"
        else:
            system_message = f"Pretend you are a {role}. Your task is to evaluate the {desc_dim} of a customer review about hotels. By definition, {pos_preference} reviews are {pos_desc}, while {neg_preference} reviews are {neg_desc}"
            prompt_augmented = f" Review text: ```{input_text_single}```"
            cue = f"How {pos_preference} is this review? Provide your response as a score between 0 and 100 where 0 means extremely {neg_preference} and 100 means extremely {pos_preference}. The higher the score, the more complex the review is. Directly output the score without giving the explanation. ###SCORE:"
        
        total_prompt = f"[INST] <<SYS>> {system_message} <</SYS>> {prompt_augmented} [/INST] {cue}"
        with torch.no_grad():
            tokenized = tokenizer(total_prompt,return_tensors = 'pt')
            outputs = model.generate(input_ids=tokenized["input_ids"].to("cuda"),
                                     temperature=1,
                                     max_length = tokenized["input_ids"][0].shape[0] + 3,
                                    )
            output_string = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
            torch.cuda.empty_cache()
        output_string_all.append(output_string.split('###SCORE:')[1])
        if n % 100 == 0:
            print(n)

            
    # Make up.        
    output_string_all_int = []
    count = 0
    for single in output_string_all:
        try:
            output_string_all_int.append(int(single.strip()))
        except:
            single_strip = single.strip()            
            single_int = []
            for k in single_strip:
                if k.isdigit():
                    single_int.append(k)
            if len(single_int) == 0:
                output_string_all_int.append(np.nan)
                count += 1
                continue
            output_string_all_int.append(int(''.join(single_int)))

    while np.nan in output_string_all_int:  
        indices = []
        for idx in range(len(output_string_all_int)):
            if output_string_all_int[idx] is np.nan:
                indices.append(idx)

        texts_makeup = []
        for i in indices:
            texts_makeup.append(all_texts[i])

        output_string_makeup = []
        for n, input_text_single in enumerate(texts_makeup):
            if rephrase_instruction:
                system_message = f"Imagine you are a {role} tasked with assessing the {desc_dim} of a hotel customer review. Complex reviews are characterized by {pos_desc}, whereas concise reivews are defined as {neg_desc}"
                prompt_augmented = f" Hotel review: ```{input_text_single}```"
                cue = f"How {pos_preference} the review is? Assign a score between 0 and 100, where 0 indicates very concise and 100 indicates very complex. The higher the score, the greater the complexity of the review. Ouput only the score without any explanation. ###SCORE:"
            else:
                system_message = f"Pretend you are a {role}. Your task is to evaluate the {desc_dim} of a customer review about hotels. By definition, {pos_preference} reviews are {pos_desc}, while {neg_preference} reviews are {neg_desc}"
                prompt_augmented = f" Review text: ```{input_text_single}```"
                cue = f"How {pos_preference} is this review? Provide your response as a score between 0 and 100 where 0 means extremely {neg_preference} and 100 means extremely {pos_preference}. The higher the score, the more complex the review is. Directly output the score without giving the explanation. ###SCORE:"
                total_prompt = f"[INST] <<SYS>> {system_message} <</SYS>> {prompt_augmented} [/INST] {cue}"

            with torch.no_grad():
                tokenized = tokenizer(total_prompt,return_tensors = 'pt')
                outputs = model.generate(input_ids=tokenized["input_ids"].to("cuda"),
                                         temperature=1,
                                         max_length = tokenized["input_ids"][0].shape[0] + 3,
                                        )
                output_string = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
                torch.cuda.empty_cache()
            output_string_makeup.append(output_string.split('###SCORE:')[1])

        # Cleaning.
        makeup_int = []
        count = 0
        for single in output_string_makeup:
            try:
                makeup_int.append(int(single.strip()))
            except:
                single_strip = single.strip()            
                single_int = []
                for k in single_strip:
                    if k.isdigit():
                        single_int.append(k)
                if len(single_int) == 0:
                    makeup_int.append(np.nan)
                    count += 1
                    continue
                makeup_int.append(int(''.join(single_int)))


        for i in range(len(indices)):
            output_string_all_int[indices[i]] = makeup_int[i]

    # Save.
#     if rephrase_definition:
#         np.save('./results/llmPromptingScoresDefinitionRephrased.npy', np.array(output_string_all_int))
#         print("Saved as llmPromptingScoresDefinitionRephrased.npy")
#     elif rephrase_instruction:
#         np.save('./results/llmPromptingScoresInstructionRephrased.npy', np.array(output_string_all_int))
#         print("Saved as llmPromptingScoresInstructionRephrased.npy")
#     else:
#         np.save('./results/llmPromptingScores.npy', np.array(output_string_all_int))
#         print("Saved as llmPromptingScores.npy")
        
        
        

if __name__=='__main__':
    main()





    
    




























