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
    
    
    
    df = pd.read_csv('./data/pc_data.csv',index_col=0)
    all_texts = df['sentence'].tolist()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", token="")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", token="",
                                                 device_map="auto",
                                                 torch_dtype=torch.float16,
                                                )
    model.eval()

    
    pos_preference = "<hawkish>"
    neg_preference = "<dovish>"
    desc_dim = "<monetary policy stance>"
    role = "financial analyst"

    
    
    if rephrase_definition:
        pos_desc = "Greater restraint in expanding money supply measures like M0, M1, and M2; increased concern over medium-term inflation risks from excess money supply; stringent controls on the monetary base to protect purchasing power; higher short-term borrowing costs; reduced asset valuations, including stocks and housing."
        neg_desc = "Greater willingness to expand the money supply rapidly through measures like quantitative easing; reduced concern about short-run inflation risks from excessive money supply growth; stimulation of aggregate demand, bank lending, and overall economic activity by providing ample liquidity in the financial system."
    else:
        
        pos_desc = "more restraint in growing various measures of money supply like M0, M1, M2, more concerned about risks of excess money supply expansion fueling inflation in the medium-term, maintaining strong controls on the monetary base to safeguard purchasing power, push up short-term borrowing costs and lowers asset valuations like stocks or housing."
        neg_desc = "more willing to grow the money supply at a robust pace through measures like quantitative easing, less concerned about risks of too much money supply expansion fueling inflation in the short-run, giving a boost to aggregate demand, bank lending, and overall economic activity by ensuring ample liquidity in the financial system."
    

    output_string_all = []
    for n, input_text_single in enumerate(all_texts):
        
        if rephrase_instruction:
            system_message = f"Imagine you are a {role} tasked with assessing the {desc_dim} of a sentence from the Federal Open Market Committee (FOMC) document. {pos_preference} sentence is characterized by {pos_desc}, whereas {neg_preference} sentence is defined as {neg_desc}"
            prompt_augmented = f" Sentence: ```{input_text_single}```"
            cue = f"How {pos_preference} the sentence is? Assign a score between 0 and 100, where 0 indicates very {neg_preference} and 100 indicates very {pos_preference}. The higher the score, the greater the {pos_preference} of the sentence. Ouput only the score without any explanation. ###SCORE:"
        else:
            system_message = f"Pretend you are a {role}. Your task is to analyze the {desc_dim} of a text snippet from the Federal Open Market Committee (FOMC) document. Particularly, you should focus on what traits in the text snippet suggesting that this text is being {pos_preference} or {neg_preference}? {pos_preference} is defined by {pos_desc}. {neg_preference} is defined by {neg_desc}"
            prompt_augmented = f" Here’s the text snippet: ```{input_text_single}```"
            cue = f"How {pos_preference} or {neg_preference} is this text snippet of the {desc_dim}? Provide your response as a score between 0 and 100 where 0 means {neg_preference} and 100 means {pos_preference}. The higher the score, the more {pos_preference} the text is. Directly output the score without giving the explanation. ###SCORE:"
        
        total_prompt = f"[INST] <<SYS>> {system_message} <</SYS>> {prompt_augmented} [/INST] {cue}"
        
        flag = True
        while flag:
            with torch.no_grad():
                tokenized = tokenizer(total_prompt,return_tensors = 'pt')

                outputs = model.generate(input_ids=tokenized["input_ids"].to("cuda"),
                                 #do_sample=False,
                                 temperature=1,
                                 max_length = tokenized["input_ids"][0].shape[0] + 3)
                outputs_string = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
                torch.cuda.empty_cache()

            single = outputs_string.split('###SCORE:')[1]
            single_int = []
            for k in single:
                if k.isdigit():
                    single_int.append(k)
            if len(single_int) != 0:
                output_string_all.append(int(''.join(single_int)))
                #flag = False
                break
        
        if n % 100 == 0:
            print(n)
    # Save.
    #if rephrase_definition:
    #    np.save('./results/rephrase_def_prompt_output_score_temp1.npy', np.array(output_string_all_int))
    #    print("Saved as rephrase_def_prompt_output_score_temp1.npy")
    #elif rephrase_instruction:
    #    np.save('./results/rephrase_inst_prompt_output_score_temp1.npy', np.array(output_string_all_int))
    #    print("Saved as rephrase_inst_prompt_output_score_temp1.npy")
    #else:
    #    np.save('./results/original_prompt_output_score_temp1.npy', np.array(output_string_all_int))
    #    print("Saved as original_prompt_output_score_temp1.npy")
        
        
        

if __name__=='__main__':
    main()





    
    
