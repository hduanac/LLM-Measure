import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'

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
from numpy.linalg import norm
import matplotlib.pyplot as plt
from  datasets import load_dataset
from readability import Readability
import statsmodels.formula.api as sm
from sklearn.decomposition import PCA
from nltk.tokenize import sent_tokenize
from dataclasses import dataclass, field
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments


# Identify concept vector.
def identify_concept_vector(probing_size, 
                            input_texts, 
                            target_layer, 
                            system_message_pos, 
                            system_message_neg, 
                            pos_preference, 
                            neg_preference, 
                            inst_rephrase,
                           ):
    print(f"Identify concept vector for layer-{target_layer}.")
    print(f"Probing size: {probing_size}.")
    
    diff_embeddings = []
    for trial in range(probing_size):
        print(f"Probing {trial+1}/{probing_size}...")
        input_text_all = input_texts[trial].split()
        embeddings_1 = []
        embeddings_2 = []
        for j in range(0, len(input_text_all), 1000):
            input_text = ' '.join(input_text_all[j:j+1000])
            prompt_pos = f"What characteristics regarding the content of the following customer review about hotels (delimited by triple quotes), do you think, may make the review {pos_preference}? Review text: ```{input_text}```"
            prompt_neg = f"What characteristics regarding the content of the following customer review about hotels (delimited by triple quotes), do you think, may make the review {neg_preference}? Review text: ```{input_text}```"     
            cue_pos = f"The characteristics making the review {pos_preference} include"
            cue_neg = f"The characteristics making the review {neg_preference} include"
            
            if inst_rephrase:
                prompt_pos = f"What aspects of the content in the following customer hotel review (enclosed in triple quotes) might contribute to its complexity? Customer review: ```{input_text}```"
                prompt_neg = f"What aspects of the content in the following customer hotel review (enclosed in triple quotes) might contribute to its conciseness? Customer review: ```{input_text}```"    

            template_pos = f"[INST] <<SYS>> {system_message_pos} <</SYS>>  {prompt_pos} [/INST] {cue_pos}"
            template_neg = f"[INST] <<SYS>> {system_message_neg} <</SYS>>  {prompt_neg} [/INST] {cue_neg}"
            
            # POSITIVE.
            with torch.no_grad():
                tokenized = tokenizer(template_pos, return_tensors = 'pt')
                output = model(input_ids = tokenized['input_ids'],
                               attention_mask = tokenized['attention_mask'],
                               output_attentions = False,
                               output_hidden_states = True,
                              ).hidden_states[target_layer][0][-1].detach().cpu().numpy()
                torch.cuda.empty_cache()
                
            embedding_1 = output.copy()
            embeddings_1.append(embedding_1)

            # NEGATIVE.
            with torch.no_grad():
                tokenized = tokenizer(template_neg, return_tensors = 'pt')
                output = model(input_ids = tokenized['input_ids'],
                               attention_mask = tokenized['attention_mask'],
                               output_attentions = False,
                               output_hidden_states = True,
                              ).hidden_states[target_layer][0][-1].detach().cpu().numpy()
                torch.cuda.empty_cache()
            embedding_2 = output.copy() 
            embeddings_2.append(embedding_2)
        
        diff_embeddings.append((-1)**trial * np.mean(np.array(embeddings_1),axis=0) - np.mean(np.array(embeddings_2),axis=0))

    # PCA.   
    scaler = StandardScaler(copy=True, 
                            with_mean=True, 
                            with_std=True,
                           )
    diff_embeddings_scaled = scaler.fit_transform(diff_embeddings)
    pca = PCA(n_components=6, 
              svd_solver='full',
             )
    pc = pca.fit(diff_embeddings_scaled)
    print(f"Obtained the first principal direction: {pc.components_[0,:]}.")
    return pc, scaler


def do_inference(pc, 
                 scaler, 
                 input_texts, 
                 target_layer, 
                 system_message, 
                 prompt, 
                 cue,
                 ):
    print(f"Inference size: {len(input_texts)}")
    embeddings = []
    for i in range(len(input_texts)): 
        if i%50 == 0:
            print(f"{i}/{len(input_texts)}")        
        input_text = input_texts[i].split()
        outputs = []
        for j in range(0, len(input_text), 1000):
            input_text_single = ' '.join(input_text[j:j+1000])
            prompt_augmented = prompt + f" Review text: ```{input_text_single}```"
            template = f"[INST] <<SYS>> {system_message} <</SYS>>  {prompt_augmented} [/INST] {cue}"

            
            with torch.no_grad():
                tokenized = tokenizer(template, return_tensors = 'pt')
                output = model(input_ids = tokenized['input_ids'],
                               attention_mask = tokenized['attention_mask'],
                               output_attentions = False,
                               output_hidden_states = True,
                               ).hidden_states[target_layer][0][-1].detach().cpu().numpy()
                torch.cuda.empty_cache()
            outputs.append(output)
        embeddings.append(np.mean(np.array(outputs),axis=0))
    
    embeddings = np.array(embeddings)
    embeddings = scaler.transform(embeddings)
    print(f"Done.")
    return embeddings

    
def measure(all_texts, 
         probing_texts, 
         pos_preference, 
         neg_preference, 
         desc_dim, 
         pos_desc, 
         neg_desc, 
         role,
         model_name = "Llama-2-13b-chat-hf",
         out_name = '', 
         probing_size = 64, 
         inference = True, 
         first=True, 
         inst_rephrase=False,
        ):
    
    ####################################  Load model  ############################################
    print(f"Loading " + model_name)
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/" + model_name, 
                                              token="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",
                                             )
    global model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/" + model_name,
                                                 token="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",
                                                 device_map="auto",
                                                 torch_dtype=torch.float16)
    model.eval()
    target_layer = 40
    print(f"Model loaded.")

        
    #######################################  prompts ##############################################

    system_message_pos = f"Pretend you are a {role}. Your task is to analyze the {desc_dim} of a customer review about hotels. Particularly, you should focus on the characteristics regarding the review content that may make the review {pos_preference}. More formally, a {pos_preference} review is characterized by {pos_desc}"
    
    system_message_neg = f"Pretend you are a {role}. Your task is to analyze the {desc_dim} of a customer review about hotels. Particularly, you should focus on the characteristics regarding the review content that may make the review {neg_preference}. More formally, a {neg_preference} review is characterized by {neg_desc}"    
    
    
    if inst_rephrase:
        system_message_pos = f"Imagine you are a {role} tasked with analyzing the {desc_dim} of a hotel customer review. Specifically, your focus should be on identifying features of the review that contribute to its complexity. In more precise terms, reviews are considered complex if they are {pos_desc}"
        system_message_neg = f"Imagine you are a {role} tasked with analyzing the {desc_dim} of a hotel customer review. Specifically, your focus should be on identifying features of the review that contribute to its conciseness. In more precise terms, reviews are considered concise if they are {neg_desc}"
    

        
    ######################################## Probing  ##############################################     
    if first:
        input_texts = probing_texts[:probing_size]
    else:
        indices = random.sample(range(len(probing_texts)), probing_size)
        input_texts = [i for n,i in enumerate(probing_texts) if n in indices]

    pc, scaler = identify_concept_vector(probing_size, 
                                         input_texts, 
                                         target_layer, 
                                         system_message_pos, 
                                         system_message_neg, 
                                         pos_preference, 
                                         neg_preference, 
                                         inst_rephrase,
                                        )
#     with open('./results/' + out_name + f"_{model_name}" + f"_probingSize-{probing_size}" + '_pc.pkl', 'wb') as file: 
#          pickle.dump(pc, file) 
#     with open('./results/' + out_name + f"_{model_name}" + f"_probingSize-{probing_size}" + '_scale.pkl', 'wb') as file: 
#          pickle.dump(scaler, file) 
 
    
    ########################################### Inference ##############################################     
    if inference:    
        system_message = f"Pretend you are a {role}. Your task is to analyze the {desc_dim} of a customer review about hotels. Particularly, you should focus on the characteristics regarding the review content that may make the review either {pos_preference} or {neg_preference}. More formally, a {pos_preference} review is characterized by {pos_desc}. A {neg_preference} review is characterized by {neg_desc}"
        prompt = f"What characteristics regarding the content of the following customer review about hotels (delimited by triple quotes), do you think, may contribute to the {desc_dim} of this review, making it either {pos_preference} or {neg_preference}?"
        cue = f"The characteristics contributing to its {desc_dim} include"
        
        if inst_rephrase:
            system_message = f"Imagine you are a {role} tasked with analyzing the {desc_dim} of a hotel customer review. Specifically, your focus should be on identifying features of the review that contribute to its complexity or conciseness. In more precise terms, reviews are considered {pos_preference} if they are {pos_desc}, while they are considered {neg_preference} if they are {neg_desc}"
            prompt = f"What aspects of the content in the following customer hotel review (enclosed in triple quotes) might contribute to its complexity or conciseness?"
                      
        embeddings = do_inference(pc, 
                               scaler, 
                               all_texts, 
                               target_layer, 
                               system_message, 
                               prompt, 
                               cue, 
                              )
#         with open('./results/' +  out_name + f"_{model_name}" + f"_probingSize-{probing_size}" + '_emd.pkl', 'wb') as file: 
#             pickle.dump(embeddings, file) 

 
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-M','--main_version', type=bool, default=True, help = 'Main version?')
    parser.add_argument('-D','--rephrase_definition', type=bool, default=False, help = 'Rephrase definitions?')
    parser.add_argument('-I','--rephrase_instruction', type=bool, default=False, help = 'Rephrase the instruction?')
    parser.add_argument('-P','--probing_128', type=bool, default=False, help = 'Use 128 probing samples?')

    args = vars(parser.parse_args())
    rephrase_definition = args['rephrase_definition']
    rephrase_instruction = args['rephrase_instruction']
    probing_128 = args['probing_128']
    main_version = args['main_version']
    
    print("Main version: ", main_version)
    print("Rephrase definition: ", rephrase_definition)
    print("Rephrase instruction: ", rephrase_instruction)    
    print("Use 128 probing samples: ", probing_128)
    
    
    
    # Load review data.
    with open('./data/review_total.pkl', 'rb') as file: 
        total_df = pickle.load(file) 
        all_texts = total_df['review'].tolist()
        probing_texts = total_df['review'].tolist()

    pos_preference = "complex"
    neg_preference = "concise"
    desc_dim = "content complexity"
    pos_desc = "comprehensive and detailed reviews that encompass a wide range of aspects, incorporate diverse topics, and provide nuanced insights into the hotel experience."
    neg_desc = "simplistic and superficial reviews that lack depth, focus on narrow topics, and offer limited analysis of the hotel experience."
    role = "linguist"
    
    
    
    ########################################### Main ##############################################
    if main_version:
        measure(all_texts,
             probing_texts,
             pos_preference, 
             neg_preference, 
             desc_dim, 
             pos_desc, 
             neg_desc, 
             role,  
             model_name = "Llama-2-13b-chat-hf",
             out_name = "original-signed", 
             probing_size = 64, 
             inference = True,
             first=True,
            )

    ##################################### Definition Rephrased #####################################
    if rephrase_definition:
        pos_desc_rephrased = "extensive and intricate assessments that cover a broad spectrum of elements, incorporate varied subjects, and offer subtle perspectives on the hotel stay."
        neg_desc_rephrased = "straightforward and surface-level evaluations that lack profundity, concentrate on specific subjects, and provide restricted analysis of the hotel stay."
        measure(all_texts,
             probing_texts,
             pos_preference, 
             neg_preference, 
             desc_dim, 
             pos_desc_rephrased, 
             neg_desc_rephrased, 
             role,  
             model_name = "Llama-2-13b-chat-hf",
             out_name = "definitionRephrased-signed", 
             probing_size = 64, 
             inference = True,
             first=True,
            )


    ##################################### Instruction Rephrased #####################################
    if rephrase_instruction:
        measure(all_texts,
             probing_texts,
             pos_preference, 
             neg_preference, 
             desc_dim, 
             pos_desc, 
             neg_desc, 
             role,  
             model_name = "Llama-2-13b-chat-hf",
             out_name = "instructionRephrased-signed", 
             probing_size = 64, 
             inference = True,
             first=True,
             inst_rephrase=True,
            )

    ##################################### Probing size: 128 #####################################
    if probing_128:
        measure(all_texts,
             probing_texts,
             pos_preference, 
             neg_preference, 
             desc_dim, 
             pos_desc, 
             neg_desc, 
             role,  
             model_name = "Llama-2-13b-chat-hf",
             out_name = "original-signed", 
             probing_size = 128, 
             inference = True,
             first=True,
            )   
    
        
if __name__ == '__main__':
    main()
    



            
            
            
    