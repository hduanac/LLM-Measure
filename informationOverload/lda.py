import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import re
import nltk
import json
import rdata
import torch
import random
import gensim
import pickle
import Stemmer
import datasets
import numpy as np
import transformers
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from pprint import pprint
import nltk.stem as stemmer
from typing import Optional
import statsmodels.api as sm
from numpy.linalg import norm
import matplotlib.pyplot as plt
from llm_measure import measure
import gensim.corpora as corpora
from nltk.corpus import stopwords
from datasets import load_dataset
from readability import Readability
from nltk.stem import PorterStemmer
import statsmodels.formula.api as sm
from sklearn.decomposition import PCA
from nltk.tokenize import sent_tokenize
from dataclasses import dataclass, field
from gensim.utils import simple_preprocess
from langdetect import detect, detect_langs
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from transformers import LlamaForCausalLM, LlamaTokenizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
nltk.download('stopwords')

stemmer = Stemmer.Stemmer('en')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: stemmer.stemWords(analyzer(doc))    

def main():
    best_number_topics = 25
    best_learning_decay = 0.1

    with open('./data/review_total.pkl', 'rb') as file:
        total_df = pickle.load(file) 
    total_df.review = total_df.review.str.replace("[^A-Za-z ]", " ")
    
    analyzer = CountVectorizer().build_analyzer()
    vectorizer = StemmedCountVectorizer(stop_words='english', 
                                        min_df=5, 
                                        max_df=0.5,
                                       )
    matrix = vectorizer.fit_transform(total_df.review)
    words_df = pd.DataFrame(matrix.toarray(),
                            columns=vectorizer.get_feature_names(),
                           )
    model = LatentDirichletAllocation(n_components=best_number_topics, 
                                      learning_decay=best_learning_decay,
                                     )
    model.fit(matrix)
    doc_topic_dist = model.transform(matrix)
    entropy = []
    for doc in doc_topic_dist:
        entropy.append(-sum([i*np.log(i) for i in doc]))
    
    np.save('./results/topicEntropy.npy', np.array(entropy))
    print("Saved as topicEntropy.npy.")
        
if __name__ == '__main__':
    main()

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    