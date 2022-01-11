#!/usr/bin/env python
# coding: utf-8

import langid
import pandas as pd
from langid.langid import LanguageIdentifier, model
import time
import re
import nltk
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSeq2SeqLM 
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


## Part_1: Language Detection

'''
1.Chinese -- zh
2.Dutch -- nl
3.French -- fr
4.German -- de
5.Korean -- ko
6.Portuguese -- pt
7.Spanish -- es
8.Swedish -- sv
'''
Language_List = ['zh', 'Dutch', 'fr', 'de', 'ko', 'pt', 'es', 'sv']

# define langeuage identifier
lang_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

# define language detection function
def langDetect(input_file_path):
    df = pd.read_excel(input_file_path)
    df = df[Columns]
    df['lang'] = ''
    df['langcode'] = ''
    for index,row in df.iterrows():
        #print(row)
        row['lang'] = lang_identifier.classify(str(row['Description']))
        df.at[index,'langcode'] = row['lang'][0]
    return df

# run language detection function
begin = time.time()
df = langDetect('Inputs')
end = time.time()
print(f"Total runtime of the program is {end - begin}")

# ## Part_2: Machine Translation

# ## iteration2 data loading
df_zh = pd.read_excel('Inputs')

# ### Sentence Split Functions
'''
Sentence Split Function for Chinese and other Asian languages
'''
def sent_split(paragraph):
    for sent in re.findall(u'[^!?。：？\!\?]+[!？?。：\.\!\?]?', paragraph, flags=re.U):
        yield sent


# ### Iteration2_Model: Multi-records sentence splitting

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(torch.cuda.is_available())


# #### Chinese model

# Translation Model for Chinese and other Asian languages
## Dataframe Input Version
def new_translator(src_language, df_repair):
    # Reading data from input file
    df_lang = df_repair
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B", src_lang = src_language)
    # Casting target columns to string for further processing
    col_to_translate = ['Title']
    df_lang[col_to_translate] = df_lang[col_to_translate].astype(str) 
    # Iterating over the data to translate above columns
    for index,item in df_lang.iterrows():
        for col_name in col_to_translate:
            text = sent_split(item[col_name])
            sentence_list = []
            for sentence in text:
                encoded_zh = tokenizer(sentence, return_tensors = "pt").to(device)
                generated_tokens = model.generate(**encoded_zh, max_length = 1000, forced_bos_token_id = tokenizer.get_lang_id("en")).to(device)
                tmpt_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                sentence_list += tmpt_output
            # merge a list of sentences to paragraph     
            df_lang.loc[index, 'translated_' + col_name] = ' '.join(sentence_list)
            
    return df_lang

# ## Text Input Version
# def new_translator(src_language):
#     # Reading data from cleaned sentence_splited list
#     t_list = []
#     tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B", src_lang = src_language)
#     for sentence in cleaned_list:
#         encoded_sentence = tokenizer(sentence, return_tensors = "pt").to(device)
#         generated_tokens = model.generate(**encoded_sentence, max_length = 1000,\
#                                               forced_bos_token_id = tokenizer.get_lang_id("en")).to(device)
#         translated_sentence = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
#         t_list += translated_sentence
    
#     return t_list


# #### Chinese Output

begin = time.time()
df_tmpt = new_translator('zh', df_zh)
end = time.time()
# print(f"Chinese Total runtime of the program is {end - begin}")

# save runtime to runtime_dict
runtime_dict = {}
runtime_dict['Chinese_rt'] = end - begin

df_tmpt.to_excel("Outputs")


# #### Non_Asian Model

# Use nltk for sentence spliting
nltk.download("punkt")

def new_translator_non_Asian(src_language, df_repair):
    # Reading data from input file
    df_lang = df_repair
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B", src_lang = src_language)
    col_to_translate = ['Title']
    df_lang[col_to_translate] = df_lang[col_to_translate].astype(str)
    
    # Iterating over the data to translate above columns
    for index,item in df_lang.iterrows():
        for col_name in col_to_translate:
            text = nltk.tokenize.sent_tokenize(item[col_name])
            sentence_list = []
            for sentence in text:
                encoded_zh = tokenizer(sentence, return_tensors = "pt").to(device)
                generated_tokens = model.generate(**encoded_zh, max_length = 1000,                                                  forced_bos_token_id = tokenizer.get_lang_id("en")).to(device)
                tmpt_output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                sentence_list += tmpt_output
            # merge a list of sentences to paragraph     
            df_lang.loc[index, 'translated_' + col_name] = ' '.join(sentence_list)
            
    return df_lang
