import json
from rank_bm25 import BM25Okapi
from cleantext import clean
import re
import string
from urllib.parse import unquote
import unicodedata
import os
import time
import pdb
import pandas as pd
from tqdm import tqdm

from dataloader import OpenWikiTable

pt = re.compile( r"\[\[.*?\|(.*?)]]" )
def clean_text(text):
    text = re.sub(pt, r"\1", text)
    text = unquote(text)
    text = unicodedata.normalize('NFD', text)
    text = clean(
        text.strip(),
        fix_unicode=True,               # fix various unicode errors
        to_ascii=False,                  # transliterate to closest ASCII representation
        lower=False,                     # lowercase text
        no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
        no_urls=True,                  # replace all URLs with a special token
        no_emails=False,                # replace all email addresses with a special token
        no_phone_numbers=False,         # replace all phone numbers with a special token
        no_numbers=False,               # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct=False,                 # remove punctuations
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUMBER>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        lang="en"                       # set to 'de' for German special handling
    )
    return text

import nltk
nltk.download('stopwords')
nltk.download('punkt')
ignored_words = set(nltk.corpus.stopwords.words('english'))
punct_set = set(['.', "''", '``', ',', '(', ')'] + list(string.punctuation))
ignored_words = ignored_words.union(punct_set)
import nltk.stem
stemmizer = nltk.stem.SnowballStemmer('english')

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [stemmizer.stem(tk) for tk in tokens if tk.lower() not in ignored_words]
    return tokens

def my_clean_table(table) :
    return re.sub(r'(\[Title\]|\[Section title\]|\[Caption\]|\[Table name\]|\[Header\]|\[Rows\]|\[Row\]|\[sep\])+', ' ', table)

tables = pd.read_json(os.path.join('data',"splitted_tables.json"))

train_dataset = OpenWikiTable(pd.read_json(os.path.join('data',"train.json")), train = False)
valid_dataset = OpenWikiTable(pd.read_json(os.path.join('data',"valid.json")), train = False)
test_dataset = OpenWikiTable(pd.read_json(os.path.join('data',"test.json")), train = False)

val_claims, val_labels, val_table_ids = [], [], []
for val_sample in valid_dataset :
    val_claims.append(val_sample['question'])
    val_labels.append(val_sample['answer'])
    val_table_ids.append(val_sample['hard_positive_idx']+val_sample['positive_idx'])


if not os.path.exists('data/validation_preds_50.json') :

    start = time.time()

    tokenized_corpus = [ tokenize(clean_text(my_clean_table(table))) for table in tables["flattened"]]


    corpus_ids = list(range(len(tables["flattened"])))

    bm25 = BM25Okapi(tokenized_corpus)

    print(f'BM25 constructing:{time.time()-start:.1f}s')

    start = time.time()
    val_preds = []
    for val_claim in tqdm(val_claims) :
        tokenized_query = tokenize(val_claim)
        res_tables = bm25.get_top_n(tokenized_query, corpus_ids, n=50)
        val_preds.append(res_tables)

    with open('data/validation_preds_50.json', 'w') as file:
        json.dump(val_preds, file)   
    print(f'BM25 searching:{time.time()-start:.1f}s, avg:{(time.time()-start)/len(val_claims)}s per sample')


with open('data/validation_preds_50.json', 'r') as file:
    val_preds = json.load(file)
acc = [0, 0, 0, 0, 0, 0]

def recall_eval(pred, gt, acc):
    if all(elem in pred[:1] for elem in gt):
        acc[0]+=1
    if all(elem in pred[:2] for elem in gt):
        acc[1]+=1
    if all(elem in pred[:5] for elem in gt):
        acc[2]+=1
    if all(elem in pred[:10] for elem in gt):
        acc[3]+=1
    if all(elem in pred[:20] for elem in gt):
        acc[4]+=1
    if all(elem in pred[:50] for elem in gt):
        acc[5]+=1

for i in range(len(val_claims)) :
    # ground truth 1-indexed, prediction 0-indexed
    pred = val_preds[i]

    gt = [ val_table_id-1 for val_table_id in val_table_ids[i]]
    recall_eval(pred, gt, acc)

print(f'recall@1:{acc[0]/len(val_claims):.3f}, \
recall@2:{acc[1]/len(val_claims):.3f}, \
recall@5:{acc[2]/len(val_claims):.3f}, \
recall@10:{acc[3]/len(val_claims):.3f} \
recall@20:{acc[4]/len(val_claims):.3f} \
recall@50:{acc[5]/len(val_claims):.3f}' )

