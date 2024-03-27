from rank_bm25 import BM25Okapi
from cleantext import clean
import re
import string
from urllib.parse import unquote
import unicodedata

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

def bm25_reranking(rows, query) :

    tokenized_corpus = [ tokenize(clean_text('\t'.join(row))) for row in rows]
    corpus_ids = list(range(len(rows)))

    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenize(query)

    reranking = bm25.get_top_n(tokenized_query, corpus_ids, n=len(rows))

    return reranking

