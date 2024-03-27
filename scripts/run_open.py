import os

ROOT_DIR = os.path.join(os.path.dirname(__file__))

# Disable the TOKENIZERS_PARALLELISM
TOKENIZER_FALSE = "export TOKENIZERS_PARALLELISM=false\n"

engine = 'gpt-3.5-turbo-0613' #'gpt-3.5-turbo'
dataset = 'open_wikitq'

# Multi BM25 table setting top 10 one-by-one
for dataset_chunk_idx in range(20) :
    for retrieval_idx in range(10) :
        print(f'One by one index {retrieval_idx}')
        os.system(fr"""{TOKENIZER_FALSE}python program_main.py --dataset open_wikitq \
        --dataset_split validation \
        --dataset_sub 128 \
        --dataset_chunk_idx {dataset_chunk_idx} \
        --prompt_file templates/prompts/wikitq_ltm.txt \
        --prompt_style create_table_select_3 \
        --generate_type sqlite \
        --n_parallel_prompts 1 \
        --temperature 0.4 \
        --sampling_n 1 \
        --save_dir results/results-opentab-rerank-newreader-ltm-bm25-t20-one_by_one-{engine}/chunk-{dataset_chunk_idx}/idx-{retrieval_idx}  \
        --n_processes 4 \
        --engine {engine} \
        --n_shots 2 \
        --openai_api_key OPENAI_API_KEY \
        --bm25_rerank_rows \
        --ltm \
        --multi_table_type bm25 \
        --multi_table_count 20 \
        --gold_table --one_by_one_idx {retrieval_idx} \
        -v""")