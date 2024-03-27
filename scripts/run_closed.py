import os

ROOT_DIR = os.path.join(os.path.dirname(__file__))

# Disable the TOKENIZERS_PARALLELISM
TOKENIZER_FALSE = "export TOKENIZERS_PARALLELISM=false\n"

engine = 'gpt-3.5-turbo-0613' #'gpt-3.5-turbo'
dataset = 'open_wikitq' # 'wikitq'

# Open-wiki closed-domain setting
os.system(fr"""{TOKENIZER_FALSE}python program_main.py --dataset open_wikitq \
--dataset_split validation \
--dataset_sub 2048 \
--dataset_chunk_idx 0 \
--prompt_file templates-full/prompts/wikitq_ltm.txt \
--prompt_style create_table_select_3 \
--generate_type sqlite \
--n_parallel_prompts 1 \
--temperature 0.4 \
--sampling_n 1 \
--save_dir results/results-0923-ours-rerank-newreader-ltm-bm25-t20-one_by_one-{engine}/chunk-0/closed  \
--n_processes 4 \
--engine {engine} \
--n_shots 2 \
--openai_api_key OPENAI_API_KEY \
--bm25_rerank_rows \
--ltm \
--gold_table \
-v""")