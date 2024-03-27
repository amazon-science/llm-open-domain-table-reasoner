import sys
sys.path.append('../')

import time
import json
import argparse
import os

import platform
import multiprocessing
import torch

from generation.generator import Generator
from worker_func import worker_annotate
from data_loading import wikitable, open_wikitable, feverous

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")

def main():

    assert args.dataset_split == 'validation'

    # Build paths
    args.api_keys_file = os.path.join(ROOT_DIR, args.api_keys_file)
    args.prompt_file = os.path.join(ROOT_DIR, args.prompt_file)
    args.save_dir = os.path.join(ROOT_DIR, args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    start_time = time.time()
    
    # Load dataset
    if args.dataset == 'open_wikitq' :
        dataset = open_wikitable(args)
    elif args.dataset == 'wikitq' :
        assert args.gold_table
        dataset = wikitable(args)
    elif args.dataset == 'feverous' :
        dataset = feverous(args)
    

    # Annotate
    generator = Generator(args, keys=None)
    generate_eids = list(range(len(dataset)))
    generate_eids_group = [[] for _ in range(args.n_processes)]
    for g_eid in generate_eids:
        generate_eids_group[int(g_eid) % args.n_processes].append(g_eid)
    print('\n******* Annotating *******')
    g_dict = dict()
    worker_results = []

    if args.debug : # Debug
        for pid in range(args.n_processes):
            worker_results.append(worker_annotate(
                pid,
                args,
                generator,
                generate_eids_group[pid],
                dataset
            ))
        g_dict = worker_results[0]

    else :
        pool = multiprocessing.Pool(processes=args.n_processes)
        for pid in range(args.n_processes):
            worker_results.append(pool.apply_async(worker_annotate, args=(
                pid,
                args,
                generator,
                generate_eids_group[pid],
                dataset
            )))
        # Merge annotation results
        for r in worker_results:
            worker_g_dict = r.get()
            g_dict.update(worker_g_dict)
        pool.close()
        pool.join()

    acc_result = torch.tensor([g_dict[k]['pred_score'] for k in g_dict.keys()]).sum().item()/len(g_dict.keys())
    print(f'EM Acc:{acc_result:.4f}')
    print(f"Elapsed time: {time.time() - start_time}")

    g_dict.update({'EM_Acc': acc_result})

    # Save annotation results
    save_file_name = f'program_{args.dataset}_{args.dataset_split}.json'
    with open(os.path.join(args.save_dir, save_file_name), 'w') as f:
        json.dump(g_dict, f, indent=4)


if __name__ == '__main__':
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    # File path or name
    parser.add_argument('--dataset', type=str, default='open_wikitq',
                        choices=['open_wikitq', 'wikitq', 'feverous'])
    parser.add_argument('--dataset_split', type=str, default='validation', choices=['train', 'validation', 'test'])
    parser.add_argument('--dataset_sub', type=int, default=128)
    parser.add_argument('--dataset_chunk_idx', type=int, default=0)

    parser.add_argument('--api_keys_file', type=str, default='key.txt')
    parser.add_argument('--prompt_file', type=str, default='templates/prompts/wikitq.txt')
    parser.add_argument('--save_dir', type=str, default='results/')

    # Multiprocess options
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n_processes', type=int, default=2)
    parser.add_argument('--gold_table', action='store_true')
    parser.add_argument('--multi_table_count', type=int, default=2)
    parser.add_argument('--openai_api_key', type=str, default='OPENAI_API_KEY')
    parser.add_argument('--multi_table_type', type=str, default='random', choices=['bm25', 'random'])
    parser.add_argument('--bm25_rerank_rows', action='store_true')
    parser.add_argument('--ltm', action='store_true')
    parser.add_argument('--ltm_select', type=str, default=None)
    parser.add_argument('--one_by_one_idx', type=int, default=None)
    parser.add_argument('--rankgpt', action='store_true')

    # program generation options
    parser.add_argument('--prompt_style', type=str, default='create_table_select_3_full_table',
                        choices=['create_table_select_3_full_table',
                                 'create_table_select_full_table',
                                 'create_table_select_3',
                                 'create_table',
                                 'create_table_select_3_full_table_w_all_passage_image',
                                 'create_table_select_3_full_table_w_gold_passage_image',
                                 'no_table'])
    parser.add_argument('--generate_type', type=str, default='sqlite',
                        choices=['answer', 'sql', 'sqlite', 'sqlite-verifier'])
    parser.add_argument('--n_shots', type=int, default=14)
    parser.add_argument('--seed', type=int, default=42)

    # options
    parser.add_argument('--engine', type=str, default="gpt-3.5-turbo", 
                        choices=['gpt-3.5-turbo', 'gpt-4', 'falcon', 'falcon-chat', "gpt-3.5-turbo-0613"])
    parser.add_argument('--n_parallel_prompts', type=int, default=1)
    parser.add_argument('--max_generation_tokens', type=int, default=512)
    parser.add_argument('--max_api_total_tokens', type=int, default=8001)
    parser.add_argument('--temperature', type=float, default=0.4)
    parser.add_argument('--sampling_n', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--stop_tokens', type=str, default='\n\n',
                        help='Split stop tokens by ||')

    # debug options
    parser.add_argument('-v', '--verbose', action='store_false')

    args = parser.parse_args()
    args.stop_tokens = args.stop_tokens.split('||')
    print("Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    main()
