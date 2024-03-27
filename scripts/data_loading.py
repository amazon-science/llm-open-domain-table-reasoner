import random
import sys
sys.path.append('../')

import pandas as pd
import os
import json
import jsonlines

from open_wikitable.dataloader import OpenWikiTableFull
from utils.bm25 import bm25_reranking
from utils.utils import load_data_split


ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")

def table_check(table_dict, table_id) :
    table_content, table_type, _ = table_dict[table_id]
    if table_type == 'general' :
        row_lens =  [ len(row) for row in table_content ]
        if all(row_len == row_lens[0] for row_len in row_lens) :
            cells = []
            for row in table_content :
                for cell in row :
                    cells.append(cell)

            if all( cell != '' and type(cell) == str for cell in cells ) :
                return True    
    return False

def wikitable(args) :
    dataset = load_data_split(args.dataset, args.dataset_split)
    random.seed(42)

    dataset = [item for item in dataset]
    dataset = dataset[:args.dataset_sub]

    return dataset

def feverous(args) :

    abs_path = '/FEVEROUS/FEVEROUS-Parse'

    train_list, val_list = [], []
    with jsonlines.open(os.path.join(abs_path, 'data/feverous_train_opentab.jsonl'), 'r') as reader:
        for line in reader:
            train_list.append(line)
    with jsonlines.open(os.path.join(abs_path, 'data/feverous_validation_opentab.jsonl'), 'r') as reader:
        for line in reader:
            val_list.append(line)

    table_dict = {}
    def to_samples(split_list) :
        claim_samples, label_samples, table_id_samples, context_samples = [], [], [], []
        for raw_data_dict in split_list :

            for table in raw_data_dict['table_list'] :
                if table['table_id'] not in table_dict :
                    table_dict[table['table_id']] = (table['table_content'], table['table_type'], table['table_abnormal'])

            claim_samples.append(raw_data_dict['statement'])
            label_samples.append(raw_data_dict['label'])
            table_id_samples.append([ table['table_id'] for table in raw_data_dict['table_list']])
            context_samples.append(raw_data_dict['context'])
            
        return claim_samples, label_samples, table_id_samples, context_samples

    train_claims, _, _, _ = to_samples(train_list)
    val_claims, val_labels, val_table_ids, val_contexts = to_samples(val_list)

    intact_val_ids = []
    for i, val_table_id_list in enumerate(val_table_ids) :
        if len(val_table_id_list) == 1 and val_contexts[i] == '' and val_labels[i] != 'not enough info' :
            val_table_id = val_table_id_list[0]

            if table_check(table_dict, val_table_id) :
                intact_val_ids.append(i)


    with open('/FEVEROUS/FEVEROUS-BM25/val_preds_50.json', 'r') as file :
        bm25_table_id_lists = json.load(file)


    dataset = []
    for val_id in intact_val_ids :
        bm25_table_id_list = bm25_table_id_lists[val_id]
        original_table_id = val_table_ids[val_id][0]

        # GT table
        original_table = table_dict[original_table_id][0]

        the_data_dict = {
            'question': val_claims[val_id],
            'table_id': original_table_id,
            'table': {
                'page_title': original_table_id, 
                'header': original_table[0],
                'rows': original_table[1:],
            },
            'answer_text': [val_labels[val_id],]
        }

        # multi table
        if (not args.gold_table) or (args.gold_table and args.one_by_one_idx is not None) :            
            # bm25 tables purely
            if args.multi_table_type == 'bm25' :  
                random_tables = [
                    (table_dict[idx][0], idx) if table_check(table_dict, idx) else ([['NAN'],['NAN']],idx) for idx in bm25_table_id_list[:args.multi_table_count]
                ]
            the_data_dict.update({
                'other_tables': [{
                    'page_title': random_table[1],
                    'header': random_table[0][0],
                    'rows': random_table[0][1:], 
                } for random_table in random_tables]
            })

            if args.gold_table and args.one_by_one_idx is not None :
                one_by_one_table = the_data_dict['other_tables'][args.one_by_one_idx]
                one_by_one_table['retrieve_correct'] = bm25_table_id_list[args.one_by_one_idx] == original_table_id
                
                the_data_dict.update({
                    'table': one_by_one_table,
                })
                
        dataset.append(the_data_dict)

    random.seed(42)
    random.shuffle(dataset)

    start_idx = args.dataset_chunk_idx * args.dataset_sub
    end_idx = (args.dataset_chunk_idx+1) * args.dataset_sub
    if end_idx > len(dataset) :
        exit(0)

    dataset = dataset[start_idx:end_idx]

    if args.bm25_rerank_rows :
        for data_dict in dataset :
            query = data_dict['question']
            if args.gold_table :
                rows = data_dict['table']['rows']
                reranking = bm25_reranking(rows=rows, query=query)
                reranked_rows = [rows[i] for i in reranking]
                data_dict['table']['rows'] = reranked_rows
            else :
                for the_tab in the_data_dict['other_tables'] :
                    rows = the_tab['rows']
                    reranking = bm25_reranking(rows=rows, query=query)
                    reranked_rows = [rows[i] for i in reranking]
                    the_tab['rows'] = reranked_rows

    return dataset

def open_wikitable(args) :
    tables = pd.read_json(os.path.join(ROOT_DIR, 'open_wikitable', 'data', "tables.json"))
    splitted_tables = pd.read_json(os.path.join(ROOT_DIR, 'open_wikitable', 'data', "splitted_tables.json"))
    original_table_id_convert_list = splitted_tables['original_table_id'].tolist()

    # data can be downloaded here https://github.com/sean0042/Open_WikiTable/tree/main
    if args.dataset_split == 'train' :
        pred_dataset = OpenWikiTableFull(pd.read_json(os.path.join(ROOT_DIR, 'open_wikitable', 'data', "train.json")), train = False)
    elif args.dataset_split == 'validation' :
        pred_dataset = OpenWikiTableFull(pd.read_json(os.path.join(ROOT_DIR, 'open_wikitable', 'data', "valid.json")), train = False)
    elif args.dataset_split == 'test' :
        pred_dataset = OpenWikiTableFull(pd.read_json(os.path.join(ROOT_DIR, 'open_wikitable', 'data', "test.json")), train = False)
    else :
        raise ValueError('Not implemented')

    with open(os.path.join(ROOT_DIR, 'open_wikitable', 'data', f"{args.dataset_split}_preds_50.json")) as file:
        bm25_table_id_lists = json.load(file)

    dataset = []
    for val_sample, bm25_table_id_list in zip(pred_dataset, bm25_table_id_lists) :
        # 6021 vs. 6602
        if val_sample['answer'][0] == 'None' : 
            continue

        original_table_id = val_sample['original_table_id']

        # GT table
        original_table = tables[tables['original_table_id'] == original_table_id]

        the_data_dict = {
            'question': val_sample['question'],
            'table_id': list(original_table['dataset'])[0] + '_' + list(original_table['original_table_id'])[0],
            'table': {
                'page_title': list(original_table['page_title'])[0] + list(original_table['original_table_id'])[0],
                'header': list(original_table['header'])[0],
                'rows': list(original_table['rows'])[0],
            },
            'answer_text': val_sample['answer']
        }



        # multi table
        if (not args.gold_table) or (args.gold_table and args.one_by_one_idx is not None) :
            # Random tables including GT
            if args.multi_table_type == 'random' :
                random_tables = [tables.sample(n=1) for _ in range(args.multi_table_count-1)]
                random_tables.append(original_table)
                random.shuffle(random_tables)
            
            # bm25 tables purely
            elif args.multi_table_type == 'bm25' :  

                random_tables = [
                    tables[tables['original_table_id'] == original_table_id_convert_list[idx]] for idx in bm25_table_id_list[:args.multi_table_count]
                ]


            the_data_dict.update({
                'other_tables': [{
                    'page_title': list(random_table['page_title'])[0] + list(original_table['original_table_id'])[0],
                    'header': list(random_table['header'])[0],
                    'rows': list(random_table['rows'])[0], 
                } for random_table in random_tables]
            })

            if args.gold_table and args.one_by_one_idx is not None :
                one_by_one_table = the_data_dict['other_tables'][args.one_by_one_idx]
                the_data_dict.update({
                    'table': one_by_one_table
                })
                
        dataset.append(the_data_dict)

    random.seed(42)
    random.shuffle(dataset)

    start_idx = args.dataset_chunk_idx * args.dataset_sub
    end_idx = (args.dataset_chunk_idx+1) * args.dataset_sub
    if end_idx > len(dataset) :
        exit(0)

    dataset = dataset[start_idx:end_idx]

    if args.bm25_rerank_rows :
        for data_dict in dataset :
            query = data_dict['question']
            if args.gold_table :
                rows = data_dict['table']['rows']
                reranking = bm25_reranking(rows=rows, query=query)
                reranked_rows = [rows[i] for i in reranking]
                data_dict['table']['rows'] = reranked_rows
            else :
                for the_tab in the_data_dict['other_tables'] :
                    rows = the_tab['rows']
                    reranking = bm25_reranking(rows=rows, query=query)
                    reranked_rows = [rows[i] for i in reranking]
                    the_tab['rows'] = reranked_rows

    return dataset