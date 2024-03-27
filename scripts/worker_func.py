import sys
sys.path.append('../')

from generation.generator import Generator

from typing import List

from nsql.database import NeuralDB

from utils.sql_utils import sqlite_execute, sqlite_syntax_check
from utils.evaluator import Evaluator

import copy

def worker_annotate(
    pid: int,
    args,
    generator: Generator,
    g_eids: List,
    dataset
):
    """
    A worker process for annotating.
    """
    g_dict = dict()
    built_few_shot_prompts = []
    for g_eid in g_eids:

        accu_error_string, error_string, rerun_round, subtable, sql_prog_to_save, sql_type_to_save = '', '', 1, '', None, None 
        g_data_item = dataset[g_eid]
        g_dict[g_eid] = {
            'generations': [],
            'db_data_item': g_data_item['table']
        }

        if args.gold_table :
            db = NeuralDB(
                tables=[{'title': g_data_item['table']['page_title'], 'table': g_data_item['table']}]
            )
        else :
            db = NeuralDB(
                tables=[{'title': other_table['page_title'], 'table':  other_table} for other_table in g_data_item['other_tables']]
            )
        
        # g_dict[g_eid].update({
        #     'db_data_item': {
        #         'headers': list(db.get_table_df().columns),
        #         'rows': db.get_table_df().values.tolist()
        #     }
        # })

        few_shot_prompt = generator.build_few_shot_prompt_from_file(
            file_path=args.prompt_file,
            n_shots=args.n_shots
        ) #[SEP]

        while True :

            if args.gold_table :
                generate_prompt = generator.build_generate_prompt(
                    data_item={
                        'tables': [db.get_table_df()],
                        'titles': [db.get_table_title()],
                        'question': g_data_item['question']
                    },
                    generate_type=(args.generate_type,),
                    error_string=error_string
                )
            else :
                generate_prompt = generator.build_generate_prompt(
                    data_item={
                        'tables': [db.tables[i]['table'] for i in range(len(db.tables))],
                        'titles': db.table_names,
                        'question': g_data_item['question']
                    },
                    generate_type=(args.generate_type,),
                    error_string=error_string
                )     

            prompt = few_shot_prompt + "\n\n" + generate_prompt 

            built_few_shot_prompts = [(g_eid, prompt)]

            response_dict = generator.generate_one_pass(
                prompts=built_few_shot_prompts,
                verbose=args.verbose
            )
            tmp_sql = list(response_dict.values())[0][0][0]

            if args.ltm : 
                tmp_sql_list = tmp_sql.split('[SQLSEP]')

                if args.ltm_select is None : # no select

                    tmp_sql_types = ['advan', 'inter', 'basic']
                    for sql_idx, sql_prog in enumerate(reversed(tmp_sql_list)) :

                        sql_type_to_save = tmp_sql_types[2 if sql_idx > 2 else sql_idx]
                        sql_prog_to_save = sql_prog

                        check_result, pred_result, subtable = sqlite_syntax_check(sql_prog, db, generator, g_eid, g_data_item['question'], g_data_item['answer_text'], args)
                        error_string = copy.deepcopy(check_result)
                        print(check_result)
                        if check_result is None :
                            break

                else :
                    num_prog = len(tmp_sql_list)

                    if args.ltm_select == 'advan' :
                        sql_prog = tmp_sql_list[-1]
                    elif args.ltm_select == 'inter' :
                        sql_prog = tmp_sql_list[1-num_prog]
                    elif args.ltm_select == 'basic' :
                        sql_prog = tmp_sql_list[0]
                    else :
                        raise NotImplementedError

                    sql_prog_to_save = sql_prog

                    check_result, pred_result, subtable = sqlite_syntax_check(sql_prog, db, generator, g_eid, g_data_item['question'], g_data_item['answer_text'], args)
                    error_string = copy.deepcopy(check_result)

                break

            if args.generate_type == 'answer' :
                pred_result = [tmp_sql,]
                break

            if args.generate_type == 'sql' :
                pred_result = sqlite_execute(tmp_sql, db)
                break

            check_result, pred_result, subtable = sqlite_syntax_check(tmp_sql, db, generator, g_eid, g_data_item['question'], g_data_item['answer_text'], args)
            error_string = copy.deepcopy(check_result)

            if check_result is None :
                print('Execution check passed!')
                break
            else :
                print(f'Execution check failed. {check_result}. Rerun round: {rerun_round}')
                rerun_round += 1
                accu_error_string += f'Avoid generating programs that will yield error like \"{check_result}\" when executed \n'

            if rerun_round > 5 :
                break

        for eid, g_pairs in response_dict.items():
            g_pairs = sorted(g_pairs, key=lambda x: x[-1], reverse=True)
            g_dict[eid]['generations'] = g_pairs
        
        g_dict[g_eid]['run_rounds'] = rerun_round
        score = Evaluator().evaluate(
            pred_result,
            g_data_item['answer_text'],
            dataset=args.dataset,
            question=g_data_item['question']
        )
        g_dict[g_eid]['pred_score'] = score
        g_dict[g_eid]['answer_text'] = g_data_item['answer_text']
        g_dict[g_eid]['pred_result'] = pred_result
        g_dict[g_eid]['subtable'] = subtable
        g_dict[g_eid]['error'] = error_string
        g_dict[g_eid]['question'] = g_data_item['question']
        g_dict[g_eid]['sql_prog_used'] = sql_prog_to_save
        g_dict[g_eid]['sql_type_used'] = sql_type_to_save

    return g_dict