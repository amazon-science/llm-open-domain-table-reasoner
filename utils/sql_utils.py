import re
import os

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../")

def sqlite_execute(sql_statement, db) :

    error, answer, subtable = None, [], None
    try :
        output = db.records_conn.query(sql_statement)
        # <empty>
        if len(output.all()) == 0 :
            error = f'The previously generated SQLite execution is empty and not desired, \
                        so this time you need to avoid generating the same. \
                        \n{sql_statement}'
        else :
            headers, results = output.dataset.headers, output.all()
            unmerged_results, merged_results = [], []
            for i in range(len(results)):
                unmerged_results.append(list(results[i].values()))
                merged_results.extend(results[i].values())

            if all(item is None for item in merged_results):
                # <empty>
                error = f'The previously generated SQLite execution is empty and not desired, \
                            so this time you need to avoid generating the same. \
                            \n{sql_statement}'
            else :
                return merged_results

    except Exception as e:
        error = str(e)

    return [error]


def get_next_word_after_from(text):
    # The regular expression searches for the word "FROM" followed by 
    # any non-word characters and then captures the first word.
    match = re.search(r'FROM[^\w]+(\w+)', text, re.IGNORECASE)
    
    if match:
        return match.group(1)  # Return the captured word.
    else:
        return None


def sqlite_syntax_check(sql_statement, db, generator, g_eid, question, gt, args) :

    error, answer, subtable = None, [], None
    try :
        output = db.records_conn.query(sql_statement)
        # <empty>
        if len(output.all()) == 0 :
            error = f'The previously generated SQLite execution is empty and not desired, so this time you need to avoid generating the same.\n{sql_statement}'

        else :
            headers, results = output.dataset.headers, output.all()
            unmerged_results, merged_results = [], []
            for i in range(len(results)):
                unmerged_results.append(list(results[i].values()))
                merged_results.extend(results[i].values())

            if all(item is None for item in merged_results):
                # <empty>
                error = f'The previously generated SQLite execution is empty and not desired, so this time you need to avoid generating the same.\n{sql_statement}'
            else :
                if args.dataset == 'feverous' :
                    few_shot_prompt = generator.build_few_shot_prompt_from_file(
                        file_path=os.path.join(ROOT_DIR, 'templates/prompts/feverous_reader.txt'), 
                        n_shots=2
                    )
                else :
                    few_shot_prompt = generator.build_few_shot_prompt_from_file(
                        file_path=os.path.join(ROOT_DIR, 'templates/prompts/wikitq_reader.txt'),
                        n_shots=2
                    )

                tab_str = ''
                for header in headers :
                    tab_str += str(header) + '\t'
                for row in unmerged_results :
                    tab_str += '\n'
                    for item in row :
                        tab_str += str(item) + '\t'
                        
                if args.gold_table :
                    generate_prompt = generator.build_generate_prompt(
                        data_item={
                            'tables': [db.get_table_df()],
                            'titles': [db.get_table_title()],
                            'question': question
                        },
                        generate_type=('sqlite-reader',),
                        sqlite_program=sql_statement,
                        execution_result=tab_str
                    )              
                else :
                    table_name = get_next_word_after_from(sql_statement)
                    generate_prompt = generator.build_generate_prompt(
                        data_item={
                            'tables': [db.table_dict[table_name]],
                            'titles': [table_name],
                            'question': question
                        },
                        generate_type=('sqlite-reader',),
                        sqlite_program=sql_statement,
                        execution_result=tab_str
                    )

                prompt = few_shot_prompt + "\n\n" + generate_prompt 
                built_few_shot_prompts = [(g_eid, prompt)]
                response_dict = generator.generate_one_pass(
                    prompts=built_few_shot_prompts,
                    verbose=args.verbose
                )
                answer = list(response_dict.values())[0][0][0]
                answer = answer.split('[SEP]')

                subtable = (headers, unmerged_results)


    except Exception as e:
        error = str(e)

    return error, answer, subtable