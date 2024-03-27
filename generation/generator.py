"""
Generate nsql and questions.
"""

from typing import Dict, List, Union, Tuple

import time
import os

from text_generation import Client
import openai
from generation.prompt import PromptBuilder


class Generator(object):
    """
    Codex generation wrapper.
    """

    def __init__(self, args, keys=None):
        self.args = args
        self.keys = keys
        self.current_key_id = 0

        # if the args provided, will initialize with the prompt builder for full usage
        self.prompt_builder = PromptBuilder(args) if args else None
        self.openai_api_key = args.openai_api_key


    def prompt_row_truncate(
            self,
            prompt: str,
            num_rows_to_remain: int,
            table_end_token: str = '*/',
    ):
        """
        Fit prompt into max token limits by row truncation.
        """
        table_end_pos = prompt.rfind(table_end_token)
        assert table_end_pos != -1
        prompt_part1, prompt_part2 = prompt[:table_end_pos], prompt[table_end_pos:]
        prompt_part1_lines = prompt_part1.split('\n')[::-1]
        trunc_line_index = None
        for idx, line in enumerate(prompt_part1_lines):
            if '\t' not in line:
                continue
            row_id = int(line.split('\t')[0])
            if row_id <= num_rows_to_remain:
                trunc_line_index = idx
                break
        new_prompt_part1 = '\n'.join(prompt_part1_lines[trunc_line_index:][::-1])
        prompt = new_prompt_part1 + '\n' + prompt_part2
        return prompt

    def build_few_shot_prompt_from_file(
            self,
            file_path: str,
            n_shots: int
    ):
        """
        Build few-shot prompt for generation from file.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        few_shot_prompt_list = []
        one_shot_prompt = ''
        last_line = None
        for line in lines:
            if line == '\n' and last_line == '\n':
                few_shot_prompt_list.append(one_shot_prompt)
                one_shot_prompt = ''
            else:
                one_shot_prompt += line
            last_line = line
        few_shot_prompt_list.append(one_shot_prompt)
        few_shot_prompt_list = few_shot_prompt_list[:n_shots]
        few_shot_prompt_list[-1] = few_shot_prompt_list[
            -1].strip()  # It is essential for prompting to remove extra '\n'
        few_shot_prompt = '\n'.join(few_shot_prompt_list)
        return few_shot_prompt

    def build_few_shot_prompt_from_file_chat(
            self,
            file_path: str,
            n_shots: int
    ):
        """
        Build few-shot prompt for generation from file.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        few_shot_prompt_list = []
        one_shot_prompt = ''
        last_line = None
        for line in lines:
            if line == '\n' and last_line == '\n':
                few_shot_prompt_list.append(one_shot_prompt)
                one_shot_prompt = ''
            else:
                one_shot_prompt += line
            last_line = line
        few_shot_prompt_list.append(one_shot_prompt)
        few_shot_prompt_list = few_shot_prompt_list[:n_shots]
        few_shot_prompt_list[-1] = few_shot_prompt_list[
            -1].strip()  # It is essential for prompting to remove extra '\n'

        return few_shot_prompt_list

    def build_generate_prompt(
            self,
            data_item: Dict,
            generate_type: Tuple,
            **kwargs
    ):
        """
        Build the generate prompt
        """
        return self.prompt_builder.build_generate_prompt(
            **data_item,
            generate_type=generate_type,
            **kwargs
        )

    def generate_one_pass(
            self,
            prompts: List[Tuple],
            verbose: bool = False
    ):
        """
        Generate one pass with codex according to the generation phase.
        """
        result_idx_to_eid = []
        for p in prompts:
            result_idx_to_eid.extend([p[0]] * self.args.sampling_n)
        prompts = [p[1] for p in prompts]

        result = self._call_codex_api(
            engine=self.args.engine,
            prompt=prompts,
            max_tokens=self.args.max_generation_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            n=self.args.sampling_n,
            stop=self.args.stop_tokens
        )

        if verbose:
            print('\n', '*' * 20, 'Codex API Call', '*' * 20)
            for prompt in prompts:
                print(prompt)
                print('\n')
            print('- - - - - - - - - - ->>')

        # parse api results
        response_dict = dict()
        for idx, g in enumerate(result['choices']):
            try:
                text = g['message']['content']
                eid = result_idx_to_eid[idx]
                eid_pairs = response_dict.get(eid, None)
                if eid_pairs is None:
                    eid_pairs = []
                    response_dict[eid] = eid_pairs
                eid_pairs.append((text, 1))

                if verbose:
                    print(text)

            except ValueError as e:
                if verbose:
                    print('----------- Error Msg--------')
                    print(e)
                    print(text)
                    print('-----------------------------')
                pass

        return response_dict

    def generate_one_pass_end2end(
            self,
            prompts: List[Tuple],
            verbose: bool = False
    ):
        """
        Generate one pass with codex according to the generation phase.
        """
        result_idx_to_eid = []
        for p in prompts:
            result_idx_to_eid.extend([p[0]] * self.args.sampling_n)
        prompts = [p[1] for p in prompts]

        result = self._call_codex_api(
            engine=self.args.engine,
            prompt=prompts,
            max_tokens=self.args.max_generation_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            n=self.args.sampling_n,
            stop=self.args.stop_tokens
        )

        if verbose:
            print('\n', '*' * 20, 'Codex API Call', '*' * 20)
            for prompt in prompts:
                print(prompt)
                print('\n')
            print('- - - - - - - - - - ->>')

        # parse api results
        response_dict = dict()
        for idx, g in enumerate(result['choices']):

            try:
                text = g['message']['content']
                eid = result_idx_to_eid[idx]
                eid_pairs = response_dict.get(eid, None)
                if eid_pairs is None:
                    eid_pairs = []
                    response_dict[eid] = eid_pairs
                eid_pairs.append((text, result['usage']['total_tokens'], 1))

                if verbose:
                    print(text)

            except ValueError as e:
                if verbose:
                    print('----------- Error Msg--------')
                    print(e)
                    print(text)
                    print('-----------------------------')
                pass

        return response_dict

    def _call_codex_api(
            self,
            engine: str,
            prompt: Union[str, List],
            max_tokens,
            temperature: float,
            top_p: float,
            n: int,
            stop: List[str]
    ):
        start_time = time.time()
        result = None
        
        count = 0
        while result is None :
            if engine == 'falcon' :
                try :
                    client = Client("http://127.0.0.1:8080", timeout=100)
                    content = client.generate(prompt[0], max_new_tokens=150, stop_sequences=['\n\n',]).generated_text
                    result = {
                        'choices': [{
                            'message': {
                                'content': content
                            }
                        }],
                        'usage':{
                            'total_tokens': 1
                        }
                    }
                    return result

                except Exception as e:
                    result = {
                        'choices': [{
                            'message': {
                                'content': str(e)
                            }
                        }],
                        'usage':{
                            'total_tokens': 4097
                        }
                    }
                    return result

            else :
                try:
                    count += 1
                    openai.api_key = os.environ[self.openai_api_key]
                    result = openai.ChatCompletion.create(
                        model=engine,
                        messages=[
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt[0]}
                            ],
                        # max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=n,
                        stop=stop,
                    )
                    # print('Openai api inference time:', time.time() - start_time)
                    return result

                except Exception as e:
                    print(e, 'Retry.')
                    time.sleep(2)

                    if count > 3 :
                        result = {
                            'choices': [{
                                'message': {
                                    'content': str(e)
                                }
                            }],
                            'usage':{
                                'total_tokens': 4097
                            }
                        }
                        return result

# class ChatApp(object):
#     def __init__(self, openai_api_key, engine, demonstrations):
#         self.engine = engine
#         self.openai_api_key = openai_api_key
#         self.messages = [
#             {"role": "system", "content": "You are a helpful assistant."},
#         ]
#         self.icl(demonstrations)

#     def icl(self, demonstrations) :
#         for demonstration in demonstrations :
#             query, response = demonstration[0], demonstration[1]
#             self.messages.append({"role": "user", "content": query})
#             self.messages.append({"role": "assistant", "content": response })

#     def chat(self, message):
#         self.messages.append({"role": "user", "content": message})
#         response, count = None, 0
#         try :
#             count += 1
#             openai.api_key = os.environ[self.openai_api_key]
#             response = openai.ChatCompletion.create(
#                 model=self.engine,
#                 messages=self.messages
#             )
#         except Exception as e:
#             time.sleep(2)
#             if count > 5 :
#                 response = {
#                     'choices': [{
#                         'message': {
#                             'content': str(e)
#                         }
#                     }]
#                 }

#         self.messages.append({"role": "assistant", "content": response["choices"][0]["message"]['content']})
#         return response["choices"][0]["message"]['content']

#     def simple_chat(self, message):
#         self.messages.append({"role": "user", "content": message})
#         response, count = None, 0
#         while response is None :
#             try :
#                 count += 1
#                 openai.api_key = os.environ[self.openai_api_key]
#                 response = openai.ChatCompletion.create(
#                     model=self.engine,
#                     messages=self.messages
#                 )
#             except Exception as e:
#                 time.sleep(2)
#                 if count > 5 :
#                     response = {
#                         'choices': [{
#                             'message': {
#                                 'content': str(e)
#                             }
#                         }]
#                     }

#         return response["choices"][0]["message"]['content']