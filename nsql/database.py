import copy
import os
import sqlite3
import records
import sqlalchemy
import pandas as pd
from typing import Dict, List
import uuid
import re

from utils.normalizer import convert_df_type, prepare_df_for_neuraldb_from_table
from utils.mmqa.image_stuff import get_caption

import time

def check_in_and_return(key: str, source: dict):
    # `` wrapped means as a whole
    if key.startswith("`") and key.endswith("`"):
        key = key[1:-1]
    if key in source.keys():
        return source[key]
    else:
        for _k, _v in source.items():
            if _k.lower() == key.lower():
                return _v
        raise ValueError("{} not in {}".format(key, source))

def make_sqlite_friendly(name: str):
    """
    Convert name into SQLite-friendly format:
    It should start with a letter or underscore, followed by any alphanumeric characters or underscore.
    """
    # Replace invalid characters with underscore
    new_name = re.sub(r'[^_a-zA-Z0-9]', '_', name)

    # Make sure it starts with a letter or underscore
    if not re.match(r'^[_a-zA-Z]', new_name):
        new_name = '_' + new_name

    return new_name

class NeuralDB(object):
    def __init__(self, tables: List[Dict[str, Dict]], passages=None, images=None):
        self.raw_tables = copy.deepcopy(tables)
        self.passages = {}
        self.images = {}
        self.image_captions = {}
        self.passage_linker = {}  # The links from cell value to passage
        self.image_linker = {}  # The links from cell value to images

        # Get passages
        if passages:
            for passage in passages:
                title, passage_content = passage['title'], passage['text']
                self.passages[title] = passage_content

        # Get images
        if images:
            for image in images:
                _id, title, picture = image['id'], image['title'], image['pic']
                self.images[title] = picture
                self.image_captions[title] = get_caption(_id)

        # Link grounding resources from other modalities(passages, images).
        if self.raw_tables[0]['table'].get('rows_with_links', None):
            rows = self.raw_tables[0]['table']['rows']
            rows_with_links = self.raw_tables[0]['table']['rows_with_links']

            link_title2cell_map = {}
            for row_id in range(len(rows)):
                for col_id in range(len(rows[row_id])):
                    cell = rows_with_links[row_id][col_id]
                    for text, title, url in zip(cell[0], cell[1], cell[2]):
                        text = text.lower().strip()
                        link_title2cell_map[title] = text

            # Link Passages
            for passage in passages:
                title, passage_content = passage['title'], passage['text']
                linked_cell = link_title2cell_map.get(title, None)
                if linked_cell:
                    self.passage_linker[linked_cell] = title

            # Images
            for image in images:
                title, picture = image['title'], image['pic']
                linked_cell = link_title2cell_map.get(title, None)
                if linked_cell:
                    self.image_linker[linked_cell] = title

        for table_info in tables:
            table_info['table'] = prepare_df_for_neuraldb_from_table(table_info['table'], normalize=True, add_row_id=False)

        self.tables = tables

        # Connect to SQLite database
        self.tmp_path = "tmp"
        os.makedirs(self.tmp_path, exist_ok=True)
        # self.db_path = os.path.join(self.tmp_path, '{}.db'.format(hash(time.time())))
        self.db_path = os.path.join(self.tmp_path, '{}.db'.format(uuid.uuid4()))
        self.sqlite_conn = sqlite3.connect(self.db_path)


        # MYSQL
        # database = 'test_name'
        # self.db_path = os.path.join(self.tmp_path, '{}.db'.format(database))
        # host, user = 'localhost', 'root'
        # password = 'kong_mysql_ec2_2023_Amazon'
        # try :
        #     self.mysql_conn = mysql.connector.connect(
        #         host=host,
        #         user=user,
        #         password=password,
        #         database=database
        #     )
        # except :
        #     self.mysql_conn = mysql.connector.connect(
        #         host=host,
        #         user=user,
        #         password=password
        #     )
        #     mycursor = self.mysql_conn.cursor()
        #     mycursor.execute(f"CREATE DATABASE {database}")
        #     mycursor.close()



        # Create DB
        assert len(tables) >= 1, "DB has no table inside"

        self.table_names, self.table_dict = [], {}
        if len(tables) > 1:
            for table_0 in tables :
                # Column name processing 
                new_column_names = [make_sqlite_friendly(name) for name in table_0["table"].columns.tolist()]
                table_0["table"].rename(columns=dict(zip(table_0["table"].columns.tolist(), new_column_names)), inplace=True)

                # Table name processing 
                table_title = table_0.get('title', None)
                if table_title is None :
                    table_name = 'table_name_tmp'
                else :
                    table_name = make_sqlite_friendly(table_title)

                try :
                    table_0["table"].to_sql(table_name, self.sqlite_conn)
                except Exception as e :
                    print(e)

                self.table_names.append(table_name)
                self.table_dict[table_name] = table_0["table"]

        else:
            table_0 = tables[0]
            # Column name processing 
            new_column_names = [make_sqlite_friendly(name) for name in table_0["table"].columns.tolist()]
            table_0["table"].rename(columns=dict(zip(table_0["table"].columns.tolist(), new_column_names)), inplace=True)

            # Table name processing 
            self.table_title = table_0.get('title', None)
            if self.table_title is None :
                self.table_name = 'table_name_tmp'
            else :
                self.table_name = make_sqlite_friendly(self.table_title)

            try :
                table_0["table"].to_sql(self.table_name, self.sqlite_conn)
            except Exception as e :
                print(e)

        # Records conn
        self.db = records.Database('sqlite:///{}'.format(self.db_path))
        self.records_conn = self.db.get_connection()

    def __str__(self):
        return str(self.execute_query("SELECT * FROM {}".format(self.table_name)))

    def get_table(self, table_name=None):
        table_name = self.table_name if not table_name else table_name
        sql_query = "SELECT * FROM {}".format(table_name)
        _table = self.execute_query(sql_query)
        return _table

    def get_header(self, table_name=None):
        _table = self.get_table(table_name)
        return _table['header']

    def get_rows(self, table_name):
        _table = self.get_table(table_name)
        return _table['rows']

    def get_table_df(self):
        return self.tables[0]['table']

    def get_table_raw(self):
        return self.raw_tables[0]['table']

    def get_table_title(self):
        # return self.tables[0]['title']
        return self.table_name

    def get_passages_titles(self):
        return list(self.passages.keys())

    def get_images_titles(self):
        return list(self.images.keys())

    def get_passage_by_title(self, title: str):
        return check_in_and_return(title, self.passages)

    def get_image_by_title(self, title):
        return check_in_and_return(title, self.images)

    def get_image_caption_by_title(self, title):
        return check_in_and_return(title, self.image_captions)

    def get_image_linker(self):
        return copy.deepcopy(self.image_linker)

    def get_passage_linker(self):
        return copy.deepcopy(self.passage_linker)


    def execute_query_my(self, sql_query: str) :
        # cursor = self.sqlite_conn.cursor()
        # # cursor.executescript(sql_query)
        # cursor.executescript('SELECT * FROM w')
        # results = cursor.fetchall()
        # cursor.close()

        try :
            out = self.records_conn.query(sql_query)
            headers = out.dataset.headers
            results = out.all()
        except Exception as e :
            results, headers = [], []
            print(f"Execution error: {e}")
            time.sleep(20)
            return {"header": ['<execution error>'], "rows": [['<execution error>']]}

        unmerged_results = []
        merged_results = []

        for i in range(len(results)):
            unmerged_results.append(list(results[i].values()))
            merged_results.extend(results[i].values())

        return {"header": headers, "rows": unmerged_results}


    def execute_query(self, sql_query: str):
        """
        Basic operation. Execute the sql query on the database we hold.
        @param sql_query:
        @return:
        """
        # When the sql query is a column name (@deprecated: or a certain value with '' and "" surrounded).
        if len(sql_query.split(' ')) == 1 or (sql_query.startswith('`') and sql_query.endswith('`')):
            col_name = sql_query
            new_sql_query = r"SELECT row_id, {} FROM {}".format(col_name, self.table_name)
            # Here we use a hack that when a value is surrounded by '' or "", the sql will return a column of the value,
            # while for variable, no ''/"" surrounded, this sql will query for the column.
            out = self.records_conn.query(new_sql_query)
        # When the sql query wants all cols or col_id, which is no need for us to add 'row_id'.
        elif sql_query.lower().startswith("select *") or sql_query.startswith("select col_id"):
            out = self.records_conn.query(sql_query)
        else:
            try:
                # SELECT row_id in addition, needed for result and old table alignment.
                new_sql_query = "SELECT row_id, " + sql_query[7:]
                out = self.records_conn.query(new_sql_query)
            except sqlalchemy.exc.OperationalError as e:
                # Execute normal SQL, and in this case the row_id is actually in no need.
                out = self.records_conn.query(sql_query)

        results = out.all()
        unmerged_results = []
        merged_results = []

        headers = out.dataset.headers
        for i in range(len(results)):
            unmerged_results.append(list(results[i].values()))
            merged_results.extend(results[i].values())

        return {"header": headers, "rows": unmerged_results}

    def add_sub_table(self, sub_table, table_name=None, verbose=True):
        """
        Add sub_table into the table.
        @return:
        """
        table_name = self.table_name if not table_name else table_name
        sql_query = "SELECT * FROM {}".format(table_name)
        oring_table = self.execute_query(sql_query)
        old_table = pd.DataFrame(oring_table["rows"], columns=oring_table["header"])
        # concat the new column into old table
        sub_table_df_normed = convert_df_type(pd.DataFrame(data=sub_table['rows'], columns=sub_table['header']))
        new_table = old_table.merge(sub_table_df_normed,
                                    how='left', on='row_id')  # do left join
        new_table.to_sql(table_name, self.sqlite_conn, if_exists='replace',
                         index=False)
        if verbose:
            print("Insert column(s) {} (dtypes: {}) into table.\n".format(', '.join([_ for _ in sub_table['header']]),
                                                                          sub_table_df_normed.dtypes))
