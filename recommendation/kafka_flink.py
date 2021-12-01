# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from pyflink.datastream.stream_execution_environment import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment


stream_env = StreamExecutionEnvironment.get_execution_environment()
table_env = StreamTableEnvironment.create(stream_env)
statement_set = table_env.create_statement_set()


table_env.execute_sql('''
            create table raw_input (
                record varchar
            ) with (
                'connector' = 'kafka',
                'topic' = 'raw_input',
                'properties.bootstrap.servers' = 'localhost:9092',
                'properties.group.id' = 'raw_input',
                'format' = 'csv',
                'csv.field-delimiter' = '|',
                'scan.startup.mode' = 'earliest-offset'
            )
        ''')


table_env.execute_sql('''
                        create table print (
                            record varchar
                        ) with (
                            'connector' = 'print'
                        )
                    ''')

source = table_env.from_path('raw_input')

statement_set.add_insert('print', source)

job_client = statement_set.execute().get_job_client()
if job_client is not None:
    job_client.get_job_execution_result().result()
