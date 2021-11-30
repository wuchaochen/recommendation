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
import os
from pyflink.table import EnvironmentSettings, TableEnvironment

data_dir = os.path.join(os.path.dirname(__file__), '../data/')


class DataProcessor(object):

    def run(self):
        t_env = TableEnvironment.create(EnvironmentSettings.in_streaming_mode())
        t_env.get_config().get_configuration().set_string("parallelism.default", "1")
        t_env.execute_sql('''
                    create table raw_input (
                        record varchar
                    ) with (
                        'connector' = 'kafka',
                        'topic' = 'raw_input',
                        'properties.bootstrap.servers' = 'localhost:9092',
                        'properties.group.id' = 'raw_input',
                        'format' = 'raw',
                        'scan.startup.mode' = 'earliest-offset'
                    )
                ''')
        # t_env.execute_sql('''
        #             create table print (
        #                 record varchar
        #             ) with (
        #                 'connector' = 'print'
        #             )
        #         ''')
        t_env.execute_sql('''
                    create table sample_queue (
                        record varchar
                    ) with (
                        'connector' = 'kafka',
                        'topic' = 'sample_input',
                        'properties.bootstrap.servers' = 'localhost:9092',
                        'format' = 'raw'
                    )
                ''')
        t_env.execute_sql('''
                    create table sample_files (
                        record varchar
                    ) with (
                        'connector' = 'filesystem',
                        'path' = '{}',
                        'format' = 'raw',
                        'partition.default-name' = 'part-'
                    )
                '''.format(data_dir+'/output'))

        table = t_env.from_path('raw_input')
        st_1 = t_env.create_statement_set()
        # st_1.add_insert('sample_files', table)
        st_1.add_insert('sample_queue', table)
        st_1.execute().get_job_client().get_job_execution_result().result()


if __name__ == '__main__':
    dp = DataProcessor()
    dp.run()
