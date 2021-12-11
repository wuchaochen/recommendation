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
import time

from pyflink.table import EnvironmentSettings, TableEnvironment, ScalarFunction, FunctionContext, DataTypes
from pyflink.table.udf import udf
from recommendation import db
from recommendation import config


class BuildFeature(ScalarFunction):

    def __init__(self):
        super().__init__()

    def open(self, function_context: FunctionContext):
        db.init_db(uri=config.DbConn)

    def eval(self, uid, country, infer, click, f1, f2):
        db.update_user_click_info(uid=uid, fs=infer + ' ' + str(click))
        return ' '.join([str(uid), str(country), f1, f2, str(click)])


class DataProcessor(object):

    def run(self):
        t_env = TableEnvironment.create(EnvironmentSettings.in_streaming_mode())
        t_env.get_config().get_configuration().set_string("parallelism.default", str(config.partition_num))
        t_env.get_config().get_configuration().set_string("execution.checkpointing.interval", "30sec")
        t_env.register_function('feature',
                                udf(f=BuildFeature(),
                                    input_types=[DataTypes.INT(), DataTypes.STRING(), DataTypes.STRING(),
                                                 DataTypes.STRING(), DataTypes.INT()],
                                    result_type=DataTypes.STRING()))
        t_env.execute_sql('''
                    create table raw_input (
                        uid int,
                        infer varchar,
                        click int,
                        proc_time as PROCTIME()
                    ) with (
                        'connector' = 'kafka',
                        'topic' = 'raw_input',
                        'properties.bootstrap.servers' = 'localhost:9092',
                        'properties.group.id' = 'raw_input',
                        'format' = 'csv',
                        'csv.field-delimiter' = ' ',
                        'scan.startup.mode' = 'latest-offset'
                    )
                ''')
        t_env.execute_sql('''
                    create table print (
                        record varchar
                    ) with (
                        'connector' = 'blackhole'
                    )
                ''')
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
                        'sink.rolling-policy.rollover-interval' = '60sec',
                        'sink.rolling-policy.check-interval' = '10sec'
                    )
                '''.format(config.SampleFileDir))

        t_env.execute_sql(f'''
                    create table user_c (
                        uid int,
                        country int
                    ) with (
                        'connector' = 'jdbc',
                        'url' = 'jdbc:mysql://localhost:3306/user_info',
                        'table-name' = 'user',
                        'username' = '{config.DbUserName}',
                        'password' = '{config.DbPassword}'
                    )
                        ''')

        t_env.execute_sql(f'''
                    create table user_click (
                        uid int,
                        fs_1 varchar,
                        fs_2 varchar
                    ) with (
                        'connector' = 'jdbc',
                        'url' = 'jdbc:mysql://localhost:3306/user_info',
                        'table-name' = 'user_click',
                        'username' = '{config.DbUserName}',
                        'password' = '{config.DbPassword}'
                    )
                        ''')
        result = t_env.sql_query('''
            select feature(t.uid, t.country, t.infer, t.click, uc.fs_1, uc.fs_2) from
                    (select r.uid, c.country, r.infer, r.click, r.proc_time from raw_input as r
                        left outer join user_c FOR SYSTEM_TIME AS OF r.proc_time AS c
                        on r.uid = c.uid) as t
                    left outer join user_click FOR SYSTEM_TIME AS OF t.proc_time AS uc
                    on t.uid = uc.uid
        ''')
        st_1 = t_env.create_statement_set()
        st_1.add_insert('sample_queue', result)
        # st_1.add_insert('sample_files', result)
        # st_1.add_insert('print', result)
        st_1.execute().get_job_client().get_job_execution_result().result()


if __name__ == '__main__':
    dp = DataProcessor()
    dp.run()
