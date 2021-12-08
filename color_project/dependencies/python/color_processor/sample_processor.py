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
from typing import List, Tuple

from ai_flow_plugins.job_plugins import flink
from ai_flow_plugins.job_plugins.flink.flink_processor import ExecutionContext
from ai_flow_plugins.job_plugins.flink.flink_wrapped_env import WrappedStatementSet, WrappedTableEnvironment
from pyflink.table import Table, EnvironmentSettings, TableEnvironment
from pyflink.table import ScalarFunction, FunctionContext, DataTypes
from pyflink.table.udf import udf
from recommendation import db
from recommendation import config


class DataStreamEnv(flink.FlinkEnv):
    """
    FlinkStreamEnv is the default implementation of FlinkEnv, used in flink streaming jobs.
    """

    def create_env(self) -> Tuple[WrappedTableEnvironment, WrappedStatementSet]:
        _t_env = TableEnvironment.create(EnvironmentSettings.in_streaming_mode())
        _t_env.get_config().get_configuration().set_string("parallelism.default", "1")
        t_env = WrappedTableEnvironment.create_from(_t_env)
        t_env.get_config().get_configuration().set_string("taskmanager.memory.task.off-heap.size", '80m')
        t_env.get_config().get_configuration().set_string("execution.checkpointing.interval", "30sec")
        statement_set = t_env.create_statement_set()
        return t_env, statement_set


class BuildFeature(ScalarFunction):

    def __init__(self):
        super().__init__()

    def open(self, function_context: FunctionContext):
        db.init_db(uri=config.DbConn)

    def eval(self, uid, country, infer, click, f1, f2):
        print(uid)
        db.update_user_click_info(uid=uid, fs=infer + ' ' + str(click))
        return ' '.join([str(uid), str(country), f1, f2, str(click)])


class DataSourceProcessor(flink.FlinkPythonProcessor):
    def process(self, execution_context: ExecutionContext, input_list: List[Table] = None) -> List[Table]:
        uri = execution_context.config['dataset'].uri.split(',')
        print('Raw Queue uri {}'.format(uri))
        t_env = execution_context.table_env
        t_env.execute_sql('''
                            create table raw_input (
                                uid int,
                                infer varchar,
                                click int,
                                proc_time as PROCTIME()
                            ) with (
                                'connector' = 'kafka',
                                'topic' = '{}',
                                'properties.bootstrap.servers' = '{}',
                                'properties.group.id' = 'raw_input',
                                'format' = 'csv',
                                'csv.field-delimiter' = ' ',
                                'scan.startup.mode' = 'latest-offset'
                            )
                        '''.format(uri[1], uri[0]))
        return [t_env.from_path('raw_input')]


class SampleProcessor(flink.FlinkPythonProcessor):
    def process(self, execution_context: ExecutionContext, input_list: List[Table] = None) -> List[Table]:
        t_env = execution_context.table_env
        t_env.register_function('feature',
                                udf(f=BuildFeature(),
                                    input_types=[DataTypes.INT(), DataTypes.STRING(), DataTypes.STRING(),
                                                 DataTypes.STRING(), DataTypes.INT()],
                                    result_type=DataTypes.STRING()))
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
        return [result]


class QueueSinkProcessor(flink.FlinkPythonProcessor):
    def process(self, execution_context: ExecutionContext, input_list: List[Table] = None) -> List[Table]:
        uri = execution_context.config['dataset'].uri.split(',')
        print('Sample Queue uri {}'.format(uri))
        t_env = execution_context.table_env
        t_env.execute_sql('''
                            create table sample_queue (
                                record varchar
                            ) with (
                                'connector' = 'kafka',
                                'topic' = '{}',
                                'properties.bootstrap.servers' = '{}',
                                'format' = 'raw'
                            )
                        '''.format(uri[1], uri[0]))
        st_1 = execution_context.statement_set
        st_1.add_insert('sample_queue', input_list[0])
        return []


class FileSinkProcessor(flink.FlinkPythonProcessor):
    def process(self, execution_context: ExecutionContext, input_list: List[Table] = None) -> List[Table]:
        t_env = execution_context.table_env
        uri = execution_context.config['dataset'].uri
        print('uri {}'.format(uri))
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
                       '''.format(uri))
        st_1 = execution_context.statement_set
        st_1.add_insert('sample_files', input_list[0])
        return []
