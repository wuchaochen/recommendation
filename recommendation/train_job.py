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
import subprocess
import shutil
from pyflink.datastream.stream_execution_environment import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment

from flink_ml_tensorflow.tensorflow_TFConfig import TFConfig
from flink_ml_tensorflow.tensorflow_on_flink_table import train
from flink_ml_tensorflow.tensorflow_on_flink_mlconf import MLCONSTANTS


class TrainJob(object):
    @staticmethod
    def batch_train():
        stream_env = StreamExecutionEnvironment.get_execution_environment()
        table_env = StreamTableEnvironment.create(stream_env)
        statement_set = table_env.create_statement_set()

        work_num = 2
        ps_num = 1
        python_file = "model_trainer.py"
        func = "batch_train"
        prop = {MLCONSTANTS.PYTHON_VERSION: '3.7',
                MLCONSTANTS.USE_DISTRIBUTE_CACHE: 'false',
                MLCONSTANTS.CONFIG_STORAGE_TYPE: MLCONSTANTS.STORAGE_ZOOKEEPER,
                MLCONSTANTS.CONFIG_ZOOKEEPER_CONNECT_STR: 'localhost:2181',
                MLCONSTANTS.REMOTE_CODE_ZIP_FILE: 'file:///tmp/code.zip',
                'checkpoint_dir': '/tmp/model/batch/v1',
                'input_files': os.path.dirname(__file__) + '/../data/train_sample_2.csv'}
        env_path = None

        input_tb = None
        output_schema = None

        tf_config = TFConfig(work_num, ps_num, prop, python_file, func, env_path)

        train(stream_env, table_env, statement_set, input_tb, tf_config, output_schema)

        job_client = statement_set.execute().get_job_client()
        if job_client is not None:
            job_client.get_job_execution_result().result()

    @staticmethod
    def stream_train():
        stream_env = StreamExecutionEnvironment.get_execution_environment()
        table_env = StreamTableEnvironment.create(stream_env)
        statement_set = table_env.create_statement_set()

        def input_table():
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
            return table_env.from_path('raw_input')

        work_num = 2
        ps_num = 1
        python_file = "model_trainer.py"
        func = "stream_train"
        prop = {MLCONSTANTS.PYTHON_VERSION: '3.7',
                MLCONSTANTS.USE_DISTRIBUTE_CACHE: 'false',
                MLCONSTANTS.CONFIG_STORAGE_TYPE: MLCONSTANTS.STORAGE_ZOOKEEPER,
                MLCONSTANTS.CONFIG_ZOOKEEPER_CONNECT_STR: 'localhost:2181',
                MLCONSTANTS.REMOTE_CODE_ZIP_FILE: 'file:///tmp/code.zip',
                MLCONSTANTS.ENCODING_CLASS: 'com.alibaba.flink.ml.operator.coding.RowCSVCoding',
                MLCONSTANTS.DECODING_CLASS: 'com.alibaba.flink.ml.operator.coding.RowCSVCoding',
                "sys:csv_encode_types": 'STRING',
                "sys:csv_decode_types": 'STRING',
                'checkpoint_dir': '/tmp/model/stream/v1'}

        env_path = None

        input_tb = input_table()
        output_schema = None

        tf_config = TFConfig(work_num, ps_num, prop, python_file, func, env_path)

        train(stream_env, table_env, statement_set, input_tb, tf_config, output_schema)

        job_client = statement_set.execute().get_job_client()
        if job_client is not None:
            job_client.get_job_execution_result().result()


if __name__ == '__main__':
    if os.path.exists('code.zip'):
        os.remove('code.zip')
    if os.path.exists('temp'):
        shutil.rmtree('temp')
    subprocess.call('zip -r code.zip code && mv code.zip /tmp/', shell=True)
    # TrainJob.batch_train()
    TrainJob.stream_train()
