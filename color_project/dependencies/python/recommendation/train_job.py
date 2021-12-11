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
import glob
import os
import shutil
import subprocess

from flink_ml_tensorflow.tensorflow_TFConfig import TFConfig
from flink_ml_tensorflow.tensorflow_on_flink_ml import Tensorflow
from flink_ml_tensorflow.tensorflow_on_flink_mlconf import MLCONSTANTS
from pyflink.datastream.stream_execution_environment import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, DataTypes

from recommendation import config
from recommendation.config import SampleFileDir, TrainModelDir, BatchModelDir


class TrainJob(object):
    @staticmethod
    def batch_train(base_model_save_dir, sample_dir, max_step=3000):
        stream_env = StreamExecutionEnvironment.get_execution_environment()
        table_env = StreamTableEnvironment.create(stream_env)
        statement_set = table_env.create_statement_set()

        sample_files = glob.glob(os.path.join(sample_dir, "*"))

        work_num = 2
        ps_num = 1
        python_file = "model_trainer.py"
        func = "batch_train"
        prop = {MLCONSTANTS.PYTHON_VERSION: '3.7',
                MLCONSTANTS.USE_DISTRIBUTE_CACHE: 'false',
                MLCONSTANTS.CONFIG_STORAGE_TYPE: MLCONSTANTS.STORAGE_ZOOKEEPER,
                MLCONSTANTS.CONFIG_ZOOKEEPER_CONNECT_STR: 'localhost:2181',
                MLCONSTANTS.REMOTE_CODE_ZIP_FILE: 'file:///tmp/code.zip',
                'checkpoint_dir': '/tmp/model/batch',
                'model_save_path': base_model_save_dir,
                'max_step': str(max_step),
                'batch_model_name': config.BatchModelName,
                'input_files': ",".join(sample_files)}
        env_path = None

        tf_config = TFConfig(work_num, ps_num, prop, python_file, func, env_path)

        tensorflow = Tensorflow(tf_config, ["top_1_indices", "top_1_values"], [DataTypes.STRING(), DataTypes.STRING()],
                                table_env, statement_set)
        model = tensorflow.fit()

        model.statement_set.execute().wait()

    @staticmethod
    def stream_train(base_model_checkpoint_dir, stream_model_dir, kafka_broker, topic):
        stream_env = StreamExecutionEnvironment.get_execution_environment()
        table_env = StreamTableEnvironment.create(stream_env)
        statement_set = table_env.create_statement_set()

        def input_table():
            table_env.execute_sql(f'''
                        create table sample_input (
                            record varchar
                        ) with (
                            'connector' = 'kafka',
                            'topic' = '{topic}',
                            'properties.bootstrap.servers' = '{kafka_broker}',
                            'properties.group.id' = '{topic}',
                            'format' = 'csv',
                            'csv.field-delimiter' = '|',
                            'scan.startup.mode' = 'latest-offset'
                        )
                    ''')
            return table_env.from_path('sample_input')

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
                'stream_model_name': config.StreamModelName,
                'checkpoint_dir': '/tmp/model/stream/v1',
                'base_model_checkpoint': base_model_checkpoint_dir,
                'model_save_path': stream_model_dir}

        env_path = None

        input_tb = input_table()

        tf_config = TFConfig(work_num, ps_num, prop, python_file, func, env_path)

        tensorflow = Tensorflow(tf_config, ["top_1_indices", "top_1_values"], [DataTypes.STRING(), DataTypes.STRING()],
                                table_env=table_env,
                                statement_set=statement_set)
        model = tensorflow.fit(input_tb)

        model.statement_set.execute().wait()


if __name__ == '__main__':

    if os.path.exists('code.zip'):
        os.remove('code.zip')
    if os.path.exists('temp'):
        shutil.rmtree('temp')
    subprocess.call('zip -r code.zip code && mv code.zip /tmp/', shell=True)
    # model_dir = '/tmp/model/batch'
    # if os.path.exists(model_dir):
    #     shutil.rmtree(model_dir)
    # TrainJob.batch_train('/tmp/model/test/1', '/tmp/data/sample_1')

    base_model_dir = '/tmp/model/test/1/20211211135301'
    stream_model_dir = '/tmp/model/stream'
    if os.path.exists(stream_model_dir):
        shutil.rmtree(stream_model_dir)
    TrainJob.stream_train(base_model_checkpoint_dir=base_model_dir,
                          stream_model_dir=config.StreamModelDir,
                          kafka_broker="localhost:9092",
                          topic="sample_input")
