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

import ai_flow as af
from ai_flow_plugins.job_plugins import flink
from ai_flow_plugins.job_plugins.flink.flink_processor import ExecutionContext
from ai_flow_plugins.job_plugins.flink.flink_wrapped_env import WrappedStatementSet, WrappedTableEnvironment
from flink_ml_tensorflow.tensorflow_TFConfig import TFConfig
from flink_ml_tensorflow.tensorflow_on_flink_ml import Tensorflow
from flink_ml_tensorflow.tensorflow_on_flink_mlconf import MLCONSTANTS
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import DataTypes
from pyflink.table import StreamTableEnvironment
from pyflink.table import Table
from typing import List
from typing import Tuple

from recommendation import config


class TrainFlinkEnv(flink.FlinkEnv):

    def create_env(self) -> Tuple[WrappedTableEnvironment, WrappedStatementSet]:
        _env = StreamExecutionEnvironment.get_execution_environment()
        _t_env = StreamTableEnvironment.create(_env)
        _t_env.get_config().get_configuration().set_string("parallelism.default", "1")
        _t_env.get_config().get_configuration().set_string("taskmanager.memory.task.off-heap.size", '80m')
        _t_env.get_config().get_configuration().set_string("execution.checkpointing.interval", "30sec")
        t_env = WrappedTableEnvironment.create_from(_t_env)
        statement_set = t_env.create_statement_set()

        return t_env, statement_set


class BatchTrainDataReader(flink.FlinkPythonProcessor):

    def process(self, execution_context: ExecutionContext, input_list: List[Table] = None) -> List[Table]:
        return [execution_context.config.get('dataset').uri]


class BatchTrainProcessor(flink.FlinkPythonProcessor):
    def __init__(self, max_step=2000):
        self.max_step = max_step

    def process(self, execution_context: ExecutionContext, input_list: List[Table] = None) -> List[Table]:
        sample_dir: str = input_list[0]

        table_env = execution_context.table_env
        statement_set = execution_context.statement_set

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
                'model_save_path': config.BatchModelDir,
                'max_step': str(self.max_step),
                'batch_model_name': config.BatchModelName,
                'input_files': ",".join(sample_files)}
        env_path = None

        tf_config = TFConfig(work_num, ps_num, prop, python_file, func, env_path)

        tensorflow = Tensorflow(tf_config, ["top_1_indices", "top_1_values"], [DataTypes.STRING(), DataTypes.STRING()],
                                table_env, statement_set)

        tensorflow.fit()
        execution_context.statement_set.wrapped_context.need_execute = True

        return []


class StreamTrainDataReader(flink.FlinkPythonProcessor):

    def process(self, execution_context: ExecutionContext, input_list: List[Table] = None) -> List[Table]:
        return [execution_context.config.get('dataset').uri]


class StreamTrainProcessor(flink.FlinkPythonProcessor):

    def process(self, execution_context: ExecutionContext, input_list: List[Table] = None) -> List[Table]:
        while True:
            try:
                af.init_ai_flow_client("localhost:50051", "color_project", notification_server_uri="localhost:50052")
                client = af.get_ai_flow_client()
                break
            except Exception:
                pass

        broker_ip, topic = input_list[0].split(",")
        base_model_info = execution_context.config.get('base_model_info')

        base_model_version_meta = client.get_latest_generated_model_version(base_model_info.name)
        if not base_model_version_meta:
            raise RuntimeError("Cannot found latest generated model version of model {}".format(base_model_info))

        table_env = execution_context.table_env
        statement_set = execution_context.statement_set

        table_env.execute_sql(f'''
                    create table raw_input (
                        record varchar
                    ) with (
                        'connector' = 'kafka',
                        'topic' = '{topic}',
                        'properties.bootstrap.servers' = '{broker_ip}',
                        'properties.group.id' = '{topic}',
                        'format' = 'csv',
                        'csv.field-delimiter' = '|',
                        'scan.startup.mode' = 'latest-offset'
                    )
                ''')

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
                'base_model_checkpoint': base_model_version_meta.model_path,
                'model_save_path': config.StreamModelDir}

        env_path = None
        input_tb = table_env.from_path('raw_input')

        tf_config = TFConfig(work_num, ps_num, prop, python_file, func, env_path)

        tensorflow = Tensorflow(tf_config, ["top_1_indices", "top_1_values"], [DataTypes.STRING(), DataTypes.STRING()],
                                table_env=table_env,
                                statement_set=statement_set)
        tensorflow.fit(input_tb)
        execution_context.statement_set.wrapped_context.need_execute = True

        return []
