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
import ai_flow as af

from ai_flow_plugins.job_plugins import python
from ai_flow_plugins.job_plugins.python.python_processor import ExecutionContext
from typing import List

from recommendation import train_job, config


class BatchTrainDataReader(python.PythonProcessor):

    def process(self, execution_context: ExecutionContext, input_list: List) -> List:
        return [execution_context.config.get('dataset').uri]


class BatchTrainProcessor(python.PythonProcessor):
    def __init__(self, max_step=2000):
        self.max_step = max_step

    def process(self, execution_context: ExecutionContext, input_list: List) -> List:
        sample_dir = input_list[0]
        train_job.TrainJob.batch_train(config.BatchModelDir, sample_dir, self.max_step)
        return []


class StreamTrainDataReader(python.PythonProcessor):

    def process(self, execution_context: ExecutionContext, input_list: List) -> List:
        return [execution_context.config.get('dataset').uri]


class StreamTrainProcessor(python.PythonProcessor):

    def process(self, execution_context: ExecutionContext, input_list: List) -> List:
        af.init_ai_flow_client("localhost:50051", "color_project", notification_server_uri="localhost:50052")

        broker_ip, topic = input_list[0].split(",")
        base_model_info = execution_context.config.get('base_model_info')

        base_model_version_meta = af.get_latest_generated_model_version(base_model_info.name)
        if not base_model_version_meta:
            raise RuntimeError("Cannot found latest generated model version of model {}".format(base_model_info))
        train_job.TrainJob.stream_train(base_model_version_meta.model_path, config.StreamModelDir, broker_ip, topic)
        return []
