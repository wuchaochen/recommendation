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
import time
from typing import List
import ai_flow as af
from ai_flow_plugins.job_plugins import python
from ai_flow_plugins.job_plugins.python.python_processor import ExecutionContext
from recommendation.validate_job import ValidateJob
from recommendation import config


class BatchValidateProcessor(python.PythonProcessor):
    def process(self, execution_context: ExecutionContext, input_list: List) -> List:
        validate_job = ValidateJob()
        m_version = af.get_latest_generated_model_version(config.BatchModelName)
        acc = validate_job.batch_validate(checkpoint_dir=m_version.model_path,
                                          validate_files=config.ValidateFilePath,
                                          data_count=10000)
        af.register_metric_summary(metric_name=config.BatchACC,
                                   metric_key='acc',
                                   metric_value=str(acc),
                                   metric_timestamp=int(time.time()))
        af.update_model_version(model_name=config.BatchModelName,
                                model_version=m_version.version,
                                current_stage=af.ModelVersionStage.DEPLOYED)
        return []


class StreamValidateProcessor(python.PythonProcessor):
    def process(self, execution_context: ExecutionContext, input_list: List) -> List:
        validate_job = ValidateJob()
        m_version = af.get_latest_generated_model_version(config.StreamModelName)
        acc = validate_job.stream_validate(checkpoint_dir=m_version.model_path,
                                           topic=config.SampleQueueName,
                                           data_count=10000)
        af.register_metric_summary(metric_name=config.StreamACC,
                                   metric_key='acc',
                                   metric_value=str(acc),
                                   metric_timestamp=int(time.time()))
        if acc > 0.3:
            af.update_model_version(model_name=config.StreamModelName,
                                    model_version=m_version.version,
                                    current_stage=af.ModelVersionStage.DEPLOYED)
        else:
            af.update_model_version(model_name=config.StreamModelName,
                                    model_version=m_version.version,
                                    current_stage=af.ModelVersionStage.DEPRECATED)
        return []
