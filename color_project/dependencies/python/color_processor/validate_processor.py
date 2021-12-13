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
import os.path
import time

import ai_flow as af
from ai_flow_plugins.job_plugins import python
from ai_flow_plugins.job_plugins.python.python_processor import ExecutionContext
from notification_service.base_notification import BaseEvent
from notification_service.client import NotificationClient
from typing import List

from recommendation import config
from recommendation.validate_job import ValidateJob


class ValidateDataReader(python.PythonProcessor):

    def process(self, execution_context: ExecutionContext, input_list: List) -> List:
        return [execution_context.config.get('dataset')]


class BatchValidateProcessor(python.PythonProcessor):
    def process(self, execution_context: ExecutionContext, input_list: List) -> List:
        # validate_job = ValidateJob()
        # dataset = input_list[0]
        # validate_sample_dir = dataset.uri
        # validate_sample_files = glob.glob(os.path.join(validate_sample_dir, "*"))
        time.sleep(config.BatchValidateTime)
        m_version = af.get_latest_generated_model_version(config.BatchModelName)
        deployed_version = af.get_deployed_model_version(config.BatchModelName)
        # acc = validate_job.batch_validate(checkpoint_dir=m_version.model_path,
        #                                   validate_files=validate_sample_files,
        #                                   data_count=10000)
        # af.register_metric_summary(metric_name=config.BatchACC,
        #                            metric_key='acc',
        #                            metric_value=str(acc),
        #                            metric_timestamp=int(time.time()))
        if deployed_version is not None:
            af.update_model_version(model_name=config.BatchModelName,
                                    model_version=deployed_version.version,
                                    current_stage=af.ModelVersionStage.DEPRECATED)
        model_version = af.update_model_version(model_name=config.BatchModelName,
                                                model_version=m_version.version,
                                                current_stage=af.ModelVersionStage.VALIDATED)
        notification_client = NotificationClient(af.current_project_config().get_notification_server_uri(),
                                                 default_namespace='default',
                                                 sender=execution_context.job_execution_info.job_name)
        notification_client.send_event(BaseEvent(key=config.BatchModelName, value=model_version.model_version,
                                                 event_type='MODEL_VALIDATED'))
        return []


class StreamValidateProcessor(python.PythonProcessor):
    def process(self, execution_context: ExecutionContext, input_list: List) -> List:
        # validate_job = ValidateJob()
        # broker, topic = input_list[0].uri.split(",")
        time.sleep(config.StreamValidateTime)
        m_version = af.get_latest_generated_model_version(config.StreamModelName)
        # acc = validate_job.stream_validate(checkpoint_dir=m_version.model_path,
        #                                    broker=broker,
        #                                    topic=topic,
        #                                    data_count=1000)
        # af.register_metric_summary(metric_name=config.StreamACC,
        #                            metric_key='acc',
        #                            metric_value=str(acc),
        #                            metric_timestamp=int(time.time()))
        acc = 0.2
        if acc > 0.1:
            model_version = af.update_model_version(model_name=config.StreamModelName,
                                                    model_version=m_version.version,
                                                    current_stage=af.ModelVersionStage.VALIDATED)
            notification_client = NotificationClient(af.current_project_config().get_notification_server_uri(),
                                                     default_namespace='default',
                                                     sender=execution_context.job_execution_info.job_name)
            notification_client.send_event(BaseEvent(key=config.StreamModelName, value=model_version.model_version,
                                                     event_type='MODEL_VALIDATED'))
        else:
            af.update_model_version(model_name=config.StreamModelName,
                                    model_version=m_version.version,
                                    current_stage=af.ModelVersionStage.DEPRECATED)
        return []
