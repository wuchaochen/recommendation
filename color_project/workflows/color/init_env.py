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

import ai_flow as af
from ai_flow.context.project_context import init_project_config
from ai_flow.api.ai_flow_context import ensure_project_registered
from recommendation import config


def get_project_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def init():
    af.register_dataset(name=config.RawQueueName, uri="{},{}".format(config.KafkaConn, config.RawQueueName))
    af.register_dataset(name=config.SampleFileName, uri=config.SampleFileDir)
    af.register_dataset(name=config.SampleQueueName, uri="{},{}".format(config.KafkaConn, config.SampleQueueName))
    af.register_dataset(name=config.ValidateDataset, uri=config.ValidateFilePath)
    af.register_dataset(name=config.UserProfileDataset, uri=config.UserProfileTableName)
    af.register_dataset(name=config.UserClickDataset, uri=config.UserClickTableName)

    af.register_metric_meta(metric_name=config.BatchACC,
                            metric_type=af.MetricType.MODEL,
                            project_name=af.current_project_config().get_project_name())

    af.register_metric_meta(metric_name=config.StreamACC,
                            metric_type=af.MetricType.MODEL,
                            project_name=af.current_project_config().get_project_name())

    af.register_model(model_name=config.BatchModelName)
    af.register_model(model_name=config.StreamModelName)


if __name__ == '__main__':
    init_project_config(get_project_path() + '/project.yaml')
    ensure_project_registered()
    init()
