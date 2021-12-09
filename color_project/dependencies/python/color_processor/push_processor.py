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

from recommendation import config


class ModelPushProcessor(python.PythonProcessor):
    def process(self, execution_context: ExecutionContext, input_list: List) -> List:
        model_info = execution_context.config['model_info']
        model_name = model_info.name
        latest_validated_model_version = af.get_latest_validated_model_version(model_name)
        deployed_model_version = af.get_deployed_model_version(model_name)

        if not latest_validated_model_version:
            print("No validated model version for model {} is found".format(model_name))
            return []

        if not deployed_model_version:
            af.update_model_version(model_name=model_name,
                                    model_version=latest_validated_model_version.version,
                                    current_stage=af.ModelVersionStage.DEPLOYED)
            return []

        if latest_validated_model_version.version < deployed_model_version.version:
            # This is a staled validated model version
            print("The validated model version is staled, validated version: {}, deployed version: {}".format(
                latest_validated_model_version.version, deployed_model_version.version))
            return []

        af.update_model_version(model_name=config.StreamModelName,
                                model_version=deployed_model_version.version,
                                current_stage=af.ModelVersionStage.DEPRECATED)
        af.update_model_version(model_name=config.StreamModelName,
                                model_version=latest_validated_model_version.version,
                                current_stage=af.ModelVersionStage.DEPLOYED)
        return []
