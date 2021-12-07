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
from color_processor.validate_processor import BatchValidateProcessor, StreamValidateProcessor
from color_processor.sample_processor import SampleProcessor


def workflow():
    af.init_ai_flow_context()
    with af.job_config(job_name='batch_validate'):
        af.user_define_operation(input=None, processor=BatchValidateProcessor())

    with af.job_config(job_name='stream_validate'):
        af.user_define_operation(input=None, processor=StreamValidateProcessor())

    with af.job_config(job_name='data_process'):
        af.user_define_operation(input=None, processor=SampleProcessor())

    # Run workflow
    af.workflow_operation.stop_all_workflow_executions(af.current_workflow_config().workflow_name)
    af.workflow_operation.submit_workflow(af.current_workflow_config().workflow_name)
    workflow_execution = af.workflow_operation.start_new_workflow_execution(af.current_workflow_config().workflow_name)
    print(workflow_execution)


if __name__ == '__main__':
    workflow()
