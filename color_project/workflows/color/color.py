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
from ai_flow_plugins.job_plugins import flink

from color_processor.sample_processor import SampleProcessor, DataSourceProcessor, QueueSinkProcessor, \
    FileSinkProcessor, DataStreamEnv
from color_processor.train_procssor import BatchTrainDataReader, BatchTrainProcessor, StreamTrainProcessor, \
    TrainFlinkEnv
from color_processor.validate_processor import BatchValidateProcessor, StreamValidateProcessor
from recommendation import config


def workflow():
    af.init_ai_flow_context()
    with af.job_config(job_name='batch_train'):
        flink.set_flink_env(TrainFlinkEnv())
        data = af.read_dataset(config.SampleFileName, read_dataset_processor=BatchTrainDataReader())
        af.train(data, model_info=config.BatchModelName, training_processor=BatchTrainProcessor(200))

    with af.job_config(job_name='batch_validate'):
        af.user_define_operation(input=None, processor=BatchValidateProcessor())

    with af.job_config(job_name='stream_train'):
        flink.set_flink_env(TrainFlinkEnv())
        data = af.read_dataset(config.SampleQueueName, read_dataset_processor=BatchTrainDataReader())
        af.train(data, model_info=config.StreamModelName, base_model_info=config.BatchModelName,
                 training_processor=StreamTrainProcessor())

    with af.job_config(job_name='stream_validate'):
        af.user_define_operation(input=None, processor=StreamValidateProcessor())

    with af.job_config(job_name='data_process'):
        flink.set_flink_env(DataStreamEnv())
        raw_input = af.read_dataset(dataset_info=config.RawQueueName, read_dataset_processor=DataSourceProcessor())
        sample = af.user_define_operation(input=raw_input, processor=SampleProcessor())
        af.write_dataset(input=sample, dataset_info=config.SampleQueueName,
                         write_dataset_processor=QueueSinkProcessor())
        af.write_dataset(input=sample, dataset_info=config.SampleFileName, write_dataset_processor=FileSinkProcessor())

    af.action_on_job_status("batch_validate", "batch_train")
    af.action_on_model_version_event("stream_train", config.BatchModelName, 'MODEL_GENERATED')
    af.action_on_model_version_event("stream_validate", config.StreamModelName, 'MODEL_GENERATED')

    # Run workflow
    af.workflow_operation.stop_all_workflow_executions(af.current_workflow_config().workflow_name)
    af.workflow_operation.submit_workflow(af.current_workflow_config().workflow_name)
    workflow_execution = af.workflow_operation.start_new_workflow_execution(af.current_workflow_config().workflow_name)
    print(workflow_execution)


if __name__ == '__main__':
    workflow()
