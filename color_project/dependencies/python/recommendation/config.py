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
from recommendation import kafka_utils


KafkaConn = 'localhost:9092'

UserProfileDataset = "MYSQL:user_profile"
UserProfileTableName = "user"
UserClickDataset = "MYSQL:user_click_history"
UserClickTableName = "user_click"

RawQueueDataset = "Kafka:raw_input"
RawQueueName = 'raw_input'
SampleQueueDataset = "Kafka:sample_input"
SampleQueueName = 'sample_input'

DbUserName = 'root'
DbPassword = 'root'
DbConn = 'mysql://{}:{}@localhost:3306/user_info'.format(DbUserName, DbPassword)

ModelDir = '/tmp/model'
BaseModelDir = ModelDir + '/base'
TrainModelDir = ModelDir + '/train'
BatchModelDir = TrainModelDir + '/batch'
StreamModelDir = TrainModelDir + '/stream'

DataDir = '/tmp/data'
UserDictFile = DataDir + '/users.csv'
SampleFileDataset = 'File:sample_dataset'
SampleFileDir = DataDir + '/samples'
ValidateFileDir = DataDir + '/validate'
ValidateFilePath = ValidateFileDir + '/train_sample_1.csv'
ValidateDataset = "File:validate_dataset"

BatchModelName = 'batch_color_model'
StreamModelName = 'stream_color_model'

BatchACC = 'batch_acc'
StreamACC = 'stream_acc'

AgentModelDir = '/tmp/model/base/2'
InferenceModelDir = '/tmp/model/base/2'

threshold = 0.1

BatchTrainTime = 5
BatchValidateTime = 1
StreamTrainTime = 10
StreamValidateTime = 1



def init():
    kafka_util = kafka_utils.KafkaUtils()
    kafka_util.create_topic(RawQueueName)
    kafka_util.create_topic(SampleQueueName)
    if not os.path.exists(ModelDir):
        os.makedirs(ModelDir)
    if not os.path.exists(BaseModelDir):
        os.makedirs(BaseModelDir)
    if not os.path.exists(TrainModelDir):
        os.makedirs(TrainModelDir)
    if not os.path.exists(DataDir):
        os.makedirs(DataDir)
    if not os.path.exists(SampleFileDir):
        os.makedirs(SampleFileDir)


if __name__ == '__main__':
    init()
