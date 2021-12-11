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
BaseDir = '/Users/chenwuchao/tmp/rc_root'
ModelDir = BaseDir + '/model'
BaseModelDir = ModelDir + '/base'
TrainModelDir = ModelDir + '/train'
BatchModelDir = TrainModelDir + '/batch'
StreamModelDir = TrainModelDir + '/stream'
TempModelDir = ModelDir + '/temp'
BatchTempModelDir = TempModelDir + '/batch'
StreamTempModelDir = TempModelDir + '/stream'

AgentModelDir = BaseModelDir + '/1'
InferenceModelDir = BaseModelDir + '/1'

BatchModelName = 'batch_color_model'
StreamModelName = 'stream_color_model'

BatchACC = 'batch_acc'
StreamACC = 'stream_acc'

threshold = 0.1

val_threshold = 0.9

DataDir = BaseDir + '/data'
TestDataDir = DataDir + '/test/'
UserDictFile = DataDir + '/users.csv'
SampleFileDataset = 'File:sample_dataset'
SampleFileDir = DataDir + '/samples'
ValidateFileDir = DataDir + '/validate'
ValidateDataset = "File:validate_dataset"

user_count = 100
color_count = 32
