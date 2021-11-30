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


class SimpleAgent(object):
    def __init__(self,
                 topic_name,
                 sample_data_1,
                 epoch_1,
                 sample_data_2,
                 epoch_2,
                 interval=1
                 ):
        self.sample_data_1 = sample_data_1
        self.epoch_1 = epoch_1
        self.sample_data_2 = sample_data_2
        self.epoch_2 = epoch_2
        self.topic_name = topic_name
        self.interval = interval

    def send_data_to_raw_input(self):
        kafka_util = kafka_utils.KafkaUtils()

        kafka_util.send_data_loop(file_path=self.sample_data_1,
                                  topic_name=self.topic_name,
                                  max_epoch=self.epoch_1,
                                  interval=self.interval)

        kafka_util.send_data_loop(file_path=self.sample_data_2,
                                  topic_name=self.topic_name,
                                  max_epoch=self.epoch_2,
                                  interval=self.interval)


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/')
    agent = SimpleAgent(topic_name='raw_input',
                        sample_data_1=data_dir + 'train_sample_1.csv',
                        epoch_1=2,
                        sample_data_2=data_dir + 'train_sample_2.csv',
                        epoch_2=2,
                        interval=10)
    agent.send_data_to_raw_input()

