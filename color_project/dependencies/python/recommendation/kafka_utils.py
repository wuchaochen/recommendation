#
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
#
import os
import sys
import time
import uuid

from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import NewTopic
from recommendation import config


class KafkaUtils(object):
    def __init__(self, broker='localhost:9092'):
        super().__init__()
        self.bootstrap_servers = broker
        self.admin_client = KafkaAdminClient(bootstrap_servers=self.bootstrap_servers)

    def send_data_loop(self, file_path, topic_name,  max_epoch=None, interval=1):
        raw_data = []
        with open(file=file_path, mode='r') as f:
            for line in f.readlines():
                raw_data.append(line[:-1])
        producer = KafkaProducer(bootstrap_servers=[self.bootstrap_servers])
        num = 0
        epoch = 0
        if max_epoch is None:
            max_epoch = sys.maxsize
        while epoch < max_epoch:
            epoch += 1
            for line in raw_data:
                num += 1
                producer.send(topic_name,
                              value=bytes(line, encoding='utf8'))
                if 0 == num % 1000:
                    print("send data {}".format(num))
                    time.sleep(interval)

    def _clean_create(self, new_topic, topics):
        if new_topic in topics:
            self.admin_client.delete_topics(topics=[new_topic], timeout_ms=5000)
            print("{} is deleted.".format(new_topic))
            time.sleep(5)
        self.admin_client.create_topics(
            new_topics=[NewTopic(name=new_topic, num_partitions=1, replication_factor=1)])

    def create_topic(self, topic_name):
        topics = self.admin_client.list_topics()
        print(topics)
        self._clean_create(topic_name, topics)

    def read_data(self, topic, count=None, offset='earliest'):
        consumer = KafkaConsumer(topic, bootstrap_servers=[self.bootstrap_servers], group_id=str(
            uuid.uuid1()), auto_offset_reset=offset)
        num = 0
        result = []
        if count is None:
            count = sys.maxsize
        for message in consumer:
            num += 1
            result.append(message.value.decode("utf-8"))
            if num >= count:
                break
        return result

    def read_data_into_file(self, topic, filepath, count=None):
        consumer = KafkaConsumer(topic, bootstrap_servers=[self.bootstrap_servers], group_id=str(
            uuid.uuid1()), auto_offset_reset='earliest')
        num = 0
        if count is None:
            count = sys.maxsize
        with open(filepath, "wb") as f:
            for message in consumer:
                num += 1
                f.write(message.value)
                f.write(b'\n')
                if num >= count:
                    break

    def delete_topic(self, topic_name):
        topics = self.admin_client.list_topics()
        print(topics)
        if topic_name in topics:
            self.admin_client.delete_topics(topics=[topic_name], timeout_ms=5000)
            print("{} is deleted.".format(topic_name))
            time.sleep(5)
        topics = self.admin_client.list_topics()
        print(topics)


def init():
    kafka_util = KafkaUtils()
    kafka_util.create_topic(config.RawQueueName)
    kafka_util.create_topic(config.SampleQueueName)
    if not os.path.exists(config.ModelDir):
        os.makedirs(config.ModelDir)
    if not os.path.exists(config.BaseModelDir):
        os.makedirs(config.BaseModelDir)
    if not os.path.exists(config.TrainModelDir):
        os.makedirs(config.TrainModelDir)
    if not os.path.exists(config.DataDir):
        os.makedirs(config.DataDir)
    if not os.path.exists(config.SampleFileDir):
        os.makedirs(config.SampleFileDir)


if __name__ == '__main__':
    init()
