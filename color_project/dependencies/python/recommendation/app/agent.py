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
import threading
import time
import random
from typing import List

from kafka import KafkaProducer
from notification_service.base_notification import EventWatcher, BaseEvent
from notification_service.client import NotificationClient
from recommendation.inference_service import ModelInference
from recommendation.inference_client import InferenceClient
from recommendation import db
from recommendation import config


class UpdateModel(EventWatcher):
    def __init__(self, agent):
        self.agent = agent

    def process(self, events: List[BaseEvent]):
        event = events[0]
        try:
            print(event.value)
            self.agent.lock.acquire()
            self.agent.mi = ModelInference(checkpoint_dir=event.value)
        finally:
            self.agent.lock.release()


class Agent(object):
    def __init__(self, user_count, checkpoint_dir, topic, interval=0.1, batch_size=100, inference_uri='localhost:30002'):
        self.user_count = user_count
        self.mi = ModelInference(checkpoint_dir)
        self.producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
        self.topic = topic
        self.inference_uri = inference_uri
        self.ns_client = NotificationClient(server_uri='localhost:50052')
        self.interval = interval
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.ns_client.start_listen_event(key='update_agent', watcher=UpdateModel(self), event_type='update_agent',
                                          start_time=int(time.time() * 1000))

    def random_user(self):
        random.seed(time.time_ns())
        return random.randint(0, self.user_count - 1)

    def click(self, record):
        try:
            self.lock.acquire()
            return self.mi.inference_click(record=record)
        finally:
            self.lock.release()

    def write_log(self, uid, inference_result, click_result):
        self.producer.send(topic=self.topic, value=bytes(str(uid) + ' ' + inference_result + ' ' + str(click_result),
                                                         encoding='utf8'))

    def update_state(self, uid, inference_result, click_result):
        db.update_user_click_info(uid=uid, fs=inference_result + ' ' + str(click_result))

    def close(self):
        self.mi.close()

    def build_features(self, uids):
        users_info_dict = {}
        users_info = db.get_users_info(uids)
        for i in users_info:
            users_info_dict[i.uid] = i

        users_click_dict = {}
        users_click = db.get_users_click_info(uids)
        for i in users_click:
            users_click_dict[i.uid] = i

        records = []
        for i in range(self.batch_size):
            record = []
            record.append(uids[i])
            record.append(users_info_dict[uids[i]].country)
            record.append(users_click_dict[uids[i]].fs_1)
            record.append(users_click_dict[uids[i]].fs_2)
            records.append(' '.join(map(lambda x: str(x), record)))
        return records

    def action(self):
        client = InferenceClient(self.inference_uri)
        count = 0
        start_time = time.monotonic()
        while True:
            count += 1
            uids = []
            for i in range(self.batch_size):
                uid = self.random_user()
                uids.append(uid)
            colors_results = client.inference(uids)
            batch_colors = []
            batch_features = self.build_features(uids)
            for i in range(self.batch_size):
                colors = set(map(int, colors_results[i].split(',')))
                batch_colors.append(colors)
            click_results = self.click(batch_features)
            for i in range(self.batch_size):
                if click_results[i] not in batch_colors[i]:
                    click_result = -1
                else:
                    click_result = click_results[i]
                self.write_log(uid=uids[i], inference_result=colors_results[i], click_result=click_result)
            time.sleep(self.interval)
            if count % 10 == 0:
                end_time = time.monotonic()
                print('{} records/sec'.format((10*self.batch_size)/(end_time-start_time)))
                start_time = end_time

    def start(self):
        thread = threading.Thread(target=self.action, args=())
        thread.setDaemon(True)
        thread.start()


if __name__ == '__main__':
    db.init_db(config.DbConn)
    agent_model_dir = config.AgentModelDir
    as_ = Agent(user_count=100,
                checkpoint_dir=agent_model_dir,
                topic=config.RawQueueName,
                interval=0,
                batch_size=500,
                inference_uri='localhost:30002')
    as_.action()