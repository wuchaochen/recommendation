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
from concurrent import futures
from typing import List

import grpc
from kafka import KafkaProducer
from notification_service.base_notification import BaseNotification, EventWatcher, BaseEvent
from notification_service.client import NotificationClient
from recommendation.inference_service import ModelInference
from recommendation.proto.service_pb2 import RecordRequest, RecordResponse
from recommendation.proto.service_pb2_grpc import AgentServiceServicer, AgentServiceStub, \
    add_AgentServiceServicer_to_server
from recommendation.inference_client import InferenceClient
from recommendation import db


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
    def __init__(self, user_count, checkpoint_dir, topic, interval=0.1):
        self.user_count = user_count
        self.mi = ModelInference(checkpoint_dir)
        self.producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
        self.topic = topic
        self.ns_client = NotificationClient(server_uri='localhost:50052')
        self.interval = interval
        self.lock = threading.Lock()
        self.ns_client.start_listen_event(key='update_agent', watcher=UpdateModel(self), event_type='update_agent',
                                          start_time=int(time.time() * 1000))

    def request(self):
        random.seed(time.time_ns())
        return random.randint(0, self.user_count - 1)

    def click(self, record):
        try:
            self.lock.acquire()
            return self.mi.inference_click(record=record)[0]
        finally:
            self.lock.release()

    def write_log(self, uid, inference_result, click_result):
        print(uid, inference_result, click_result)
        self.producer.send(topic=self.topic, value=bytes(str(uid) + ' ' + inference_result + ' ' + str(click_result), encoding='utf8'))

    def update_state(self, uid, inference_result, click_result):
        db.update_user_click_info(uid=uid, fs=inference_result + ' ' + str(click_result))

    def close(self):
        self.mi.close()

    def build_features(self, uid):
        record = []
        record.append(uid)
        user_info = db.get_user_info(uid)
        record.append(user_info.country)
        user_click = db.get_user_click_info(uid)
        record.append(user_click.fs_1)
        record.append(user_click.fs_2)
        return ' '.join(map(lambda x: str(x), record))

    def action(self):
        client = InferenceClient('localhost:30002')
        while True:
            uid = str(self.request())
            res = client.inference(uid)
            features = self.build_features(uid)
            click_result = self.click([features])
            self.write_log(uid=uid, inference_result=res, click_result=click_result)
            time.sleep(self.interval)

    def start(self):
        thread = threading.Thread(target=self.action, args=())
        thread.setDaemon(True)
        thread.start()


class AgentService(AgentServiceServicer):
    def __init__(self, agent: Agent):
        self.agent: Agent = agent

    def click(self, request, context):
        res = self.agent.click([request.record])
        return RecordResponse(record=str(res))


class AgentServer(object):
    def __init__(self, checkpoint_dir, interval):
        self.agent = Agent(user_count=100, checkpoint_dir=checkpoint_dir, topic='raw_input', interval=interval)
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        add_AgentServiceServicer_to_server(AgentService(self.agent), self.server)
        self.server.add_insecure_port('[::]:' + str(30001))
        self._stop = threading.Event()

    def start(self):
        db.init_db(uri='mysql://root:chen@localhost:3306/user_info')
        self.server.start()
        time.sleep(1)
        self.agent.start()
        try:
            while not self._stop.is_set():
                self._stop.wait(3600)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.server.stop(0)


if __name__ == '__main__':
    agent_model_dir = '/tmp/model/1'
    as_ = AgentServer(agent_model_dir, interval=0.1)
    as_.start()