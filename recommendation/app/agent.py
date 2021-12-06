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

import grpc
from kafka import KafkaProducer

from recommendation.inference_service import ModelInference
from recommendation.proto.service_pb2 import RecordRequest, RecordResponse
from recommendation.proto.service_pb2_grpc import AgentServiceServicer, AgentServiceStub, \
    add_AgentServiceServicer_to_server
from recommendation.inference_client import InferenceClient
from recommendation import db


class Agent(object):
    def __init__(self, user_count, checkpoint_dir, topic):
        self.user_count = user_count
        self.mi = ModelInference(checkpoint_dir)
        self.producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
        self.topic = topic

    def request(self):
        random.seed(time.time_ns())
        return random.randint(0, self.user_count - 1)

    def click(self, record):
        return self.mi.inference_click(record=record)[0]

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
            time.sleep(1)

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
    def __init__(self, checkpoint_dir):
        self.agent = Agent(user_count=100, checkpoint_dir=checkpoint_dir, topic='raw_input')
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
        self.server.stop()


if __name__ == '__main__':
    agent_model_dir = '/tmp/model/2'
    as_ = AgentServer(agent_model_dir)
    as_.start()

    # agent = Agent(user_count=100, checkpoint_dir=agent_model_dir)
    # res = agent.click(['82 13 8,14,27,49,107,110 -1 29,60,65,79,86,102 60'])
    # print(res)