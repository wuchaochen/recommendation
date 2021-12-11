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
import json
import numpy
import threading
import time
from concurrent import futures
import grpc
import tensorflow as tf
from notification_service.base_notification import EventWatcher, BaseEvent
from notification_service.client import NotificationClient
from typing import List

from recommendation.data import SampleData, ColourData
from recommendation.code.r_model import RecommendationModel
from recommendation.proto.service_pb2 import RecordResponse
from recommendation.proto.service_pb2_grpc import InferenceServiceServicer, add_InferenceServiceServicer_to_server
from recommendation import db
from recommendation import config


class ModelInference(object):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        tf.reset_default_graph()
        m = RecommendationModel(colour_count=128, recommend_num=6, user_count=config.user_count, country_count=20)
        record = tf.placeholder(dtype=tf.string, name='record', shape=[None])

        def multiply_split(value):
            tmp = tf.strings.to_number(tf.sparse.to_dense(tf.string_split(value, sep=',')), tf.int32)
            return tmp

        def parse_csv(value):
            columns_ = tf.decode_csv(value,
                                     record_defaults=[0, 0, '0,0,0,0,0,0', 0, '0,0,0,0,0,0', 0], field_delim=' ')
            return columns_[0], columns_[1], multiply_split(columns_[2]), columns_[3], multiply_split(columns_[4]), \
                   columns_[5]

        columns = parse_csv(record)

        features = {'user': columns[0], 'country': columns[1],
                    'recommend_colours_1': columns[2],
                    'click_colour_1': columns[3],
                    'recommend_colours_2': columns[4], 'click_colour_2': columns[5]}
        fs = m.features(features)
        last_layer = m.forward(fs)
        self.top_indices, self.top_values = m.output(last_layer)
        self.top_1_indices, self.top_1_values = m.top_one_output(last_layer)
        global_step = tf.train.get_or_create_global_step()
        print('load checkpoint {}'.format(checkpoint_dir))
        self.mon_sess = tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir)

    def inference(self, record):
        indices_res, values_res = self.mon_sess.run([self.top_indices, self.top_values], feed_dict={'record:0': record})
        results = []
        for r in indices_res:
            r = sorted(r)
            r = ','.join(map(lambda x: str(x), r))
            results.append(r)
        return results

    def inference_click(self, record):
        indices_res, values_res = self.mon_sess.run([self.top_1_indices, self.top_1_values], feed_dict={'record:0': record})
        results = []
        if isinstance(indices_res, list) or isinstance(indices_res, numpy.ndarray):
            for i in range(len(indices_res)):
                if values_res[i] > config.threshold:
                    results.append(indices_res[i])
                else:
                    results.append(-1)
        else:
            if values_res > config.threshold:
                results.append(indices_res)
            else:
                results.append(-1)
        return results

    def close(self):
        self.mon_sess.close()


class DeployModel(EventWatcher):
    def __init__(self, util):
        self.util = util

    def process(self, events: List[BaseEvent]):
        event = events[0]
        try:
            print(event.create_time, event.value)
            try:
                model_path = json.loads(event.value)["_model_path"]
                self.util.lock.acquire()
                self.util.checkpoint_dir = model_path
                self.util.init_model()
            except Exception as e:
                pass
        finally:
            self.util.lock.release()


class InferenceUtil(object):
    def __init__(self, checkpoint_dir):
        self.user_dict = SampleData.load_user_dict()
        self.colour_data = ColourData(count=128, select_count=6)
        self.checkpoint_dir = checkpoint_dir
        self.mi = None
        self.agent_client = None
        self.ns_client = NotificationClient(server_uri='localhost:50052')
        start_time = int(time.time() * 1000)
        print(start_time)
        self.ns_client.start_listen_event(key=config.StreamModelName, watcher=DeployModel(self),
                                          event_type='MODEL_DEPLOYED',
                                          start_time=start_time)
        self.lock = threading.Lock()

    def random_click_record(self):
        c1, c2 = self.colour_data.random_colours()
        return sorted(c1), c2[0]

    def init_model(self):
        self.mi = ModelInference(self.checkpoint_dir)

    @db.provide_session
    def init_user_cache(self, session=None):
        session.query(db.User).delete()
        session.query(db.UserClick).delete()
        for k, v in self.user_dict.items():
            user = db.User()
            user.uid = k
            user.country = v
            session.add(user)
        session.commit()

        for i in range(len(self.user_dict)):
            cc = []
            for j in range(2):
                r, c = self.random_click_record()
                r = ','.join(map(lambda x: str(x), r))
                cc.append((r, c))
            user_click = db.UserClick()
            user_click.uid = i
            user_click.fs_1 = cc[0][0] + ' ' + str(cc[0][1])
            user_click.fs_2 = cc[1][0] + ' ' + str(cc[1][1])
            session.add(user_click)
        session.commit()

    def init(self):
        self.init_model()
        self.init_user_cache()

    def build_features(self, uids):
        users_click_dict = {}
        users_click = db.get_users_click_info(uids)
        for i in users_click:
            users_click_dict[i.uid] = i
        records = []
        for uid in uids:
            record = []
            record.append(uid)
            record.append(self.user_dict[uid])
            record.append(users_click_dict[uid].fs_1)
            record.append(users_click_dict[uid].fs_2)
            records.append(' '.join(map(lambda x: str(x), record)))
        return records

    def inference(self, features):
        try:
            self.lock.acquire()
            return self.mi.inference(features)
        finally:
            self.lock.release()

    def process_request(self, uids):
        batch_feature = self.build_features(uids)
        inference_result = self.inference(batch_feature)
        results = []
        for i in range(len(uids)):
            results.append(inference_result[i] + '*' + batch_feature[i])
        return results


class InferenceService(InferenceServiceServicer):
    def __init__(self, util: InferenceUtil):
        self.util: InferenceUtil = util

    def inference(self, request, context):
        uids = request.uids
        res = self.util.process_request(uids)
        return RecordResponse(records=res)


class InferenceServer(object):
    def __init__(self, checkpoint_dir):
        db.init_db(uri=config.DbConn)
        self.inference_util = InferenceUtil(checkpoint_dir)
        self.inference_util.init_user_cache()
        self.inference_util.init_model()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        add_InferenceServiceServicer_to_server(InferenceService(self.inference_util), self.server)
        self.server.add_insecure_port('[::]:' + str(30002))
        self._stop = threading.Event()

    def start(self):
        self.server.start()
        try:
            while not self._stop.is_set():
                self._stop.wait(3600)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.server.stop(grace=True)


if __name__ == '__main__':
    inference_model_dir = config.InferenceModelDir

    # inference_util = InferenceUtil(inference_model_dir)
    # inference_util.init_user_cache()
    # inference_util.init_model()
    # res = inference_util.build_features(1)
    # res = inference_util.inference([res])
    # print(res)
    is_ = InferenceServer(inference_model_dir)
    is_.start()
