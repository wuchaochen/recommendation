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
import queue
import random
import time
import tensorflow as tf
from recommendation.data import SampleData
from recommendation.code.r_model import RecommendationModel

threshold = 0.3


class ModelInference(object):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        m = RecommendationModel(colour_count=128, recommend_num=6, user_count=100, country_count=20)
        record = tf.placeholder(dtype=tf.string, name='record', shape=[None])

        def multiply_split(value):
            tmp = tf.strings.to_number(tf.sparse.to_dense(tf.string_split(value, sep=',')), tf.int32)
            return tmp
            # return tf.squeeze(tmp)

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
        if isinstance(indices_res, list):
            for i in range(len(indices_res)):
                if values_res[i] > threshold:
                    results.append(indices_res[i])
                else:
                    results.append(-1)
        else:
            if values_res > threshold:
                results.append(indices_res)
            else:
                results.append(-1)
        return results

    def close(self):
        self.mon_sess.close()


class Agent(object):
    def __init__(self, user_count, checkpoint_dir):
        self.user_count = user_count
        self.mi = ModelInference(checkpoint_dir)

    def request(self):
        random.seed(time.time_ns())
        return random.randint(0, self.user_count - 1)

    def click(self, record):
        return self.mi.inference_click(record=record)

    def close(self):
        self.mi.close()


class InferenceService(object):
    def __init__(self):
        self.user_dict = SampleData.load_user_dict()
        self.agent = Agent(100)
        self.queue = queue.Queue(maxsize=500)
        self.user_cache = {}  # uid: [(recommend_colours_1, click_1), (recommend_colours_2, click_2)]

    def start_agent(self):
        pass

    def build_features(self, uid):
        pass

    def inference(self, features):
        pass

    def write_log(self):
        pass

    def update_state(self):
        pass

    def run(self):
        self.start_agent()
        while True:
            uid = self.queue.get()
            features = self.build_features(uid)
            inference_result = self.inference(features)
            click_result = self.agent.click(inference_result)
            self.update_state()
            self.write_log()


if __name__ == '__main__':
    agent_model_dir = '/tmp/model/batch/v1'
    inference_model_dir = '/tmp/model/stream/v1'
    agent = Agent(100, agent_model_dir)
    record = ['88 5 12,36,53,58,103,115 115 53,58,68,103,106,115 103']
    res = agent.click(record)
    print(res)
    agent.close()
    # print(agent.request())
    # records = ['88 5 12,36,53,58,103,115 115 53,58,68,103,106,115 103',
    #            '73 17 8,66,76,83,109,120 8 3,8,32,76,83,109 8',
    #            '73 17 8,66,76,83,109,120 8 3,8,32,76,83,109 8']
    # inference = ModelInference(agent_model_dir)
    # result = inference.inference_click(record=records)
    # print(result)
    # inference.close()
