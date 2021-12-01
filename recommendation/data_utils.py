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
import csv
import os
import random
import tensorflow as tf
from recommendation.code.r_model import RecommendationModel, Sample
threshold = 0.3

data_dir = os.path.dirname(__file__) + '/../data/'
base_model_dir = os.path.dirname(__file__) + '/../models/base/'
train_model_dir = os.path.dirname(__file__) + '/../models/train/'
test_model_dir = os.path.dirname(__file__) + '/../models/test/'


class SampleGenerator(object):

    @staticmethod
    def generate_sample(user, country, colour_and_value):
        batch_size = len(user)
        res = []
        colour = colour_and_value[0]
        value = colour_and_value[1]
        for i in range(batch_size):
            record = []
            record.append(user[i])
            record.append(country[i])
            cc = colour[i].tolist()
            cc = sorted(cc)
            cc = ','.join(str(x) for x in cc)
            record.append(cc)
            if value[i][0] > threshold:
                record.append(colour[i][0])
            else:
                record.append(-1)
            record.append(value[i][0])
            res.append(record)
        return res

    @staticmethod
    def generate_label_sample(user, country, colour_1, click_1, colour_2, click_2, indices, values):
        batch_size = len(user)
        res = []
        for i in range(batch_size):
            record = []
            record.append(user[i])
            record.append(country[i])

            cc = colour_1[i].tolist()
            cc = sorted(cc)
            cc = ','.join(str(x) for x in cc)
            record.append(cc)
            record.append(click_1[i])

            cc = colour_2[i].tolist()
            cc = sorted(cc)
            cc = ','.join(str(x) for x in cc)
            record.append(cc)
            record.append(click_2[i])

            if values[i] > threshold:
                record.append(indices[i])
            else:
                record.append(-1)
            res.append(record)
        return res


def gen_sample_data(user_count, country_count, index=1, units=[[8], [4], [8, 3, 3], [8, 3, 3], 8, 4]):
    tf.reset_default_graph()
    batch_size = 5
    sample = Sample()
    m = RecommendationModel(colour_count=128, recommend_num=6, user_count=user_count, country_count=country_count)
    dataset = sample.read_csv(os.path.dirname(__file__) + '/../data/sample.csv', batch_size=batch_size)
    iterator = dataset.make_one_shot_iterator()
    columns = iterator.get_next()
    features = {'user': columns[0], 'country': columns[1],
                'recommend_colours_1': columns[2],
                'click_colour_1': columns[3],
                'recommend_colours_2': columns[4], 'click_colour_2': columns[5]}

    fs = m.features(features)
    output = m.forward(fs, units)
    output = m.output(output)
    global_step = tf.train.get_or_create_global_step()
    with tf.train.MonitoredTrainingSession(checkpoint_dir=base_model_dir + '/{}'.format(index)) as mon_sess:
        with open(os.path.dirname(__file__) + '/../data/org_sample_{}.csv'.format(index), 'w') as f:
            w = csv.writer(f, delimiter=' ')
            try:
                while True:
                    res = mon_sess.run([columns[0], columns[1], output])
                    res = SampleGenerator.generate_sample(res[0], res[1], res[2])
                    for j in res:
                        w.writerow(j)
            except Exception as e:
                print('Read to end')
            finally:
                print('Generate org_sample.csv')


def gen_training_sample(user_count, country_count, index=1, units=[[8], [4], [8, 3, 3], [8, 3, 3], 8, 4]):
    tf.reset_default_graph()
    batch_size = 5
    sample = Sample()
    m = RecommendationModel(colour_count=128, recommend_num=6, user_count=user_count, country_count=country_count)
    input_file = data_dir + 'no_label_sample_{}.csv'.format(index)
    # input_file = data_dir + 'sample.csv'
    dataset = sample.read_csv(input_file, batch_size=batch_size)
    iterator = dataset.make_one_shot_iterator()
    columns = iterator.get_next()
    features = {'user': columns[0], 'country': columns[1],
                'recommend_colours_1': columns[2],
                'click_colour_1': columns[3],
                'recommend_colours_2': columns[4], 'click_colour_2': columns[5]}

    fs = m.features(features)
    output = m.forward(fs, units)
    output = m.top_one_output(output)
    global_step = tf.train.get_or_create_global_step()
    with tf.train.MonitoredTrainingSession(checkpoint_dir=base_model_dir + '/{}'.format(index)) as mon_sess:
        with open(data_dir + 'train_sample_{}.csv'.format(index), 'w') as f:
            wr = csv.writer(f, delimiter=' ')
            result_list = []
            try:
                while not mon_sess.should_stop():
                    f, o = mon_sess.run([features, output])
                    r = SampleGenerator.generate_label_sample(f['user'],
                                                 f['country'],
                                                 f['recommend_colours_1'],
                                                 f['click_colour_1'],
                                                 f['recommend_colours_2'],
                                                 f['click_colour_2'],
                                                 o[0], o[1])
                    for i in r:
                        result_list.append(i)
            finally:
                random.shuffle(result_list)
                print('len:' + str(len(result_list)))

                for j in result_list:
                    wr.writerow(j)


def train(index):
    tf.reset_default_graph()
    batch_size = 300
    m = RecommendationModel(colour_count=128, recommend_num=6, user_count=100, country_count=20)
    dataset = Sample.read_label_data(data_dir + 'train_sample_{}.csv'.format(index), batch_size=batch_size)
    iterator = dataset.make_one_shot_iterator()
    columns = iterator.get_next()
    features = {'user': columns[0], 'country': columns[1],
                'recommend_colours_1': columns[2],
                'click_colour_1': columns[3],
                'recommend_colours_2': columns[4], 'click_colour_2': columns[5]}
    labels = columns[6]
    label_tensor = m.input_to_one_hot_plus(labels, m.colour_count)
    fs = m.features(features)
    last_layer = m.forward(fs)
    top_indices, top_values = m.output(last_layer)
    top_indices_real = tf.cast(top_indices, tf.int32)
    acc = m.accuracy(labels, top_indices_real, m.recommend_num)
    tf.summary.scalar(name='accuracy', tensor=acc, family='train')
    loss = m.loss(logits=last_layer, labels=label_tensor)
    tf.summary.scalar(name='loss', tensor=loss, family='train')
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-04)
    train_op = optimizer.minimize(loss, global_step=global_step)
    hooks = [tf.train.StopAtStepHook(last_step=5000)]
    # ckpt_dir = test_model_dir + '/{}'.format(index)
    ckpt_dir = test_model_dir + '/mix'

    with tf.train.MonitoredTrainingSession(checkpoint_dir=ckpt_dir, hooks=hooks) as mon_sess:
        step = 0
        while not mon_sess.should_stop():
            step += 1
            _, acc_res, label_res, top_indices_real_res, loss_res, global_step_res \
                = mon_sess.run([train_op, acc, labels, top_indices_real, loss, global_step])

            if step % 100 == 0:
                print("Train step %d, loss: %f accuracy: %f" % (step, loss_res, acc_res))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    # for i in range(1, 6):
    #     train(i)
    train(2)
