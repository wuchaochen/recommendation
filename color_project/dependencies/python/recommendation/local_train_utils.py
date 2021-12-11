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
import tensorflow as tf
from recommendation.code.r_model import RecommendationModel, Sample
from recommendation import config
data_dir = os.path.dirname(__file__) + '/../data/'
base_model_dir = os.path.dirname(__file__) + '/../models/base/'
train_model_dir = os.path.dirname(__file__) + '/../models/train/'
test_model_dir = os.path.dirname(__file__) + '/../models/test/'


def train(index):
    tf.reset_default_graph()
    batch_size = 300
    m = RecommendationModel(colour_count=config.color_count, recommend_num=6, user_count=config.user_count, country_count=20)
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
    for i in range(1, 6):
        train(i)
    # train(2)
