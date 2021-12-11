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
import tensorflow as tf

from recommendation.code.r_model import RecommendationModel, Sample
from recommendation.kafka_utils import KafkaUtils
from recommendation import config


def run_validate(input_func, batch_size, checkpoint_dir):
    m = RecommendationModel(colour_count=config.color_count, recommend_num=6, user_count=config.user_count, country_count=20)
    dataset = input_func(batch_size)
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
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           summary_dir=checkpoint_dir + '/validate',
                                           save_checkpoint_secs=None,
                                           save_checkpoint_steps=None) as mon_sess:
        acc_res, loss_res, global_step_res \
            = mon_sess.run([acc, loss, global_step])

        print("Step %d, loss: %f accuracy: %f" % (global_step_res, loss_res, acc_res))
        return acc_res


class ValidateJob(object):
    @staticmethod
    def batch_validate(checkpoint_dir, validate_files, data_count):
        def file_input_func(batch_size):
            dataset = Sample.read_label_data(file_path=validate_files, batch_size=batch_size)
            return dataset

        return run_validate(input_func=file_input_func, checkpoint_dir=checkpoint_dir, batch_size=data_count)

    @staticmethod
    def stream_validate(checkpoint_dir, topic, data_count, broker):
        def kafka_input_func(batch_size):
            def multiply_split(value):
                tmp = tf.strings.to_number(tf.sparse.to_dense(tf.string_split([value], sep=',')), tf.int32)
                return tf.squeeze(tmp)

            def parse_csv(value):
                columns = tf.decode_csv(value,
                                        record_defaults=[0, 0, '0,0,0,0,0,0', 0, '0,0,0,0,0,0', 0, 0], field_delim=' ')
                return columns[0], columns[1], multiply_split(columns[2]), columns[3], multiply_split(columns[4]), \
                       columns[5], columns[6]

            kafka_utils = KafkaUtils(broker)
            val_data = kafka_utils.read_data(topic=topic, count=batch_size, offset='latest')
            dataset = tf.data.Dataset.from_tensor_slices(val_data)
            dataset = dataset.map(parse_csv)
            dataset = dataset.batch(batch_size)
            return dataset

        return run_validate(input_func=kafka_input_func, checkpoint_dir=checkpoint_dir, batch_size=data_count)


if __name__ == '__main__':
    # ValidateJob.batch_validate(checkpoint_dir='/tmp/model/batch/v1',
    #                            data_count=1000,
    #                            validate_files=os.path.dirname(__file__) + '/../data/train_sample_2.csv')
    ValidateJob.stream_validate(checkpoint_dir='/tmp/model/train/stream/v1',
                                data_count=10,
                                topic='sample_input', broker="localhost:9092")
