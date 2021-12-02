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
import sys
import tensorflow as tf
from flink_ml_tensorflow.tensorflow_context import TFContext
from r_model import Sample, RecommendationModel


class ModelTrainer(object):
    def __init__(self, tf_context, hooks, batch_size):
        self.tf_context = tf_context
        self.hooks = hooks
        self.batch_size = batch_size

    def train(self, input_func):
        job_name = self.tf_context.get_role_name()
        index = self.tf_context.get_index()
        cluster_json = self.tf_context.get_tf_cluster()
        print(cluster_json)
        checkpoint_dir = self.tf_context.properties['checkpoint_dir']
        sys.stdout.flush()
        cluster = tf.train.ClusterSpec(cluster=cluster_json)
        server = tf.train.Server(cluster, job_name=job_name, task_index=index)
        sess_config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:%d" % index])
        if 'ps' == job_name:
            from time import sleep
            while True:
                sleep(100)
        else:
            with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:' + str(index),
                                                          cluster=cluster)):

                m = RecommendationModel(colour_count=128, recommend_num=6, user_count=100, country_count=20)
                dataset = input_func(self.batch_size)
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

            is_chief = (index == 0)
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=is_chief,
                                                   config=sess_config,
                                                   checkpoint_dir=checkpoint_dir,
                                                   hooks=self.hooks,
                                                   save_checkpoint_secs=30) as mon_sess:
                step = 0
                while not mon_sess.should_stop():
                    step += 1
                    _, acc_res, label_res, top_indices_real_res, loss_res, global_step_res \
                        = mon_sess.run([train_op, acc, labels, top_indices_real, loss, global_step])

                    if step % 100 == 0:
                        print("Index %d global_step %d step %d, loss: %f accuracy: %f"
                              % (index, global_step_res, step, loss_res, acc_res))


def stream_train(context):
    tf_context = TFContext(context)
    batch_size = 300

    def flink_input_func(batch_size):
        dataset = tf_context.flink_stream_dataset()
        dataset = Sample.read_flink_dataset(dataset, batch_size)
        return dataset

    trainer = ModelTrainer(tf_context=tf_context,
                           hooks=[],
                           batch_size=batch_size)
    trainer.train(input_func=flink_input_func)


def batch_train(context):
    tf_context = TFContext(context)
    batch_size = 300
    input_files = tf_context.properties['input_files']

    def file_input_func(batch_size):
        dataset = Sample.read_label_data(file_path=input_files, batch_size=batch_size)
        return dataset

    trainer = ModelTrainer(tf_context=tf_context,
                           hooks=[tf.train.StopAtStepHook(last_step=2050)],
                           batch_size=batch_size)
    trainer.train(input_func=file_input_func)
