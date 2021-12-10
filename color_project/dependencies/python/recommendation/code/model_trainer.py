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
import datetime
import logging
import os
import shutil
import sys

import ai_flow as af
import tensorflow as tf
from flink_ml_tensorflow.tensorflow_context import TFContext
from notification_service.base_notification import BaseEvent
from notification_service.client import NotificationClient

from r_model import Sample, RecommendationModel

logger = logging.getLogger(__name__)


class CheckpointSaver(tf.train.CheckpointSaverListener):

    def __init__(self, checkpoint_dir, target_dir):
        self.checkpoint_dir = checkpoint_dir
        self.target_dir = target_dir
        while True:
            try:
                af.init_ai_flow_client("localhost:50051", "color_project", notification_server_uri="localhost:50052")
                self.ai_flow_client = af.get_ai_flow_client()
                break
            except Exception:
                pass

    def copy_checkpoint(self) -> str:
        target = os.path.join(self.target_dir, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        logger.info("Copying model checkpoint from {} to {}".format(self.checkpoint_dir, target))
        shutil.copytree(self.checkpoint_dir, target)
        logger.info("Checkpoint copy completed")
        return target


class StreamCheckpointSaver(CheckpointSaver):

    def __init__(self, checkpoint_dir, target_dir, stream_model_name):
        super().__init__(checkpoint_dir, target_dir)
        self.stream_model_name = stream_model_name
        self.notification_client = NotificationClient(server_uri="localhost:50052", sender="stream_train")

    def after_save(self, session, global_step_value):
        target = self.copy_checkpoint()
        model_meta = self.ai_flow_client.get_model_by_name(self.stream_model_name)
        model_version = self.ai_flow_client.register_model_version(model_meta, target)
        self.notification_client.send_event(BaseEvent(key=self.stream_model_name, value=model_version.version,
                                                      event_type='MODEL_GENERATED'))


class BatchCheckpointSaver(CheckpointSaver):

    def __init__(self, checkpoint_dir, target_dir, batch_model_name):
        super().__init__(checkpoint_dir, target_dir)
        self.batch_model_name = batch_model_name
        self.notification_client = NotificationClient(server_uri="localhost:50052", sender="batch_train")

    def end(self, session, global_step_value):
        target = self.copy_checkpoint()
        model_meta = self.ai_flow_client.get_model_by_name(self.batch_model_name)
        model_version = self.ai_flow_client.register_model_version(model_meta, target)
        self.notification_client.send_event(BaseEvent(key=self.batch_model_name, value=model_version.version,
                                                      event_type='MODEL_GENERATED'))


class ModelTrainer(object):
    def __init__(self, tf_context, hooks, batch_size, chief_only_hooks=None, base_model_checkpoint=None,
                 summary_dir=None):
        self.tf_context = tf_context
        self.hooks = hooks
        self.batch_size = batch_size
        self.chief_only_hooks = chief_only_hooks if chief_only_hooks else []
        self.base_model_checkpoint = base_model_checkpoint
        self.summary_dir = summary_dir
        logger.info("ModelTrainer: {}".format(self.__dict__))

    def train(self, input_func):
        job_name = self.tf_context.get_role_name()
        index = self.tf_context.get_index()
        cluster_json = self.tf_context.get_tf_cluster()
        print(cluster_json)
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
                                                   hooks=self.hooks,
                                                   checkpoint_dir=self.base_model_checkpoint,
                                                   save_checkpoint_steps=None,
                                                   save_checkpoint_secs=None,
                                                   summary_dir=self.summary_dir,
                                                   chief_only_hooks=self.chief_only_hooks) as mon_sess:
                step = 0
                while not mon_sess.should_stop():
                    step += 1
                    _, acc_res, label_res, top_indices_real_res, loss_res, global_step_res \
                        = mon_sess.run([train_op, acc, labels, top_indices_real, loss, global_step])

                    if step % 100 == 0:
                        print("Index %d global_step %d step %d, loss: %f accuracy: %f"
                              % (index, global_step_res, step, loss_res, acc_res))
                        sys.stdout.flush()


def stream_train(context):
    tf_context = TFContext(context)
    batch_size = 300

    def flink_input_func(batch_size):
        dataset = tf_context.flink_stream_dataset()
        dataset = Sample.read_flink_dataset(dataset, batch_size)
        return dataset

    checkpoint_dir = tf_context.properties['checkpoint_dir']
    base_model_checkpoint = tf_context.properties['base_model_checkpoint']
    model_save_path = tf_context.properties['model_save_path']
    stream_model_name = tf_context.properties['stream_model_name']
    checkpoint_saver = StreamCheckpointSaver(checkpoint_dir, model_save_path, stream_model_name)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir,
                                                         save_steps=None,
                                                         save_secs=60,
                                                         listeners=[checkpoint_saver])

    trainer = ModelTrainer(tf_context=tf_context,
                           hooks=[],
                           batch_size=batch_size,
                           chief_only_hooks=[checkpoint_saver_hook],
                           base_model_checkpoint=base_model_checkpoint,
                           summary_dir=checkpoint_dir)
    trainer.train(input_func=flink_input_func)


def batch_train(context):
    tf_context = TFContext(context)
    batch_size = 300
    input_files = tf_context.properties['input_files'].split(",")

    def file_input_func(batch_size):
        role = tf_context.get_role_name()
        index = tf_context.get_index()
        parallelism = tf_context.get_role_parallelism_map()[role]
        logger.info("role: {} index: {} parallelism: {}".format(role, index, parallelism))

        dataset = Sample.read_label_data(file_path=input_files, batch_size=batch_size,
                                         parallelism=parallelism, index=index)
        return dataset

    checkpoint_dir = tf_context.properties['checkpoint_dir']
    model_save_path = tf_context.properties['model_save_path']
    max_step = int(tf_context.properties['max_step'])
    batch_model_name = tf_context.properties['batch_model_name']
    checkpoint_saver = BatchCheckpointSaver(checkpoint_dir, model_save_path, batch_model_name)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir,
                                                         save_steps=None,
                                                         save_secs=30,
                                                         listeners=[checkpoint_saver])

    trainer = ModelTrainer(tf_context=tf_context,
                           hooks=[tf.train.StopAtStepHook(last_step=max_step)],
                           batch_size=batch_size,
                           chief_only_hooks=[checkpoint_saver_hook],
                           summary_dir=checkpoint_dir)
    trainer.train(input_func=file_input_func)
