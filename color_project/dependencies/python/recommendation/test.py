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
# import tensorflow as tf
#
# lables = tf.constant([5, 2, 9, -1], dtype=tf.int32)
# predictions = tf.constant([[9, 3, 5, 2, 1], [8, 9, 0, 6, 5], [1, 9, 3, 4, 5], [1, 2, 3, 4, 5]])
# batch_size = tf.cast(tf.size(lables), tf.float32)
# a = tf.tile(lables, [5])
# a = tf.reshape(a, [5, 4])
# a = tf.transpose(a)
# a = tf.cast(tf.equal(a, predictions), tf.float32)
# b = tf.matmul(a, tf.ones([5, 1]))
# # a = tf.reduce_sum(tf.matmul(a, tf.ones([5, 1])))
# # acc = a / batch_size
#
#
# ## training
# # Run tensorflow and print the result
# with tf.Session() as sess:
#    print(sess.run([b, predictions]))
# print("[STREAM_VARS_2]:", sess.run(stream_vars))  # [3.0, 17.0]
import json
from notification_service.client import NotificationClient
from notification_service.base_notification import BaseEvent
client = NotificationClient('localhost:50052')
m_path = '/tmp/model/train/stream/20211210181444'
client.send_event(BaseEvent(key='update_agent', value=m_path, event_type='update_agent'))
ss = '{"_model_path": "%s"}' % (m_path)
client.send_event(BaseEvent(key='stream_color_model', value=ss, event_type='MODEL_DEPLOYED'))


