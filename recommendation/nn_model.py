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


class RecommendationModel(object):
    def __init__(self, colour_count, recommend_num):
        self.colour_count = colour_count
        self.recommend_num = recommend_num

    def forward(self, input_colours, recommend_colours_1, click_colour_1, recommend_colours_2, click_colour_2,
                country, uid):
        pass

    def input_to_one_hot(self, input_colours):
        batch_size = tf.size(input_colours)
        labels_1 = tf.expand_dims(input_colours, 1)
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        concat = tf.concat([indices, labels_1], 1)
        one_hot = tf.sparse_to_dense(concat, tf.stack([batch_size, self.colour_count]), 1.0, 0.0)
        return one_hot

    def network(self, inputs, units):
        first_layer = tf.layers.dense(inputs=inputs, units=units[0], kernel_initializer=tf.initializers.random_normal())
        layer = first_layer
        for i in units[1:]:
            layer = tf.layers.dense(inputs=layer, units=i, kernel_initializer=tf.initializers.random_normal())
        last_layer = tf.layers.dense(inputs=layer, units=self.colour_count,
                                     kernel_initializer=tf.initializers.random_normal())
        return last_layer

    def forward_1(self, input_colours):
        inputs = self.input_to_one_hot(input_colours)
        last_layer = self.network(inputs=inputs, units=[5, 5, 10])
        return tf.nn.softmax(last_layer)

    def output(self, input):
        top_values, top_indices = tf.nn.top_k(input, k=self.recommend_num)
        return top_indices

    def loss(self, logits, labels):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                               logits=logits))
        return cross_entropy


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    m = RecommendationModel(colour_count=200, recommend_num=6)
    input = [3, 6, 15]
    forward_output = m.forward_1(input_colours=input)
    output = m.output(forward_output)
    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init_op)
        res = session.run([forward_output, output])
        tf.logging.info(res)
