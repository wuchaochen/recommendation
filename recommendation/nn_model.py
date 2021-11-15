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


class RecommendationModel(object):
    def __init__(self, colour_count, recommend_num, user_count, country_count):
        self.colour_count = colour_count
        self.recommend_num = recommend_num
        self.user_count = user_count
        self.country_count = country_count

    def forward(self, features):
        user_layer = self.network(features['user'], [8])
        country_layer = self.network(features['country'], [4])
        click_1_feature = tf.concat([features['recommend_colours_1'], features['click_colour_1']], axis=1)
        r_1 = self.network(click_1_feature, [8, 3, 3])
        click_2_feature = tf.concat([features['recommend_colours_2'], features['click_colour_2']], axis=1)
        r_2 = self.network(click_2_feature, [8, 3, 3])
        concat_layer = tf.concat([user_layer, country_layer, r_1, r_2], axis=1)
        last_layer = self.network(concat_layer, [8, 4, 4, self.colour_count])
        return last_layer

    def features(self, features):
        user_fs = self.input_to_one_hot(features['user'], self.user_count)
        country_fs = self.input_to_one_hot(features['country'], self.country_count)
        r_c_1 = self.input_to_n_hot(features['recommend_colours_1'], self.colour_count)
        c_c_1 = self.input_to_one_hot_plus(features['click_colour_1'], self.colour_count)
        r_c_2 = self.input_to_n_hot(features['recommend_colours_2'], self.colour_count)
        c_c_2 = self.input_to_one_hot_plus(features['click_colour_2'], self.colour_count)
        return {'user': user_fs, 'country': country_fs, 'recommend_colours_1': r_c_1, 'click_colour_1': c_c_1,
                'recommend_colours_2': r_c_2, 'click_colour_2': c_c_2}

    def network(self, inputs, units):
        first_layer = tf.layers.dense(inputs=inputs, units=units[0], kernel_initializer=tf.initializers.random_normal())
        layer = first_layer
        for i in units[1:]:
            layer = tf.layers.dense(inputs=layer, units=i, kernel_initializer=tf.initializers.random_normal())
        return layer

    def input_to_one_hot(self, input, size):
        batch_size = tf.size(input)
        labels_1 = tf.expand_dims(input, 1)
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        concat = tf.concat([indices, labels_1], 1)
        one_hot = tf.sparse_to_dense(concat, tf.stack([batch_size, size]), 1.0, 0.0)
        return one_hot

    def input_to_n_hot(self, input, size):
        batch_size = tf.shape(input)[0]
        ex_input = tf.expand_dims(input, 2)
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        indices = tf.expand_dims(tf.broadcast_to(input=indices, shape=[batch_size, self.recommend_num]), 2)
        concat = tf.concat([indices, ex_input], -1)
        concat = tf.reshape(concat, [-1, 2])
        concat = tf.cast(concat, tf.int64)
        n_hot = tf.sparse_to_dense(sparse_indices=concat,
                                   output_shape=[batch_size, size],
                                   sparse_values=1.0,
                                   default_value=0.0)
        return n_hot

    def input_to_one_hot_plus(self, input, size):
        condition_mask = tf.greater_equal(input, tf.constant(0))
        partitioned_data = tf.dynamic_partition(
            input, tf.cast(condition_mask, tf.int32), 2)
        batch_size = tf.size(partitioned_data[1])
        labels_1 = tf.expand_dims(partitioned_data[1], 1)
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        concat = tf.concat([indices, labels_1], 1)
        one_hot = tf.sparse_to_dense(concat, tf.stack([batch_size, size]), 1.0, 0.0)
        ss = tf.shape(partitioned_data[0])
        zz = tf.zeros(shape=[ss[0], size])
        condition_indices = tf.dynamic_partition(
            tf.range(tf.shape(input)[0]), tf.cast(condition_mask, tf.int32), 2)
        res = tf.dynamic_stitch(condition_indices, [zz, one_hot])
        return res

    def forward_1(self, users):
        inputs = self.input_to_one_hot(users, self.user_count)
        last_layer = self.network(inputs=inputs, units=[5, 5, self.colour_count])
        return tf.nn.softmax(last_layer)

    def output(self, input):
        input = tf.nn.softmax(input)
        top_values, top_indices = tf.nn.top_k(input, k=self.recommend_num)
        return top_indices, top_values

    def inference(self, features):
        fs = self.features(features)
        l = self.forward(fs)
        return self.output(l)

    def loss(self, logits, labels):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                               logits=logits))
        return cross_entropy

    def sparse_feature(self, input, size, dimension):
        batch_size = tf.size(input)
        labels_1 = tf.expand_dims(input, 1)
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        concat = tf.concat([indices, labels_1], 1)
        st = tf.SparseTensor(indices=tf.cast(concat, tf.int64),
                                 values=tf.ones(shape=[batch_size], dtype=tf.int32),
                                 dense_shape=[batch_size, size])
        result = tf.feature_column.embedding_column(st, dimension=dimension)
        return result


class Sample(object):
    @staticmethod
    def read_csv(file_path, batch_size):
        def multiply_split(value):
            tmp = tf.strings.to_number(tf.sparse.to_dense(tf.string_split([value], sep=',')), tf.int32)
            return tf.squeeze(tmp)

        def parse_csv(value):
            columns = tf.decode_csv(value, record_defaults=[0, 0, '0,0,0,0,0,0', 0, '0,0,0,0,0,0', 0], field_delim=' ')
            return columns[0], columns[1], multiply_split(columns[2]), columns[3], multiply_split(columns[4]), columns[5]

        ds = tf.data.TextLineDataset(filenames=[file_path])
        ds = ds.map(parse_csv)
        ds = ds.repeat(2)
        return ds.batch(batch_size)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    sample = Sample()
    # m = RecommendationModel(colour_count=10, recommend_num=3, user_count=10, country_count=5)
    # batch_size = 3
    # users = [1, 5, 8]
    # country = [0, 2, 3]
    # color = [[1, 2, 9], [3, 6, 8], [4, 5, 6]]
    # click = [1, -1, 5]
    # features = {'user': users, 'country': country, 'recommend_colours_1': color, 'click_colour_1': click,
    #             'recommend_colours_2': color, 'click_colour_2': click}

    # fs = m.features(features)
    # output = m.forward(fs)
    # output = m.output(output)
    # init_op = tf.global_variables_initializer()
    # output = m.sparse_feature(users, 10, 3)
    m = RecommendationModel(colour_count=128, recommend_num=6, user_count=10000, country_count=100)
    dataset = sample.read_csv(os.path.dirname(__file__) + '/../data/sample.csv', batch_size=5)
    iterator = dataset.make_one_shot_iterator()
    columns = iterator.get_next()
    features = {'user': columns[0], 'country': columns[1],
                'recommend_colours_1': columns[2],
                'click_colour_1': columns[3],
                'recommend_colours_2': columns[4], 'click_colour_2': columns[5]}

    fs = m.features(features)
    output = m.forward(fs)
    output = m.output(output)
    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)
        res = session.run([output])
        tf.logging.info(res)

