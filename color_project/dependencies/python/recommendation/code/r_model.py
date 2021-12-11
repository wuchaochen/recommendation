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
import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


class RecommendationModel(object):
    def __init__(self, colour_count, recommend_num, user_count, country_count):
        self.colour_count = colour_count
        self.recommend_num = recommend_num
        self.user_count = user_count
        self.country_count = country_count

    def forward(self, features, units=[[8], [4], [8, 3, 3], [8, 3, 3], 4, 4]):
        user_layer = self.network(features['user'], units[0])
        country_layer = self.network(features['country'], units[1])
        click_1_feature = tf.concat([features['recommend_colours_1'], features['click_colour_1']], axis=1)
        r_1 = self.network(click_1_feature, units[2])
        click_2_feature = tf.concat([features['recommend_colours_2'], features['click_colour_2']], axis=1)
        r_2 = self.network(click_2_feature, units[3])
        concat_layer_1 = tf.concat([user_layer, country_layer], axis=1)
        concat_layer_2 = tf.concat([r_1, r_2], axis=1)
        last_layer_1 = self.network(concat_layer_1, [units[4], self.colour_count])
        last_layer_2 = self.network(concat_layer_2, [units[5], self.colour_count])
        rate = 0.001
        last_layer = (1-rate) * last_layer_1 + rate * last_layer_2
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
        first_layer = tf.layers.dense(inputs=inputs, units=units[0],
                                      kernel_initializer=tf.initializers.truncated_normal())
        layer = first_layer
        for i in units[1:]:
            layer = tf.layers.dense(inputs=layer, units=i, kernel_initializer=tf.initializers.random_uniform())
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

    def output(self, input):
        input = tf.nn.softmax(input)
        top_values, top_indices = tf.nn.top_k(input, k=self.recommend_num)
        return top_indices, top_values

    def top_one_output(self, input):
        input = tf.nn.softmax(input)
        top_values, top_indices = tf.nn.top_k(input, k=1)
        return tf.squeeze(top_indices), tf.squeeze(top_values)

    def softmax(self, input):
        return tf.nn.softmax(input)

    def inference(self, features):
        fs = self.features(features)
        l = self.forward(fs)
        return self.output(l)[0]

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

    def accuracy(self, labels, predictions, element_count):
        batch_size = tf.cast(tf.size(labels), tf.float32)
        a = tf.tile(labels, [element_count])
        a = tf.reshape(a, [element_count, batch_size])
        a = tf.transpose(a)
        a = tf.cast(tf.equal(a, predictions), tf.float32)
        a = tf.reduce_sum(tf.matmul(a, tf.ones([element_count, 1])))
        acc = a / batch_size
        return acc


class Sample(object):
    @staticmethod
    def read_csv(file_path, batch_size):
        def multiply_split(value):
            tmp = tf.strings.to_number(tf.sparse.to_dense(tf.string_split([value], sep=',')), tf.int32)
            return tf.squeeze(tmp)

        def parse_csv(value):
            columns = tf.decode_csv(value, record_defaults=[0, 0, '0,0,0,0,0,0', 0, '0,0,0,0,0,0', 0], field_delim=' ')
            return columns[0], columns[1], multiply_split(columns[2]), columns[3], multiply_split(columns[4]), columns[
                5]

        ds = tf.data.TextLineDataset(filenames=[file_path])
        ds = ds.map(parse_csv)
        ds = ds.repeat(1)
        return ds.batch(batch_size)

    @staticmethod
    def read_label_data(file_path, batch_size, parallelism=None, index=None):
        def multiply_split(value):
            tmp = tf.strings.to_number(tf.sparse.to_dense(tf.string_split([value], sep=',')), tf.int32)
            return tf.squeeze(tmp)

        def parse_csv(value):
            columns = tf.decode_csv(value,
                                    record_defaults=[0, 0, '0,0,0,0,0,0', 0, '0,0,0,0,0,0', 0, 0], field_delim=' ')
            return columns[0], columns[1], multiply_split(columns[2]), columns[3], multiply_split(columns[4]), columns[
                5], columns[6]

        if isinstance(file_path, list):
            files = file_path
        else:
            files = [file_path]

        logger.info("{} all files: {}".format(index, files))
        if parallelism is not None and index is not None:
            partitioned_files = files[index::parallelism]
        else:
            partitioned_files = files

        logger.info("index: {} read files: {}".format(index, partitioned_files))
        ds = tf.data.TextLineDataset(filenames=[partitioned_files])
        ds = ds.map(parse_csv)
        ds = ds.repeat(-1)
        return ds.batch(batch_size)

    @staticmethod
    def read_flink_dataset(dataset, batch_size):
        def multiply_split(value):
            tmp = tf.strings.to_number(tf.sparse.to_dense(tf.string_split([value], sep=',')), tf.int32)
            return tf.squeeze(tmp)

        def parse_csv(value):
            columns = tf.decode_csv(value,
                                    record_defaults=[0, 0, '0,0,0,0,0,0', 0, '0,0,0,0,0,0', 0, 0], field_delim=' ')
            return columns[0], columns[1], multiply_split(columns[2]), columns[3], multiply_split(columns[4]), columns[
                5], columns[6]

        ds = dataset.map(parse_csv)
        return ds.batch(batch_size)
