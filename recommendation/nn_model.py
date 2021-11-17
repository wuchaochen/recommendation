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

threshold = 0.3

data_dir = os.path.dirname(__file__) + '/../data/'
base_model_dir = os.path.dirname(__file__) + '/../models/base/'
train_model_dir = os.path.dirname(__file__) + '/../models/train/'
test_model_dir = os.path.dirname(__file__) + '/../models/test/'


class RecommendationModel(object):
    def __init__(self, colour_count, recommend_num, user_count, country_count):
        self.colour_count = colour_count
        self.recommend_num = recommend_num
        self.user_count = user_count
        self.country_count = country_count

    def forward(self, features, units=[[8], [4], [8, 3, 3], [8, 3, 3], 8, 4]):
        user_layer = self.network(features['user'], units[0])
        country_layer = self.network(features['country'], units[1])
        click_1_feature = tf.concat([features['recommend_colours_1'], features['click_colour_1']], axis=1)
        r_1 = self.network(click_1_feature, units[2])
        click_2_feature = tf.concat([features['recommend_colours_2'], features['click_colour_2']], axis=1)
        r_2 = self.network(click_2_feature, units[3])
        concat_layer = tf.concat([user_layer, country_layer, r_1, r_2], axis=1)
        last_layer = self.network(concat_layer, [units[4], units[5], self.colour_count])
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
        first_layer = tf.layers.dense(inputs=inputs, units=units[0], kernel_initializer=tf.initializers.truncated_normal())
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
    def read_label_data(file_path, batch_size):
        def multiply_split(value):
            tmp = tf.strings.to_number(tf.sparse.to_dense(tf.string_split([value], sep=',')), tf.int32)
            return tf.squeeze(tmp)

        def parse_csv(value):
            columns = tf.decode_csv(value,
                                    record_defaults=[0, 0, '0,0,0,0,0,0', 0, '0,0,0,0,0,0', 0, 0], field_delim=' ')
            return columns[0], columns[1], multiply_split(columns[2]), columns[3], multiply_split(columns[4]), columns[
                5], columns[6]

        ds = tf.data.TextLineDataset(filenames=[file_path])
        ds = ds.map(parse_csv)
        ds = ds.repeat(-1)
        return ds.batch(batch_size)

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
                    res = Sample.generate_sample(res[0], res[1], res[2])
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
                    r = Sample.generate_label_sample(f['user'],
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
    sample = Sample()
    m = RecommendationModel(colour_count=128, recommend_num=6, user_count=100, country_count=20)
    dataset = sample.read_label_data(data_dir + 'train_sample_{}.csv'.format(index), batch_size=batch_size)
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
