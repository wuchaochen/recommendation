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
import random
import csv
import os
import time
import pandas as pd
import copy
import tensorflow as tf
from recommendation.code.r_model import RecommendationModel, Sample
from recommendation import config

threshold = 0.1

data_dir = '/tmp/data/test_data/'
base_model_dir = '/tmp/test_models/base/'


def random_item(size):
    random.seed(time.time_ns())
    return random.randint(0, size - 1)


class UserData(object):

    def __init__(self, user_count, country_count):
        self.user_count = user_count
        self.country_count = country_count

    def random_user_info_dict(self):
        result = {}
        for i in range(self.user_count):
            random.seed(time.time_ns())
            result[i] = random.randint(0, self.country_count - 1)
        return result


class ColourData(object):
    def __init__(self, count, select_count):
        self.count = count
        self.select_count = select_count
        self.colours = range(count)

    def random_colours(self):
        random.seed(time.time_ns())
        colours = random.sample(self.colours, self.select_count)
        cc = copy.deepcopy(colours)
        cc.append(-1)
        click = random.sample(cc, 1)
        return colours, click


class SampleData(object):
    def __init__(self, user_count, country_count, colour_count, select_count):
        self.user_data = UserData(user_count=user_count, country_count=country_count)
        self.colour_data = ColourData(count=colour_count, select_count=select_count)

    @staticmethod
    def load_user_dict():
        user_dict = {}
        with open(config.UserDictFile, 'r') as f:
            for l in f.readlines():
                tmp = l.split(' ')
                user_dict[int(tmp[0])] = int(tmp[1][:-1])
        return user_dict

    def create_data(self, num, output_dir=None):
        user_dict = self.user_data.random_user_info_dict()
        if output_dir is not None:
            with open(output_dir + '/users.csv', 'w') as f:
                wr = csv.writer(f, delimiter=' ')
                for i in range(self.user_data.user_count):
                    wr.writerow([i, user_dict[i]])
            s_f = open(output_dir + '/sample.csv', 'w')
            s_wr = csv.writer(s_f, delimiter=' ')

        for i in range(num):
            uid = random_item(self.user_data.user_count)
            country = user_dict[uid]
            c_1 = self.colour_data.random_colours()
            c_2 = self.colour_data.random_colours()
            if output_dir is None:
                print(uid, country, c_1, c_2)
            else:
                tmp = sorted(c_1[0])
                c_text_1 = ','.join([str(x) for x in tmp])
                cc_text_1 = ','.join([str(x) for x in c_1[1]])
                tmp = sorted(c_2[0])
                c_text_2 = ','.join([str(x) for x in tmp])
                cc_text_2 = ','.join([str(x) for x in c_2[1]])
                s_wr.writerow([uid, country, c_text_1, cc_text_1, c_text_2, cc_text_2])

        if output_dir is not None:
            s_f.close()


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
    dataset = sample.read_csv(data_dir + '/sample.csv', batch_size=batch_size)
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
        with open(data_dir + '/org_sample_{}.csv'.format(index), 'w') as f:
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


def gen_trained_data(index=1):
    input_file_path = data_dir + 'org_sample_{}.csv'.format(index)
    output_file_path = data_dir + 'no_label_sample_{}.csv'.format(index)
    with open(input_file_path, 'r') as f:
        d1 = pd.read_csv(f, names=['f1', 'f2', 'f3', 'f4', 'f5'], sep=' ')
        d2 = d1.sort_values(by='f1')
        key = -1
        resume_record = None
        with open(output_file_path, 'w') as wf:
            wr = csv.writer(wf, delimiter=' ')
            for index, row in d2.iterrows():
                if row['f1'] == key:
                    wr.writerow([row['f1'], row['f2'], row['f3'], row['f4'], resume_record[2], resume_record[3]])
                    key = -1
                    resume_record = None
                else:
                    key = row['f1']
                    resume_record = row['f1'], row['f2'], row['f3'], row['f4']


def gen_mix_data():
    with open(data_dir+'train_sample_1.csv', 'r') as f_1, open(data_dir+'train_sample_2.csv', 'r') as f_2:
        lines_1 = f_1.readlines()
        lines_2 = f_2.readlines()
        l3 = lines_2[0: int(len(lines_2)/10)]
        lines_1.extend(l3)
        random.shuffle(lines_1)
    with open(data_dir+'train_sample_3.csv', 'w') as f_3:
        f_3.writelines(lines_1)


def gen_split_data():
    with open(data_dir+'train_sample_1.csv', 'r') as f_1:
        lines_1 = f_1.readlines()
        l1 = lines_1[0: int(len(lines_1)/2)]
        l2 = lines_1[int(len(lines_1)/2):]
    with open(data_dir+'train_sample_1_1.csv', 'w') as f_1_1:
        f_1_1.writelines(l1)

    with open(data_dir+'train_sample_1_2.csv', 'w') as f_1_2:
        f_1_2.writelines(l2)


def pipeline():
    s_data = SampleData(user_count=config.user_count, country_count=20, colour_count=128, select_count=6)
    s_data.create_data(100000, data_dir)
    print('create random sample data.')
    u = [[8], [4], [8, 3, 3], [8, 3, 3], 8, 4]

    for i in range(1, 3):
        gen_sample_data(user_count=config.user_count, country_count=20, index=i, units=u)
        print('create sample data step 1, index: {}'.format(i))

        gen_trained_data(index=i)
        print('create sample data step 2, index: {}'.format(i))

        gen_training_sample(user_count=config.user_count, country_count=20, index=i, units=u)
        print('create sample data step 3, index: {}'.format(i))


if __name__ == '__main__':
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    pipeline()
    # print(len(SampleData.load_user_dict()))
    # gen_mix_data()
    # gen_split_data()
