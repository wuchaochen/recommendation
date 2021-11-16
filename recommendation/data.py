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
from recommendation import nn_model


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


def gen_trained_data(input_file_path, output_file_path):
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


data_dir = os.path.dirname(__file__) + '/../data/'


def pipeline():
    s_data = SampleData(user_count=100, country_count=20, colour_count=128, select_count=6)
    s_data.create_data(100000, data_dir)

    nn_model.gen_sample_data(user_count=100, country_count=20)

    gen_trained_data(data_dir + 'org_sample.csv', data_dir + 'no_label_sample.csv')

    nn_model.gen_training_sample(user_count=100, country_count=20)


if __name__ == '__main__':
    pipeline()
