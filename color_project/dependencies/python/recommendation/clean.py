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
import shutil
from recommendation import config


def clean():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    generated = project_dir + '/generated'
    if os.path.exists(generated):
        shutil.rmtree(generated)
    temp = project_dir + '/temp'
    if os.path.exists(temp):
        shutil.rmtree(temp)

    if os.path.exists(config.TempModelDir):
        shutil.rmtree(config.TempModelDir)

    if os.path.exists(config.BatchModelDir):
        shutil.rmtree(config.BatchModelDir)

    if os.path.exists(config.StreamModelDir):
        shutil.rmtree(config.StreamModelDir)


if __name__ == '__main__':
    clean()