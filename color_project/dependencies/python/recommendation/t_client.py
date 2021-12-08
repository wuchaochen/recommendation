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
from recommendation.inference_client import InferenceClient
from recommendation.app.agent_client import AgentClient


# c = InferenceClient(uri='localhost:30002')
# r = c.inference('2')
# print(r)

a = AgentClient(uri='localhost:30001')
r = a.click('82 13 8,14,27,49,107,110 -1 29,60,65,79,86,102 60')
print(r)