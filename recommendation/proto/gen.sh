#!/usr/bin/env bash

python3 -m grpc.tools.protoc -I. \
  -I/usr/local/include \
  --python_out=. \
  --grpc_python_out=. \
  service.proto