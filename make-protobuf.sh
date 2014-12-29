#!/bin/bash -x
set -o errexit

rm -rf src/protobuf
mkdir src/protobuf
protoc --fastpython_out=src/protobuf --cpp_out=src/protobuf --proto_path flydra flydra/camera_feature_point.proto
