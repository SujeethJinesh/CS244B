#!/bin/bash

cd "$(git rev-parse --show-toplevel)" 

url="https://dlcdn.apache.org/zookeeper/zookeeper-3.8.4/apache-zookeeper-3.8.4-bin.tar.gz"
curl -LO "$url"

tar -xzf apache-zookeeper-3.8.4-bin.tar.gz
rm apache-zookeeper-3.8.4-bin.tar.gz

# By default zookeeper uses zoo.cfg as the server starting config.
mv apache-zookeeper-3.8.4-bin/conf/zoo_sample.cfg apache-zookeeper-3.8.4-bin/conf/zoo.cfg