#!/bin/bash

export CEPH_HOME=~/ceph
export FIO_HOME=~/fio

./preconditioning.sh

./run-fio-queueing-delay.sh
#./run-fio-2osds.sh

