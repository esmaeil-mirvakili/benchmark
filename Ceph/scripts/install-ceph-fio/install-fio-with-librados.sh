#!/bin/bash

sudo apt update
sudo apt-get install python3-routes

# install ceph
cd ~
git clone https://github.com/yzhan298/ceph.git
cd ceph	
git checkout research_rs_mg
export CEPH_HOME="$(pwd)"
./install-deps.sh
sudo apt update
./do_cmake.sh -DWITH_MANPAGE=OFF -DWITH_BABELTRACE=OFF -DWITH_MGR_DASHBOARD_FRONTEND=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
cd build
make -j`nproc`

# install package "python3-routes" to get rid of the "No module named routes" error
#sudo apt-get install python3-routes

../src/stop.sh && rm -rf out dev && MON=1 OSD=1 MGR=1 MDS=0 RGW=0 ../src/vstart.sh -n -x
./bin/ceph osd pool create rados

# install fio
cd ~
git clone https://github.com/axboe/fio.git
cd fio
export FIO_HOME="$(pwd)"
LDFLAGS=-I"$CEPH_HOME"/src/include LIBRARY_PATH="$CEPH_HOME"/build/lib:$LIBRARY_PATH ./configure
EXTFLAGS=-I"$CEPH_HOME"/src/include LIBRARY_PATH="$CEPH_HOME"/build/lib:$LIBRARY_PATH make -j `nproc`

cd "$CEPH_HOME"/build
LD_LIBRARY_PATH="$CEPH_HOME"/build/lib:$LD_LIBRARY_PATH "$FIO_HOME"/fio "$FIO_HOME"/examples/rados.fio

# set git
git config --global user.name "yzhan298"
git config --global user.email "yzhan298@ucsc.edu"

# install useful packages
sudo apt install cscope
