## Benchmark Ceph BlueStore 

If you are using the cloudlab, you proably need to delete the partitions for home derectory.
..* run `fdisk` to delete all partitions and create a new partition.
..* after reboot, run `sudo resize2fs /dev/sdx1` to resize the filesystem to match the actual size.

Then we need to install Ceph and FIO
..* source `install-fio-with-librados.sh` to install Ceph and FIO

Run benchmark
..* run `run-fio-with-preconditioning.sh` and the data will be collected and plotted automatically

Understand the benchmark scripts
..* `run-fio-with-preconditioning.sh` consists two part: preconditioning and benchamrk workload
..* `preconditioning.sh` is responsible for preconditioning the SSD (you need to change the SSD path accordingly)
..* `run-fio-queueing-delay.sh` is responsible for generating the workload and recording all data. `block size`, `queue depth`, `iotype` and `runtime` should be changed accordingly. 
