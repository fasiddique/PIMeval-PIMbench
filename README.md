## PIMeval Simulator and PIMbench Suite

![License](https://img.shields.io/badge/license-MIT-green.svg)

### Description
* PIMeval
  * A C++ library based PIM simulation and evaluation framework
  * Support various subarray-level bit-serial, subarray-level bit-paralle and bank level PIM architectures
  * Support both vertical and horizontal data layouts
  * Support multi PIM core programming model and resource management
  * Support high-level functional programming with a set of general APIs common to all PIM architectures
  * Support low-level micro-ops programming for modeling architecture details
  * Support performance and energy modeling with detailed stats tracking
  * Support multi-threaded simulation for runtime
* PIMbench
  * A rich set of PIM benchmark applications on top of the PIMeval functional simulation and evaluation framework

### Quick start
```
git clone https://github.com/deyuan/PIMeval.git
cd PIMeval/
make -j10
./PIMbench/cpp-vec-add/vec-add.out
```

### Code Structure
* PIMeval: PIM similation framework - libpimeval
  * `libpimeval/src`: PIMeval simualtor source code
  * `libpimeval.h`: PIMeval simulator library interface
  * `libpimeval.a`: PIMeval simulator library (after make)
* PIMbench: PIM benchmark suite
  * `cpp-aes`: AES encryption/decryption
  * `cpp-axpy`: aX+Y operation
  * `cpp-filter-by-key`: Filer by key
  * `cpp-gemm`: General matrix-matrix product
  * `cpp-gemv`: General matrix-vector product
  * `cpp-histogram`: Histogram
  * `cpp-image-downsampling`: Image downsampling
  * `cpp-kmeans`: K-means
  * `cpp-knn`: kNN
  * `cpp-linear-regression`: Linear regression
  * `cpp-radix-sort`: Radix sort
  * `cpp-triangle-count`: Triangle counting
  * `cpp-vec-add`: Vector addition
  * `cpp-vgg13`: VGG-13
  * `cpp-vgg16`: VGG-16
  * `cpp-vgg19`: VGG-19
* More applications
  * `cpp-brightness`: Image brightness
  * `cpp-convolution`: Convolution
  * `cpp-db-filtering`: DB filtering
  * `cpp-dot-prod`: Dot product
  * `cpp-pooling`: Max pooling
  * `cpp-relu`: ReLU
  * `cpp-sad`: Sum of absolute difference
  * `cpp-vec-arithmetic`: Vector arithmetic
  * `cpp-vec-comp`: Vector comparison
  * `cpp-vec-div`: Vector division
  * `cpp-vec-logical`: Vector logical operations
  * `cpp-vec-mul`: Vector multiplication
  * `cpp-vec-popcount`: Vector popcount
  * `cpp-vec-broadcast-popcnt`: Vector broadcast and pop count
* Bit-serial micro-program evaluation framework
  * `bit-serial`
* Functional tests
  * `tests`

### How To Build
* Run `make` at root directory or subdirectories
  * `make perf`: Build with `-Ofast` for performance measurement (default)
  * `make debug`: Build with `-g` and `-DDEBUG` for debugging and printing verbose messages
<!--
  * `make dramsim3_integ`: Enable DRAMsim3 related code with `-DDRAMSIM3_INTEG`
-->
* Multi-threaded building
  * `make -j10`
* Specify simulation target
  * `make PIM_SIM_TARGET=PIM_DEVICE_BITSIMD_V` (default)
  * `make PIM_SIM_TARGET=PIM_DEVICE_FULCRUM`
  * `make PIM_SIM_TARGET=PIM_DEVICE_BANK_LEVEL`
* Build with OpenMP
  * `make USE_OPENMP=1`
  * Guard any `-fopenmp` with this flag in Makefile used by a few applications

<!--
### About DRAMsim3 Integration
* This module contains a copy of DRAMsim3
  * Oringal DRAMsim3 repo: https://github.com/umd-memsys/DRAMsim3
  * Clone date: 05/06/2024
  * Location: ./third_party/DRAMsim3/
* DRAMsim3 related code are guarded with DRAMSIM3_INTEG flag
  * Requires `make dramsim3_integ`
* Below is needed for dramsim3_integ for now
```bash
# Build dramsim3
git clone https://github.com/fasiddique/DRAMsim3.git
cd DRAMsim3/
git checkout benchmark
mkdir build
cd build
cmake ..
make -j
# Build PIM functional simulator
git clone <url_to_this_repo>
cd pim-func-sim
export DRAMSIM3_PATH=<path_to_DRAMSIM3>
make -j
```
-->

### Contributors
This repository is the result of a collaborative effort of many talented individuals. We are grateful to everyone who contributed to this repo. Special thanks to Deyuan Guo for initially architecting the PIMeval simulator framework and bit-serial evaluation, and Farzana Siddique for her exceptional contributions on both simulator and PIMbench suite.

\<citation recommendation to be updated\>