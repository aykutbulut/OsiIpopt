#!/bin/bash
mkdir opt_build
cd opt_build
build_dir=$PWD
inc_dir=${build_dir%%/}/include
lib_dir=${build_dir%%/}/lib
pkg_dir=${lib_dir%%/}/pkgconfig
PKG_CONFIG_PATH=${pkg_dir}:$PKG_CONFIG_PATH
# export CXXFLAGS="-g -Wall"
# configure and install ASL
mkdir -p ThirdParty/ASL
cd ThirdParty/ASL
../../../ThirdParty/ASL/configure --prefix=$build_dir
make -j 10 install
cd ..
# configure and install BLAS
mkdir Blas
cd Blas
../../../ThirdParty/Blas/configure --prefix=$build_dir
make -j 10 install
cd ..
# configure and install HSL
mkdir HSL
cd HSL
../../../ThirdParty/HSL/configure --prefix=$build_dir
make -j 10 install
cd ..
# configure and install Lapack
mkdir Lapack
cd Lapack
../../../ThirdParty/Lapack/configure --prefix=$build_dir
make -j 10 install
cd ..
# configure and install Metis
mkdir Metis
cd Metis
../../../ThirdParty/Metis/configure --prefix=$build_dir
make -j 10 install
cd ..
# configure and install Mumps
mkdir Mumps
cd Mumps
../../../ThirdParty/Mumps/configure --prefix=$build_dir
make -j 10 install
cd ../..
# configure and install CoinUtils
mkdir CoinUtils
cd CoinUtils
../../CoinUtils/configure --prefix=$build_dir
make -j 10 install
cd ..
# configure and install Osi
mkdir Osi
cd Osi
../../Osi/configure --prefix=$build_dir
make -j 10 install
cd ..
# configure and install OsiConic
mkdir OsiConic
cd OsiConic
../../OsiConic/configure --prefix=$build_dir
make -j 10 install
cd ..
#configure and install Ipopt
mkdir Ipopt
cd Ipopt
../../Ipopt/configure --prefix=$build_dir
make -j 10 install
cd ..
#configure and install OsiIpopt
mkdir OsiIpopt
cd OsiIpopt
../../configure --prefix=$build_dir
make -j 10 install
cd ..
