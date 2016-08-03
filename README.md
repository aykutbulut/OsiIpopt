# OsiIpopt [![Build Status](https://travis-ci.org/aykutbulut/OsiIpopt.svg?branch=master)](https://travis-ci.org/aykutbulut/OsiIpopt)


OsiIpopt is a conic solver interface for COIN-OR's [Ipopt][2] solver. OsiIpopt
implements [OsiConic][1] interface, which extends Open Solver Interface (OSI)
to second order conic optimization problems.

OsiIpopt depends on [CoinUtils][3], [OSI][4], [OsiConic][5] and Ipopt.

OsiIpopt is used by [DisCO][6] to solve mixed integer conic optimization problems.

[1]: https://github.com/aykutbulut/OSI-CONIC
[2]: https://projects.coin-or.org/Ipopt
[3]: https://projects.coin-or.org/CoinUtils
[4]: https://projects.coin-or.org/Osi
[5]: https://github.com/aykutbulut/OSI-CONIC
[6]: https://github.com/aykutbulut/DisCO

# An example

A simple application that reads problem from an MPS file (an extended MPS
format for second order cone problems) and solves it using Ipopt is given
below.

```C++
// Solves conic problems, reads them in mosek mps input format
// usage: ./cont_solver input.mps
#include <OsiIpoptSolverInterface.hpp>
#include <iostream>
#include <iomanip>

int main(int argc, char ** argv) {
  OsiConicSolverInterface * solver = new OsiIpoptSolverInterface();
  solver->readMps(argv[1]);
  solver->initialSolve();
  std::cout << "Objective is " << std::setprecision(15)
	    << solver->getObjValue() << std::endl;
  delete solver;
  return 0;
}
```

Currently only Mosek style MPS files are supported. You can find this example
in the examples directory.

# Install

## Basic installation

OsiIpopt is tested/works in Linux environment only for now. You can use
COIN-OR BuildTools fetch and build utility for building and installing
OsiIpopt. After cloning OsiIpopt, typing following commands should work.

```shell
git clone --branch=stable/0.8 https://github.com/coin-or-tools/BuildTools
bash BuildTools/get.dependencies.sh fetch
bash BuildTools/get.dependencies.sh build
```

First command clones BuildTools, second fetches OsiIpopt dependencies and third
builds/installs OsiIpopt.

## Installation Instructions for Advanced Users

If you already have the dependencies somewhere else in your computer and you do
not want to install them again, you are covered. First you need to make sure
that dependencies can be found with package config (```pkgconfig```
command). For this you need to add the directory that has ```.pc``` files of
dependencies to your ```PKG_CONFIG_PATH```. You can test whether the
dependencies are accesible with pkg-config with the following command,
```pkg-config --cflags --libs osi```.

Once the dependencies are accessible through pkg-config you can install
OsiIpopt by using regular "configure", "make" and "make install" sequence.
Configure script will find the dependencies through pkg-config and link
OsiIpopt to them.
