# buildtools
# svn co https://projects.coin-or.org/svn/BuildTools/stable/0.8 BuildTools
# coinutils
svn co https://projects.coin-or.org/svn/CoinUtils/stable/2.10/CoinUtils CoinUtils
# get osi
svn co https://projects.coin-or.org/svn/Osi/stable/0.107/Osi Osi
# get Ipopt
svn co https://projects.coin-or.org/svn/Ipopt/releases/3.12.3/Ipopt Ipopt
# IPOPT depends the following
svn co https://projects.coin-or.org/svn/BuildTools/ThirdParty/Blas/stable/1.4   ThirdParty/Blas
cd ThirdParty/Blas
./get.Blas
cd ..
svn co https://projects.coin-or.org/svn/BuildTools/ThirdParty/Lapack/stable/1.5 ThirdParty/Lapack
cd ThirdParty/Lapack
./get.Lapack
cd ..
svn co https://projects.coin-or.org/svn/BuildTools/ThirdParty/Metis/stable/1.3  ThirdParty/Metis
cd ThirdParty/Metis
./get.Metis
cd ..
svn co https://projects.coin-or.org/svn/BuildTools/ThirdParty/Mumps/stable/1.5  ThirdParty/Mumps
cd ThirdParty/Mumps
./get.Mumps
cd ..
# get osiconic
git clone https://github.com/aykutbulut/OSI-CONIC OsiConic
