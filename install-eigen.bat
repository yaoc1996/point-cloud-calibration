cd %~dp0
mkdir eigen
cd eigen
curl https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip --output eigen.zip
tar -xf eigen.zip
cd eigen-3.4.0
mkdir build
cd build
cmake ..
cmake --build . --target install
cd ..
cd ..
cd ..
del /S /Q eigen
rmdir /S /Q eigen
PAUSE