# running in Gravity
python builder.py # 生成库文件libplugin.so(linux)和libplugin.dylib(Macos)
export PYTHONPATH=$PYTHONPATH:$pwd
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$pwd # 将库文件的路径添加进来
gfortran -ffixed-line-length-132 -o ./assignwht1 -L./ -lplugin assignwht1.f
gfortran -ffixed-line-length-132 -o ./assignwht2 -L./ -lplugin assignwht2.f
gfortran -ffixed-line-length-132 -o ./mollveiw   -L./ -lplugin mollview.f

# testing 
./assignwht1  database/healpy256_5band_counts.csv database/rand1E4_for_test.dat 1
mv results.csv  test/weights1.csv
./assignwht2 database/healpy256_stat_sampling.csv database/rand1E4_for_test.dat 1
mv results.csv  test/weights2.csv

./mollview database/rand1E4_for_test.dat 1 
