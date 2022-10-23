python builder.py # 生成库文件libplugin.so(linux)和libplugin.dylib(Macos)

# 修改应用程序对动态库的查找路径 install_name_tool(好像要安装xcode)
install_name_tool -add_rpath /opt/anaconda2/envs/py37/lib libplugin.dylib 

# -lplugin 链接到libplugin.dylib, 也需要链接到其他的python库中, -lpython3.7m
#gfortran -ffixed-line-length-132 -o ./test32 -L./ -lplugin test32.f
#install_name_tool -add_rpath /opt/anaconda2/envs/py37/lib ./test32
#./test32

#gfortran -ffixed-line-length-132 -o ./test64 -L./ -lplugin test64.f
#install_name_tool -add_rpath /opt/anaconda2/envs/py37/lib ./test64
#./test64

gfortran -ffixed-line-length-132 -o ./test -L./ -lplugin test.f
install_name_tool -add_rpath /opt/anaconda2/envs/py37/lib ./test
./test


