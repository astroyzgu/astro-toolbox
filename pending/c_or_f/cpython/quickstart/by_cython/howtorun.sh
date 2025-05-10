# cython编译可执行文件 

cython --embed -o hello.c hello.py 
gcc -Os  -o hello hello.c -lpython3.7m -lm -I /opt/anaconda2/envs/py37/include/python3.7m/ -L /opt/anaconda2/envs/py37/lib/
export LD_LIBRARY_PATH=/opt/anaconda2/envs/py37/lib/:$LD_LIBRARY_PATH 
./hello 


cython --embed -o aaa.c aaa.pyx -3  

gcc -Os  -o hello hello.c -lpython3.7m -I /opt/anaconda2/envs/py37/include/python3.7m/ -L /opt/anaconda2/envs/py37/lib/
export LD_LIBRARY_PATH=/opt/anaconda2/envs/py37/lib/:$LD_LIBRARY_PATH 
./aaa  
