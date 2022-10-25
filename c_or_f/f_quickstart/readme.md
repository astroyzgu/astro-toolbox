allocate-test.f 生成动态二维数组，并填入矩阵
module-test.f   通过module定义子程序 
1. https://blog.csdn.net/weixin_43880667/article/details/84836145
2. https://www.yiibai.com/fortran/fortran_intrinsic_functions.html 

#--------------------------------------------------------------------
# 认识库文件 
3. https://blog.csdn.net/Heyyellman/article/details/111600752
#---------------------------------------------------------------------
# Fortran程序的简单编译： 
  gfortran -o hello_world hello_world.f

#---------------------------------------------------------------------
# 在编译过程中调用外部函数库： 
  gfortran -o hello_world hello_world.f -lxxx -L/path/to/lib -I/path/to/head 
调用xxx库函数, -lxxx(-l是lib的意思，xxx为库名)
库文件的位置由 -L 后的路径给出。 
头文件则从 -I 后的路径里去找。

# xxx.so是动态函数库
# xxx.a是静态函数库。
对于动态函数库(优先)，在运行执行文件时，仍需要从.so库文件中读取函数信息。
对于静态函数库在编译的时直接整合到可执行文件，程序可以独立运行的。 

#---------------------------------------------------------------------
# 在环境变量中提供动态库文件的位置
执行编译好的可执行文件hello_world
   ./hello_world
执行可能会报错错误，是因为编译是用到了libxxx.so, 运行时得再次访问。
在环境变量（.bashrc）中提供动态库文件所在的位置。
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/lib

#---------------------------------------------------------------------
# 自定义库函数文件
将源文件编译为xxx.o目标文件: 
  gfortran -c xxx.f 
产生的xxx.o文件即为目标文件(目标文件指源代码经过编译程序产生的且能被cpu直接识别二进制代码)。再将目标文件编译成可执行文件：
  gfortran -o xxx xxx.o 
函数库就是将很多xxx.o 和在一起形成的。 
  gfortran -fPIC -shared -o libxxx.so xxx.f 
调用刚刚生成的函数库
  gfortran -o run run.f -lxxx -L./ 


#---------------------------------------------------------------------
# 初识makefile文件
4. https://zhuanlan.zhihu.com/p/341439169
