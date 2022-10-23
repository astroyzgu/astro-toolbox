#include <iostream>  // 声明标准的输入输出头文件
#include <string>
using namespace std; // 定义命名空间 
// extern "C"{ 

int main () {
        std::cout << "Hello World!";
        return 0;
}

void hehe1 () {
        std::cout << "hehe1!";
}

int  hehe2 () {
        std::cout << "hehe2!";
        return 0;
}
    
int cm(int n){ 
    int cm=0;
    int i; 
    for(i=1;i<=n;++i){
        cm += i;
    }
    std::cout << "running sum!";
    printf("running sum"); 
    // std::cout << cm;
    return(cm);
} // g++ -o hello.so -shared -fPIC hello.cpp


// }
