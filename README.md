# KC_CUDA

[CCF BDCI 2020 大规模图数据中的kmax-truss求解问题](https://www.datafountain.cn/competitions/473) 赛题复赛运行性能第1名解决方案  
本开源优化了程序中一些冗余部分，开源版本较复赛以及初赛提交版本性能有些许的提高  

## 程序代码说明
+ GPU源程序共有3个文件：kc_gpu.cpp， kc<span></span>.cu， mapfile.hpp
    - kc_gpu.cpp：内容包括主程序、由CPU计算实现的子函数、与CUDA的通信接口
    - kc<span></span>.cu：为GPU计算模块，GPU实现的函数
    - mapfile.hpp：mmap函数调用函数

+ CPU源程序共有2个文件：
    - kc_cpu.cpp：CPU版本的程序实现
    - mapfile.hpp：同上

## 程序代码编译说明
+ 程序运行依赖vmtouch加速读取速度，开源版本将此部分独立出代码，需要自行安装
    - debain或者ubuntu可以使用sudo apt install vmtouch命令进行安装
    - 其他安装方法可以参考[vmtouch项目](https://github.com/hoytech/vmtouch)
+ 编译需求g++以及nvcc，只需要进入src目录执行make命令即可
+ 注意本程序的makefile中的gpu-architecture以及gpu-code需要改成机器对应的显卡架构，默认为30系显卡架构

## 程序代码运行使用说明
+ 运行程序请用命令格式： 
```
./kc_cpu -f [数据所在目录/图数据文件]
./kc_gpu -f [数据所在目录/图数据文件]

```
例如：
```
./kc_gpu -f ../data/ktruss_example.tsv
```

## 程序输入数据格式
+ 程序支持的输入格式详见[赛题数据说明](https://www.datafountain.cn/competitions/473/datasets)
+ [数据下载地址](http://datafountain.int-yt.com/Files/BDCI2020/473HuaKeDaKtruss/ktruss-data.zip)
+ 更多数据下载，请前往[Stanford Network Analysis Project](http://snap.stanford.edu/)
