# C++模板元编程实战

阅读《[C++模板元编程实战：一个深度学习框架的初步实现](https://book.douban.com/subject/30394402/)》的同步代码实现。

直接参考来源：
- [bluealert/MetaNN-book](https://github.com/bluealert/MetaNN-book)
- [liwei-cpp/MetaNN](https://github.com/liwei-cpp/MetaNN)
- 在其基础上使用C++20进行了一定程度重构和重新组织。

编码风格与特点：
- 使用C++20标准，仅头文件（Header-Only），添加[`./MetaNN`](./MetaNN/)到包含目录即可使用。
- 所有代码包含在命名空间 `MetaNN` 中。
- 4空格缩进，大括号换行，类名大驼峰，函数与变量名小驼峰，文件采用全小写下划线连接。
- 元函数命名：
    - 使用内嵌类型或者常量作为输出的元函数使用下划线`_`结尾。
    - 而对应直接作为输出结果的原函数，比如变量模板、别名模板则在其基础上去掉末尾`_`。
    - 概念都以`C`后缀结尾。

运行测试：
```shell
cd ./test
make run
```