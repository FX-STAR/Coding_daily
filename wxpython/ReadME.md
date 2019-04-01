## 说明
- [x] **如果有用的话，给个star拿走** 

1. 实现的跨平台Python+wxPython界面，程序中添加了OpenCV代码进行简单的图像处理（截图），适合新手借鉴，利用wxPython库进行Python界面制作，如何在Python中调用OpenCV进行图像处理。
2. 代码可以在Linux和Windows下运行，编写Windows下OpenCV3.1.0，Linux下3.1.0和3.2.0都试过，Python2和Python3环境只需要改一下print函数。因为一些依赖库的原因，wxPython我下的是最新版的，具体安装步骤到我博客看看。 
3. 这像一个图片浏览器，打开选择文件夹，遍历里面的*.jpg和*.png图片，显示在窗口上，点击按钮可以浏览上、下一张图片，鼠标画矩形，再点击按钮保存矩形图像 。窗口 下有一个进度条，显示当前浏览图像的进度，额，还有一个线程函数，适合借鉴线程参数传值。 
4. 说一下，Linux下的窗口大小和Windows下的窗口大小可能不一样，可能要你自己改一改窗口大小，因为Windows下做的有点粗糙，Linux下是完全实现上述功能的。
5. blog：[地址]（<https://blog.csdn.net/mynameisyournamewuyu/article/details/79933746>）

