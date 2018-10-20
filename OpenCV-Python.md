<a name="content">目录</a>

[openCV Python 编程](#title)
- [Part-1. 入门](#beginning)
	- [1. 简介与安装](#intro-install)
	- [2. 加载与展示图片](#image)
		- [2.1. 番外篇1：Matplotlib显示图像](#extra-imshow-matplotlib)
	- [3. 打开摄像头](#open-capture)
		- [3.1. 外篇2：滑动条](#extra-trackbar)
	- [4. 图像基本操作](#basic-operate-on-image)
	- [5. 颜色空间转换](#convert-colorspace)




<h1 name="title">openCV Python 编程</h1>

<a name="beginning"><h2>Part-1. 入门 [<sup>目录</sup>](#content)</h2></a>


<a name="install"><h2>1. 简介与安装 [<sup>目录</sup>](#content)</h2></a>

需要安装`opencv-python`、`numpy`、`matplotlib`

```
pip install opencv-python
pip install numpy
pip install matplotlib
```

Python有个重要特性：它是一门胶水语言！Python可以很容易地扩展C/C++。

OpenCV-Python就是用Python包装了C++的实现，背后实际就是C++的代码在跑，所以代码的运行速度跟原生C/C++速度一样快，而且更加容易编写。

速度比较：

<p align="center"><img src=./picture/OpenCV-Python-Intro-cv2_python_vs_cplus_time.jpg /></p>

<a name="image"><h2>2. 加载与展示图片 [<sup>目录</sup>](#content)</h2></a>

OpenCV中彩色图是以B-G-R通道存储的，灰度图只有一个通道，图像坐标的起始点是在左上角：

<p align="center"><img src=./picture/OpenCV-Python-load-image-cv2_image_coordinate_channels.jpg /></p>

<p align="center"><img src=./picture/OpenCV-Python-load-image-watch.jpg width=600 /></p>

```
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE) # 读入图片，并指定读入的形式，默认为'IMREAD_COLOR'
cv2.imshow('image',img) # 显示图片
cv2.waitKey(0) # 让程序暂停，参数是等待时间（毫秒ms）。时间一到，会继续执行接下来的程序，传入0的话表示一直等待。等待期间也可以获取用户的按键输入：k = cv2.waitKey(0)
cv2.destroyAllWindows() # 关闭所有图像窗口
```

- **加载图片**

使用cv2.imread()来读入一张图片：

```
import cv2
# 灰度图加载
img = cv2.imread('lena.jpg', 0)
```

参数说明：

> 1. 参数1：图片的文件名
> 2. 参数2：读入方式，省略即采用默认值
> 	- cv2.IMREAD_COLOR：彩色图，默认值(1)
> 	- cv2.IMREAD_GRAYSCALE：灰度图(0)
> 	- cv2.IMREAD_UNCHANGED：包含透明通道的彩色图(-1)

- **展示图片**

使用cv2.imshow()显示图片，窗口会自适应图片的大小：

```
cv2.imshow('lena', img)
cv2.waitKey(0)
```

参数1是窗口的名字，参数2是要显示的图片。

我们也可以先用cv2.namedWindow()创建一个窗口，之后再显示图片：

```
# 先定义窗口，后显示图片
cv2.namedWindow('lena2', cv2.WINDOW_NORMAL)
cv2.imshow('lena2', img)
cv2.waitKey(0)
```

参数1依旧是窗口的名字，参数2默认是cv2.WINDOW_AUTOSIZE，表示窗口大小自适应图片，也可以设置为cv2.WINDOW_NORMAL，表示窗口大小可调整。图片比较大的时候，可以考虑用后者。

- **保存图片**

使用cv2.imwrite()保存图片：

```
cv2.imwrite('lena_gray.jpg', img)
```

<a name="extra-imshow-matplotlib"><h3>2.1. 番外篇1：Matplotlib显示图像 [<sup>目录</sup>](#content)</h3></a>

- **显示灰度图**

```
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('lena.jpg', 0)
# 灰度图显示，cmap(color map)设置为gray
plt.imshow(img, cmap='gray')
plt.show()
```

- **显示彩色图**

**OpenCV中的图像是以BGR的通道顺序存储的**，但Matplotlib是以RGB模式显示的，所以直接在Matplotlib中显示OpenCV图像会出现问题，因此需要转换一下:

```
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('lena.jpg')
img2 = img[:, :, ::-1]
# 或使用
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 显示不正确的图
plt.subplot(121),plt.imshow(img) 
# 显示正确的图
plt.subplot(122)
plt.xticks([]), plt.yticks([]) # 隐藏x和y轴
plt.imshow(img2)
plt.show()
```

注：

> 注解：`img[:,:,0]`表示图片的蓝色通道，熟悉Python的同学应该知道，对一个字符串s进行翻转用的是`s[::-1]`，同样`img[:,:,::-1]`就表示BGR通道翻转，变成RGB

<a name="open-capture"><h2>3. 打开摄像头 [<sup>目录</sup>](#content)</h2></a>

- **打开摄像头**

解决opencv无法打开树莓派上的CSI摄像头的问题：

> - 原因分析：
> 
> 树莓派专用CSI摄像头插到树莓派的CSI口上并在在raspi-config中打开后就可以使用Raspistill命令直接使用，但如果在OpenCV中调用CSI摄像头会出现无数据的现象（cv2.VideoCapture（0）这时不会报错）。
> 
> 这是因为树莓派中的camera module是放在/boot/目录中以固件形式加载的，不是一个标准的V4L2的摄像头驱动，所以加载起来之后会找不到/dev/video0的设备节点。
> - 解决方法：
> 
> 在/etc/modules里面添加一行bcm2835-v4l2（小写的L）就能解决问题
> 
> 注意：修改后需要重启，才能是刚才的修改生效

要使用摄像头，需要使用cv2.VideoCapture(0)创建VideoCapture对象，参数：0指的是摄像头的编号。如果你电脑上有两个摄像头的话，访问第2个摄像头就可以传入1。

```
# 打开摄像头并灰度化显示
import cv2
capture = cv2.VideoCapture(0)
while(True):
    # 获取一帧
    ret, frame = capture.read()
    # 将这帧转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
```

函数参数说明：

> - `capture.read()`
> 
> 函数返回的第1个参数ret(return value缩写)是一个布尔值，表示当前这一帧是否获取正确
> 
> - `cv2.cvtColor()`用来转换颜色，这里将彩色图转成灰度图

另外，通过cap.get(propId)可以获取摄像头的一些属性，比如捕获的分辨率，亮度和对比度等。propId是从0~18的数字，代表不同的属性，完整的属性列表可以参考：VideoCaptureProperties。也可以使用cap.set(propId,value)来修改属性值。比如说，我们在while之前添加下面的代码：

```
# 获取捕获的分辨率
# propId可以直接写数字，也可以用OpenCV的符号表示
width, height = capture.get(3), capture.get(4)
print(width, height)
# 以原分辨率的一倍来捕获
capture.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height * 2)
```

- **播放本地视频**

跟打开摄像头一样，如果把摄像头的编号换成视频的路径就可以播放本地视频了。回想一下cv2.waitKey()，它的参数表示暂停时间，所以这个值越大，视频播放速度越慢，反之，播放速度越快，通常设置为25或30。

```
# 播放本地视频
capture = cv2.VideoCapture('demo_video.mp4')
while(capture.isOpened()):
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    if cv2.waitKey(30) == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
```

- **录制视频**

也可以将摄像头捕捉到的画面写入视频文件中

要保存视频文件需要创建一个VideoWriter对象，需要给它传入四个参数：

- 输出的文件名，如'output.avi'
- 编码方式FourCC码
- 帧率FPS
- 要保存的分辨率大小

FourCC是用来指定视频编码方式的四字节码，所有的编码可参考[Video Codecs](http://www.fourcc.org/codecs.php)。如MJPG编码可以这样写：`cv2.VideoWriter_fourcc(*'MJPG')`或`cv2.VideoWriter_fourcc('M','J','P','G')`

```
capture = cv2.VideoCapture(0)
# 定义编码方式并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
outfile = cv2.VideoWriter('output.avi', fourcc, 25., (640, 480))
while(capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        outfile.write(frame)  # 写入文件
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
capture.release()
outfile.release()
cv2.destroyAllWindows()
```

注意：当从视频文件中读取视频，并想将修改后的内容输出到新视频文件中，此时需要保证输出视频文件的fourcc,fps,长和宽与原视频一致，此时需要通过`get( )`获取原视频的一些参数：

```
width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)) # 视频的高度
height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 视频的帧率
fps = reader.get(cv2.CAP_PROP_FPS) # 视频的编码
fourcc = int(reader.get(cv2.CAP_PROP_FOURCC)) # 定义视频输出
writer = cv2.VideoWriter("./Videos/out.mp4", fourcc, fps, (width, height))
```

<a name="extra-trackbar"><h3>3.1. 外篇2：滑动条 [<sup>目录</sup>](#content)</h3></a>

首先我们需要创建一个滑动条，如`cv2.createTrackbar('R','image',0,255,call_back)`，其中

> - 参数1：滑动条的名称
> - 参数2：所在窗口的名称
> - 参数3：当前的值
> - 参数4：最大值
> - 参数5：回调函数名称，回调函数默认有一个表示当前值的参数

创建好之后，可以在回调函数中获取滑动条的值，也可以用：`cv2.getTrackbarPos()`得到，其中，参数1是滑动条的名称，参数2是窗口的名称。下面我们实现一个RGB的调色板，理解下函数的使用：

```
import cv2
import numpy as np
# 回调函数，x表示滑块的位置，本例暂不使用
def nothing(x):
    pass
img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('image')
# 创建RGB三个滑动条
cv2.createTrackbar('R', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)
while(True):
    cv2.imshow('image', img)
    if cv2.waitKey(1) == 27:
        break
    # 获取滑块的值
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    # 设定img的颜色
    img[:] = [b, g, r]
```

<p align="center"><img src=./picture/OpenCV-Python-capture-cv2_track_bar_rgb.jpg width=600 /></p>

<a name="basic-operate-on-image"><h2>4. 图像基本操作 [<sup>目录</sup>](#content)</h2></a>

- **获取和修改像素点值**

通过行列的坐标来获取某像素点的值，对于彩色图，这个值是B,G,R三个值的列表，对于灰度图，只有一个值：

```
px = img[100, 100]
print(px)  # [119 108 201]
# 只获取蓝色blue通道的值
px_blue = img[100, 100, 0]
print(px_blue)  # 119
```

修改像素的值也是同样的方式：

```
img[100, 100] = [255, 255, 255]
print(img[100, 100])  # [255 255 255]
```

- **图片属性**

`img.shape`获取图像的形状，图片是彩色的话，返回一个包含高度、宽度和通道数的元组，灰度图只返回高度和宽度：

```
print(img.shape)  # (263, 263, 3)
# 形状中包括高度、宽度和通道数
height, width, channels = img.shape
# img是灰度图的话：height, width = img.shape
```

`img.dtype`获取图像数据类型：

```
print(img.dtype)  # uint8
```

`img.size`获取图像总像素数：

```
print(img.size)  # 263*263*3=207507
```

- **ROI**

ROI：region of interest，感兴趣区域

<p align="center"><img src=./picture/OpenCV-Python-cv2_lena_face_roi_crop.jpg width=600 /></p>

```
# 截取脸部ROI
face = img[100:200, 115:188]
cv2.imshow('face', face)
cv2.waitKey(0)
```

- **通道分割与合并**

彩色图的BGR三个通道是可以分开单独访问的，也可以将单独的三个通道合并成一副图像。分别使用`cv2.split()`和`cv2.merge()`：

```
b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))
```

`split()`函数比较耗时，更高效的方式是用numpy的索引，如提取B通道：

```
b = img[:, :, 0]
cv2.imshow('blue', b)
cv2.waitKey(0)
```

<a name="convert-colorspace"><h2>5. 颜色空间转换 [<sup>目录</sup>](#content)</h2></a>

- **颜色空间转换**

```
import cv2
import numpy as np
img = cv2.imread('lena.jpg')
# 转换为灰度图
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', img)
cv2.imshow('gray', img_gray), cv2.waitKey(0)
```

cv2.cvtColor()用来进行颜色模型转换，参数1是要转换的图片，参数2是转换模式，COLOR_BGR2GRAY表示BGR→Gray

- **视频中特定颜色物体追踪**

HSV是一个常用于颜色识别的模型，相比BGR更易区分颜色，转换模式用COLOR_BGR2HSV表示。

现在，我们实现一个使用HSV来只显示视频中蓝色物体的例子，步骤如下：

> - 捕获视频中的一帧
> - 从BGR转换到HSV
> - 提取蓝色范围的物体
> - 只显示蓝色物体

<p align="center"><img src=./picture/OpenCV-Python-convert-colorspace-cv2_blue_object_tracking.jpg width=600 /></p>

```
capture = cv2.VideoCapture(0)

# 蓝色的范围，不同光照条件下不一样，可灵活调整
lower_blue = np.array([100, 110, 110])
upper_blue = np.array([130, 255, 255])

while(True):
    # 1.捕获视频中的一帧
    ret, frame = capture.read()

    # 2.从BGR转换到HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 3.inRange()：介于lower/upper之间的为白色，其余黑色
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 4.只保留原图中的蓝色部分
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    if cv2.waitKey(1) == ord('q'):
        break
```

那蓝色的HSV值的lower和upper范围是怎么得到的呢？其实很简单，我们先把标准蓝色的值用cvtColor()转换下：

```
blue = np.uint8([[[255, 0, 0]]])
hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
print(hsv_blue)  # [[[120 255 255]]]
```








---

参考资料：

(1) [Image and Video Analysis](https://pythonprogramming.net/loading-images-python-opencv-tutorial/)

(2) [ex2tron's Blog，【视觉与图像】OpenCV篇：Python+OpenCV实用教程 ](http://ex2tron.wang/opencv-python/)

(3) [csdn博客：【树莓派】在OpenCV中调用CSI摄像头](https://blog.csdn.net/Deiki/article/details/71123947?utm_source=blogxgwz1)

(4) [Python OpenCV3 读取和保存视频和解决保存失败的原因分析](https://blog.csdn.net/DumpDoctorWang/article/details/80515861)
