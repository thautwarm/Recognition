
问题一
=====

目标
------

识别图中物体的运行轨迹

  .. figure:: ../_static/JPGRing/Ring01.jpg

  .. figure:: ../_static/JPGRing/Ring02.jpg


问题的分析
-------

.. code:: PlainText

  对于建模问题，实际上和大数据、机器学习还是有一些差别的。

  题目里提供的数据很少，十几张图片。如果你拿所有这些图片来训练，最后过拟合不也成了吗？
  （虽然在和出题老师的讨论中，他告诉我多数人是这么做的...过拟合似乎还是标准解??（黑人问号

  所以我采取的方法是，利用第一张图片制造训练数据，训练出识别模型，并识别其他所有图片，
  对目标对象进行追踪。这样一来，如果我们的成功地构建了追踪目标的算法，那么也就显示了算法的有效性。


流程
-------

.. code:: PlainText

    1.
    先将图片转为灰度图（只是一种方式，完成可以建立识别RGB空间物体的模型）。
    找出目标物体所在的区域。为了方便数据标记，对于一个M * N的灰度图，我建立了一个和它一一对应的
    M*N标记矩阵。并取物体所在的最小矩形域，将这个矩形域标记出来。

    2.
    这个过程,在此题中如下反应：
      建立一个和图片大小规模一致的标记矩阵，初始化全0；
      找到菱形物体所在的最小矩形域，标记对应的标记矩阵区域为1；
      找到圆形物体所在的最小矩形域，标记对应的标记矩阵区域为2；

    3.
    选取合适的参数，对整个图片进行滑窗法切分。每个窗口将被视为一条数据，对应一个target。
    每个窗口是一个unsigned int8 矩阵，将其转化为浮点数矩阵，除以255以标准化。

    target的确定，我的方法是，通过计算窗口对应的标记矩阵区域的成分比例，给出一个3维向量标签。
    即target = [theta (0) , theta (1) theta (2) ]，
    记theta 是在窗口对应的标记矩阵区域上的函数，它将种类{0,1,2}中的一个映到[0,1]区间上的一个数，
    theta(i) 表示种类i的像素点在区域中所占的比例。
        P.S: 显然，窗口规模要小于圆形物体和菱形物体所在的最小矩形域。
             记窗口为row1 * col1  矩阵 M，
             圆所在最小矩形域为 row2 * col2
             菱形 row3 * col3
             则
               row1 < min(row2,row3)
               col2 < min(col2,col3)


    4.
    建立卷积神经网络，第一张图片取得的训练数据进行训练。

    5.
    将后续图片也做滑窗处理，同样是每个窗口被视为一条数据，进行标准化，利用步骤4中生成的网络对每张图
    输出预测。显然每条数据的预测结果是一个三维向量。
    在一张图片里，确定圆形和菱形物体所在区域的方法是，
      对该图片生出的所有n条数据做预测，得到一个n * 3预测矩阵Predict_Proba，并作归一化（l2正则）。
      则类别1物体（菱形）所在位置即第 argmax(Predict_Proba[:,1]) 个窗口。
      如法炮制，类别2物体（圆形）所在位置即 第argmax(Predict_Proba[:,2]) 个窗口。
      其实也就是说，选取神经网络认为是最有可能的目标。


结果如下
-----
.. figure:: http://pan.baidu.com/disk/home#list/vmode=list&path=%2Fgithub_resources%2Fresult%2Fquestion1/q11-center.png
.. figure:: http://pan.baidu.com/disk/home#list/vmode=list&path=%2Fgithub_resources%2Fresult%2Fquestion1/q11.png
.. figure:: http://pan.baidu.com/disk/home#list/vmode=list&path=%2Fgithub_resources%2Fresult%2Fquestion1/q12-center.png
.. figure:: http://pan.baidu.com/disk/home#list/vmode=list&path=%2Fgithub_resources%2Fresult%2Fquestion1/q12.png




一些其他的话
-----

和camshift算法效果比较..

.. figure:: ../result/question1/camshift

这是个坑，我查资料+写代码花了好一会儿，最后还是用opencv做的，最新版本的opencv-python文档不全，有问题可以直接issue里问我)
