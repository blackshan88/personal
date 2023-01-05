利用deeplabv3+检测处理的mask结果，自动生成json文件；可以对新的图像样本数据进行自动标注；该代码只适合单个样本标注，如果需要多个样本标注，可以此基础上扩展

步骤：

（1）、加载deeplabv3+训练得到的pb模型，对需要标注的图片进行检测，得到分割区域，seg_map

（2）、判断分割区域的大小，如果小于整幅图像的一定尺寸，则认为没有目标，不进行标注；否则，进入下一步

（3）、利用two-pass寻找各个连同区域，找出最大区域

（4）、寻找最大区域的边缘点

（4）、任选一点作为起始点；以该点为基础，寻找距离该点最近的点；找到最近的点之后，再以此点为基础点，寻找距离最近的点，以此类推；直到最后一个边缘点
