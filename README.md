### 一、简单介绍：
CIFAR10数据集是由十类32\*32*3的彩色图像组成的，如飞机、鸟等类别下各有若干图片，现采用卷积神经网络对十类图片做分类任务，并调节参数分析作用。
### 二、模型选择：
#### 网络采用两个卷积层两个池化层三个全连接层实现。可以调节的参数有：  
卷积层1的输出通道数：C1Out  
卷积层1的卷积核大小：C1Kernel  
卷积层1的步长（选择默认不改变）：C1Step  
卷积层2的输出通道数：C2Out  
卷积层2的卷积核大小：C2Kernel  
卷积层2的步长（选择默认不改变）：C2Step  
池化层大小（选择2不改变）：PoolSize  
全连接层第二层神经元个数：linearNum2  
全连接层第三层神经元个数：linearNum3  
学习轮次：epochNum  
学习率：learnL  
学习动量大小：mom  
批大小：batchSize  
池化层方式：最大池化  
### 三、初始参数设置及效果：  
C1Out=6            #卷积层1的输出通道数  
C1Kernel=5         #卷积层1的卷积核大小  
C1Step=1           #卷积层1的步长（选择默认不改变）  
C2Out=12           #卷积层2的输出通道数  
C2Kernel=5         #卷积层2的卷积核大小  
C2Step=1           #卷积层2的步长（选择默认不改变）  
PoolSize=2         #池化层大小（选择2不改变）  
linearNum2=120     #全连接层第二层神经元个数  
linearNum3=84      #全连接层第三层神经元个数  
epochNum=2         #学习轮次  
learnL=0.001        #学习率  
mom=0.9           #学习动量大小  
batchSize=32       #批大小   

效果如下： ![参数调整前效果](https://github.com/ZhouZhongZeWHU/CIFAR10/blob/main/beforeResult.png)
可见，准确率不高只有30%，经过每两百个图片的学习后损失下降也很慢，从0.229降低到了0.190
### 四、参数调整
1.	学习率（learnL=0.001）和学习轮次（epochNum=2）调整：初始学习率过小，导致损失函数收敛很慢，而训练轮次过少导致网络还未到达最优解就停止了学习。考虑增大学习率到0.01（不能过大以免越过最低点），学习轮次提高到5（过高会导致电脑负荷增大，也有过拟合风险）。结果：损失在每一轮次间下降很快，在轮次中会上下波动，最后准确率达到61%，损失达到0.110。
2.	批大小（batchSize=32）和动量调整（mom=0.9）：当 momentum 动量越大时，其转换为势能的能量也就越大，就越有可能摆脱局部凹域的束缚，进入全局凹域，将动量增大到0.92，批大小根据经验一般是数据量开根号，这里调整到50。结果：与上一次调整区别不大。
3.	卷积层通道数调整（C1Out=6 C2Out=12）：通道数越多可能可以提取出更加精细的信息，卷积层1输出通道数调整为9，卷积层2输出通道数调整为18。结果： 最后一轮训练中损失在0.105波动，准确率略微提升至63%。
4.	卷积核大小调整（C1Kernel=5 C2Kernel=5）：尝试调整第一层卷积核大小为9，第二层不变（若第一、二层卷积核大小改变为7、9则输入到全连接层的尺寸会过小）。结果：准确率下降到55%，卷积核过大，恢复为初始卷积核大小。
5.	全连接层神经元个数调整（linearNum2=120 linearNum3=84）：现考虑输入输出层共有四层，输入神经元个数是18\*5*5，输出神经元个数是10，现遵从习惯将各层个数调整为450,256,64,10，最后结果为： ![参数调整后效果](https://github.com/ZhouZhongZeWHU/CIFAR10/blob/main/afterResult.png)相比初始参数，效果提升了百分之百以上，经过五轮学习损失降低至0.094，准确率达到64%。

最后的参数设置为：  
C1Out=9            #卷积层1的输出通道数  
C1Kernel=5         #卷积层1的卷积核大小  
C1Step=1           #卷积层1的步长（选择默认不改变）  
C2Out=18           #卷积层2的输出通道数  
C2Kernel=5         #卷积层2的卷积核大小  
C2Step=1           #卷积层2的步长（选择默认不改变）  
PoolSize=2         #池化层大小（选择2不改变）  
linearNum2=256     #全连接层第二层神经元个数  
linearNum3=64      #全连接层第三层神经元个数  
epochNum=5         #学习轮次  
learnL=0.01        #学习率  
mom=0.92           #学习动量大小  
batchSize=50       #批大小  

#### 代码见main.py
