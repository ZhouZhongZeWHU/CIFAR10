### 一、简单介绍：
CIFAR10数据集是一个用于识别普适物体的小型数据集,一共包含 10 个类别的 RGB 彩色图 片：飞机（ a叩lane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。共有60000张彩色图像，这些图像式32*32*3，分为10个类，每个类6000张,这里面有50000张用于训练，构成5个训练批，每一批10000张图；另外10000张用于测试，单独构成一批。测试批的数据里，取自10类中的每一类，每一类随机取1000张。现采用卷积神经网络对十类图片做分类任务，并调节参数分析作用。
### 二、数据读取与展示：
安装第三方包torchvision，使用如下函数读取数据集：

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,shuffle=True, num_workers=2)
可使用函数来展示一张图片：

    def imshow(img):
        img = img / 2 + 0.5     
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
### 二、模型选择与参数设置：
#### 网络采用两个卷积层两个池化层三个全连接层实现： 

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            #卷积层1输入为3（3通道）
            self.conv1 = nn.Conv2d(3, C1Out, C1Kernel)
            #池化层，大小为2
            self.pool = nn.MaxPool2d(PoolSize, 2)
            #卷积层2
            self.conv2 = nn.Conv2d(C1Out,C2Out , C2Kernel)
            #全连接层，输入要根据之前的卷积层确定
            self.fc1 = nn.Linear(C2Out * 5 * 5, linearNum2)
            self.fc2 = nn.Linear(linearNum2, linearNum3)
            self.fc3 = nn.Linear(linearNum3, 10)
        def forward(self, x):
            #前向传播，利用同一个池化层
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x  
#### 可以调节的参数有：
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
### 三、初始参数设置：  
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
1.	学习率（learnL）和学习轮次（epochNum）调整：较大的学习率加速了网络训练，但可能无法达到最优解。较小的学习率会使网络训练缓慢，也可能会使网络陷入局部最优解。学习率过大的情况小，网络无法学习到有效的知识。初始参数的学习率过小，导致损失函数收敛很慢，而训练轮次过少导致网络还未到达最优解就停止了学习。考虑增大学习率到0.01（不能过大以免越过最低点）。Epochs过大容易导致过拟合，而过小则网络训练不足，学习轮次提高到5。结果：损失在每一轮次间下降很快，在轮次中会上下波动，最后准确率达到61%，损失达到0.110。
2.	批大小（batchSize）和动量（mom）调整：当 momentum 动量越大时，其转换为势能的能量也就越大，就越有可能摆脱局部凹域的束缚，进入全局凹域，将动量增大到0.92。Batch Size主要影响的是每个Epoch的训练时长和每次迭代的梯度平滑度。批大小若过小会导致迭代的梯度不平滑且内存利用率低，过大会导致容易陷入局部最优或者内存溢出。根据经验一般是数据量开根号，这里调整到50。结果：与上一次调整区别不大。
3.	卷积层通道数调整（C1Out C2Out）：通道数越多可能可以提取出更加精细的信息，卷积层1输出通道数调整为9，卷积层2输出通道数调整为18。结果： 最后一轮训练中损失在0.105波动，准确率略微提升至63%。
4.	卷积核大小调整（C1Kernel C2Kernel）：尝试调整第一层卷积核大小为9，第二层不变（若第一、二层卷积核大小改变为7、9则输入到全连接层的尺寸会过小）。结果：准确率下降到55%，卷积核过大，恢复为初始卷积核大小。
5.	全连接层神经元个数调整（linearNum2 linearNum3）：现考虑输入输出层共有四层，输入神经元个数是18\*5*5，输出神经元个数是10，现遵从习惯将各层个数调整为450,256,64,10   
最后结果为：  
  
     ![参数调整后效果](https://github.com/ZhouZhongZeWHU/CIFAR10/blob/main/afterResult.png)相比初始参数，效果提升了百分之百以上，经过五轮学习损失降低至0.094，准确率达到64%。

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
#### 还有Dropout Rate、Activation Function、Network Depth等参数设置，本次暂未涉及

#### 代码见main.py
