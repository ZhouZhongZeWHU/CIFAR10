import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


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


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
       #正则化
    def imshow(img):
        img = img / 2 + 0.5     
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    import torch.nn as nn
    import torch.nn.functional as F


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


    net = Net()

    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learnL, momentum=mom)

    for epoch in range(epochNum):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    net = Net()
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
