# 导入所需要的库
import os  # os库包含几百个函数，常用路径操作、进程管理、环境参数等几类
import zipfile  # 用于解压zip文件
import random
import json  # JSON库是一种轻量级的数据交换格式，易于阅读和编写

import matplotlib.pyplot as plt
import paddle
import paddle.fluid as fluid
import numpy as np
from multiprocessing import cpu_count  # 用于多进程处理
from visualdl import LogWriter
from PIL import Image

# 自定义参数
BATCH_SIZE = 32  # 表示每一批读入的数量
IMG_H = 24  # 设定图片的高度
IMG_W = 24  # 设定图片的宽度
DATA_TRAIN = 0.4  # 设定训练模型时所用图片的比例
BUFFER_SIZE = 1024  # 数据缓冲区大小
USE_CUDA = True  # 是否使用CPU
EPOCH_NUM = 10  # 训练次数
model_save_dir = "work/model"
LEARNING_RATE = 0.0001

# 定义项目相关函数
'''
函数 unzipFile(source_file, target_dir)
功能: 将zip文件解压到指定路径
'''


def unzipFile(source_file, target_dir):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    zf = zipfile.ZipFile(source_file)
    try:
        zf.extractall(path=target_dir)
    except RuntimeError as e:
        print(e)
    zf.close()
    print('---原始数据文件解压完成---')


'''
函数 genDataList(dataRootPath, trainPercnet=0.8)
功能: 将数据集划分为训练集和验证集，然后生成图片的list文件，即paddlepaddle能获取信息的形式
返回值: 类别的数量
'''


def genDataList(dataRootPath, trainPercnet=0.8):
    classDirs = os.listdir(dataRootPath)  # os.listdir()用来获取指定文件夹中所有文件和子文件夹名称组成的列表
    classDirs = [x for x in classDirs if os.path.isdir(os.path.join(dataRootPath, x))]
    # os.path.isdir()函数判断某一路径是否为目录  os.path.join()函数用于路径拼接文件路径
    listDirTest = os.path.join(dataRootPath, 'test.list')
    listDirTrain = os.path.join(dataRootPath, 'train.list')
    # 清空原来的数据
    with open(listDirTest, 'w') as f:
        pass
    with open(listDirTrain, 'w') as f:
        pass
    # 随机划分训练集与测试集
    classLabel = 0  # 记录类别的标签编号
    class_detail = []  # 记录每个类别的描述
    classList = []  # 记录所有的类别名
    num_images = 0  # 统计图片的总数量
    for classDir in classDirs:
        classPath = os.path.join(dataRootPath, classDir)
        imgPaths = os.listdir(classPath)
        # 从中取trainPercent作为训练集
        imgIndex = list(range(len(imgPaths)))
        random.shuffle(imgIndex)  # random.shuffle()函数用于将一个列表中的元素打乱顺序
        imgIndexTrain = imgIndex[:int(len(imgIndex) * trainPercnet)]
        imgIndexTest = imgIndex[int(len(imgIndex) * trainPercnet):]
        with open(listDirTest, 'a') as f:  # 模式'a'打开一个文件用于追加
            for imgIndex in imgIndexTest:
                imgPath = os.path.join(classPath, imgPaths[imgIndex])
                f.write(imgPath + '\t%d' % classLabel + '\n')
        with open(listDirTrain, 'a') as f:  # 模式'a'打开一个文件用于追加
            for imgIndex in imgIndexTrain:
                imgPath = os.path.join(classPath, imgPaths[imgIndex])
                f.write(imgPath + '\t%d' % classLabel + '\n')

        num_images += len(imgPaths)

        classList.append(classDir)
        class_detail_list = {}
        class_detail_list['class_name'] = classDir  # 类别名称
        class_detail_list['class_label'] = classLabel  # 类别标签
        class_detail_list['class_test_images'] = len(imgIndexTest)  # 该类数据的测试集数目
        class_detail_list['class_trainer_images'] = len(imgIndexTrain)  # 该类数据的训练数目
        class_detail.append(class_detail_list)
        classLabel += 1

    # 说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = classList  # 所有的标签
    readjson['all_class_sum'] = len(classDirs)  # 总类别数量
    readjson['all_class_images'] = num_images  # 总图片数量
    readjson['class_detail']  # 每种类别情况
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ':'))
    with open(os.path.join(dataRootPath, 'readme.json'), 'w') as f:
        f.write(jsons)
    print('---生成数据列表完成！共有%d个标签' % readjson['all_class_sum'])
    print('---标签分别是:', classList)
    # 返回标签的数量
    return readjson['all_class_sum']


'''
功能: 对训练集的图像进行处理（修剪和数组变换），返回img数组和标签，sample是一个python元组，里面保存着图片的地址和标签
'''


def trainMapper(sample):
    global IMG_H
    global IMG_W
    img, label = sample
    # 读取图像
    img = paddle.dataset.image.load_image(img)
    # 对图像进行修剪并转换为数组
    img = paddle.dataset.image.simple_transform(
        im=img,
        resize_size=IMG_H,
        crop_size=IMG_W,
        is_color=True,
        is_train=True
    )
    # img数组归一化处理，得到0到1之间的数值
    img = img.flatten().astype('float32') / 255.0
    return img, label


def trainReader(train_list):
    global DATA_TRAIN
    global BUFFER_SIZE

    def reader():
        with open(train_list, 'r') as f:
            lines = [line.strip() for line in f]
            # strip()方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
            np.random.shuffle(lines)
            lines = lines[:int(len(lines) * DATA_TRAIN)]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)  # yield在返回数据的同时，还保存了当前的执行内容

    return paddle.reader.xmap_readers(trainMapper, reader, cpu_count(), BUFFER_SIZE)


def testMapper(sample):
    global IMG_H
    global IMG_W
    img, label = sample
    # 读取图像
    img = paddle.dataset.image.load_image(img)
    # 对图像进行修剪并转换为数组
    img = paddle.dataset.image.simple_transform(
        im=img,
        resize_size=IMG_H,
        crop_size=IMG_W,
        is_color=True,
        is_train=True
    )
    # img数组归一化处理，得到0到1之间的数值
    img = img.flatten().astype('float32') / 255.0
    return img, label


def testReader(test_list):
    global DATA_TRAIN
    global BUFFER_SIZE

    def reader():
        with open(test_list, 'r') as f:
            lines = [line.strip() for line in f]
            # strip()方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
            np.random.shuffle(lines)
            lines = lines[:int(len(lines) * DATA_TRAIN)]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)  # yield在返回数据的同时，还保存了当前的执行内容

    return paddle.reader.xmap_readers(testMapper, reader, cpu_count(), BUFFER_SIZE)
    # paddle.reader.xmap_readers()函数的参数理解
    # testMapper从reader生成器里获得一个个数据之后会先使用testMapper的函数来处理数据
    # reader参数是一个生成器函数
    # cpu_count()线程数
    # BUFFER_SIZE:指一次读取多少个数据


'''
函数 createDataReader(BATCH_SIZE)
功能: 用于构建读取训练集和测试集数据的读取器
返回值: 训练集读取器和测试集读取器
'''


def createDataReader(BATCH_SIZE):
    global BUFFER_SIZE
    trainer_reader = trainReader(train_list=os.path.join(dataRootPath, 'train.list'))
    train_reader = paddle.batch(paddle.reader.shuffle(reader=trainer_reader, buf_size=BUFFER_SIZE),
                                batch_size=BATCH_SIZE)
    # paddle.batch()把数据封装成输出一个个Batch的形式
    tester_reader = testReader(test_list=os.path.join(dataRootPath, 'test.list'))
    test_reader = paddle.batch(tester_reader, batch_size=BATCH_SIZE)
    print('---train_reader, test_reader创建完成！')
    return train_reader, test_reader


'''
函数 setNN(predict, image, label, LEARNING_RATE)
功能: 用于配置神经网络的损失函数和优化算法、计算准确率、创建执行器、定义映射器、定义测试程序
返回值: 损失函数avg_loss、 准确率acc、 执行器exe、 数据映射器feeder以及测试程序test_program
'''


def setNN(predict, image, label, LEARNING_RATE):
    global USE_CUDA
    # 定义损失函数
    loss = fluid.layers.cross_entropy(input=predict, label=label)  # 交叉熵损失函数
    avg_loss = fluid.layers.mean(loss)  # 计算（每一批的）平均损失值
    # 定义优化方法
    optimizer = fluid.optimizer.Adam(learning_rate=LEARNING_RATE)  # Adagrad优化算法
    optimizer.minimize(avg_loss)
    # 计算准确率acc
    acc = fluid.layers.accuracy(input=predict, label=label)
    # 创建执行器exe
    place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()  # 指定训练环境，是否使用GPU
    exe = fluid.Executor(place)  # 创建执行器
    exe.run(fluid.default_startup_program())  # 初始化化执行器
    # 定义映射器feeder
    feeder = fluid.DataFeeder(feed_list=[image, label], place=place)
    # 定义测试程序
    test_program = fluid.default_main_program().clone(for_test=True)
    return avg_loss, acc, exe, feeder, test_program


'''
函数 draw_train_process(epoch, dictLossAcc)
功能: 绘制训练过程中，每次迭代损失值、准确值的变化，和每次训练后训练集和测试集的损失值、准确值的变化
'''


def draw_figure(dictLossAcc, xlabel, ylabel_1, ylabel_2):
    '''
    :param dictLossAcc: 为一个字典类型
    :param xlabel:是一个key值，表示需要取出dicLossAcc中key值为xlabel的数据
    '''
    plt.xlabel(xlabel, fontsize=20)
    plt.plot(dictLossAcc[xlabel], dictLossAcc[ylabel_1], color='red', label=ylabel_1)
    plt.plot(dictLossAcc[xlabel], dictLossAcc[ylabel_2], color='green', label=ylabel_2)
    plt.legend()
    plt.grid()  # grid()函数用于设置绘图区网格线


def draw_train_process(epoch, dictLossAcc):
    '''
    dicLossAcc是一个字典，存储了画图所需的数据
    key值包括：iter_loss iter_acc: 每次迭代后，训练集的损失值和准确值
    loss_train loss_test: 完成epoch后，训练集和测试集的损失值
    acc_train acc_test: 完成epoch后训练集和测试集的准确度
    '''
    plt.figure(figsize=(10, 3))
    plt.title('epoch - ' + str(epoch), fontsize=24)
    plt.subplot(1, 3, 1)
    draw_figure(dictLossAcc, 'iteration', 'iter_loss', 'iter_acc')
    plt.subplot(1, 3, 2)
    draw_figure(dictLossAcc, 'epoch', 'loss_train', 'loss_test')
    plt.subplot(1, 3, 3)
    draw_figure(dictLossAcc, 'epoch', 'acc_train', 'acc_test')
    plt.show()


'''
函数 training(image, label, predict, avg_loss, acc, exe, feeder, test_program)
功能: 用于模型训练
'''


def training(image, label, predict, avg_loss, acc, exe, feeder, test_program):
    global BATCH_SIZE, model_save_dir, EPOCH_NUM
    # 记录迭代过程中，每一步的loss和accuracy
    all_train_iter = 0
    all_trian_iters = []
    all_train_losses = []
    all_train_accs = []
    # 记录迭代过程中，每一次epoch的平均loss和accuracy
    epoch_train_losses = []
    epoch_test_losses = []
    epoch_train_accs = []
    epoch_test_accs = []

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    print('---开始训练---')
    for pass_id in range(1, EPOCH_NUM + 1):
        print('epoch %d --------' % pass_id)
        train_accs = []
        train_losses = []
        for batch_id, data in enumerate(train_reader()):  # 遍历test_reader
            # enumerate()函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标
            train_loss, train_acc = exe.run(
                program=fluid.default_main_program(),
                feed=feeder.feed(data),
                fetch_list=[avg_loss, acc]
            )

            all_train_iter = all_train_iter + BATCH_SIZE
            all_trian_iters.append(all_train_iter)
            all_train_losses.append(train_loss[0])
            all_train_losses.append(train_acc[0])
            train_losses.append(train_loss[0])
            train_accs.append(train_acc[0])
            # 每10次batch打印一次训练、进行一次测试
            if batch_id % 50 == 0:
                print("\tPass %d, Step %d, loss %f, Acc %f" % (pass_id, batch_id, train_loss[0], train_acc[0]))

        epoch_train_losses.append(sum(train_losses) / len(train_losses))
        epoch_train_accs.append(sum(train_accs) / len(train_accs))
        print('\t\tTrain:%d, Loss:%0.5f, ACC:%0.5f' % (pass_id, epoch_train_losses[-1], epoch_train_accs[-1]))

        # 开始测试
        test_accs = []
        test_losses = []
        # 每训练一轮，进行一次测试
        for batch_id, data in enumerate(test_reader()):
            test_loss, test_acc = exe.run(
                program=test_program,
                feed=feeder.feed(data),
                fetch_list=[avg_loss, acc]
            )
            test_accs.append(test_acc[0])
            test_losses.append(test_loss[0])

        epoch_test_losses.append(sum(test_losses) / len(test_losses))
        epoch_test_accs.append(sum(test_accs) / len(test_accs))
        print('\t\tTest:%d, Loss:%0.5f, ACC:%0.5f' % (pass_id, epoch_test_losses[-1], epoch_test_accs[-1]))
        print('\n')

        if pass_id % 5 == 0:
            # 每5个epoch观察一下训练过程的趋势，并保存一次
            dicLossAcc = {}
            dicLossAcc['iteration'] = all_trian_iters
            dicLossAcc['iter_loss'] = all_train_losses
            dicLossAcc['iter_acc'] = all_train_accs
            dicLossAcc['epoch'] = list(range(1, pass_id + 1))
            dicLossAcc['loss_train'] = epoch_train_losses
            dicLossAcc['loss_test'] = epoch_test_losses
            dicLossAcc['acc_train'] = epoch_train_accs
            dicLossAcc['acc_test'] = epoch_test_accs
            draw_train_process(pass_id, dicLossAcc)

            # 保存一个模型
            model_dir = os.path.join(model_save_dir, str(pass_id))
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            fluid.io.save_inference_model(model_dir, ['image'], [predict], exe)
            print('第%d个epoch的训练模型保存完成！' % pass_id)
        else:  # 保存最后一个模型
            dicLossAcc = {}
            dicLossAcc['iteration'] = all_trian_iters
            dicLossAcc['iter_loss'] = all_train_losses
            dicLossAcc['iter_acc'] = all_train_accs
            dicLossAcc['epoch'] = list(range(1, pass_id + 1))
            dicLossAcc['loss_train'] = epoch_train_losses
            dicLossAcc['loss_test'] = epoch_test_losses
            dicLossAcc['acc_train'] = epoch_train_accs
            dicLossAcc['acc_test'] = epoch_test_accs
            draw_train_process(pass_id, dicLossAcc)

            model_dir = os.path.join(model_save_dir, str(pass_id))
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            fluid.io.save_inference_model(model_dir, ['image'], [predict], exe)
            print('第%d个epoch的训练模型保存完成！' % pass_id)


'''
函数 getPredList(predPath)
功能: 获取指定路径下待预测图片的路径，并存储在列表中
返回值：待预测图片路径列表
'''


def getPredList(predPath):
    predIDs = os.listdir(predPath)
    # 排除掉所有非jpg或png的图片
    predIDs = [x for x in predIDs if (x.find('.jpg') > 0 or x.find('.png') > 0)]
    if len(predIDs) == 0:
        print('---没有图片！ 请检查文件夹%s' % predPath)
    else:
        predList = [os.path.join(predPath, x) for x in predIDs]
        return predList


'''
函数 predImgs(pathList, optimalEpoch)
功能: 给定待预测图片的路径，指定模型得出预测结果
返回值: 预测的图像分类
'''


def createInfer():
    '''
    函数功能：创建预测器
    函数返回值： 待预测图像的预测器
    '''
    global USE_CUDA
    place = fluid.CUDAPlace(0) if USE_CUDA else fluid.CPUPlace()
    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()
    return infer_exe, inference_scope


def load_image(predPath):
    '''
    函数功能：导入图片、进行尺寸裁剪， 归一化后转换为图像特征向量
    函数返回值：图像特征向量
    '''
    global IMG_W, IMG_H
    img = paddle.dataset.image.load_and_transform(predPath, IMG_H, IMG_W, False).astype('float32')
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # np.expand_dims()函数就是增加一个维度


def getClassList(path):
    '''
    函数功能： 获取图片的分类名，读取的是之前生成的readme.json文件
    函数返回值： 数据集类别数量
    '''
    with open(path, 'r') as load_f:
        new_dict = json.load(load_f)
        return new_dict['all_class_name']


def predImgs(pathImgList, optimalEpoch):
    pred_label_list = []
    pred_class_list = []
    if pathImgList == False:
        return
    modelpath = os.path.join(model_save_dir, str(optimalEpoch))
    for pathImg in pathImgList:
        infer_exe, inference_scope = createInfer()
        with fluid.scope_guard(inference_scope):
            # 从指定目录中加载推理model(inference model)
            [inference_program,  # 预测用的program
             feed_target_names,  # 是一个str列表，它包含需要在推理Program中土工数据的变量名称
             fetch_targets
             ] = fluid.io.load_inference_model(modelpath, infer_exe)
            # 画出图像
            img = Image.open(pathImg)
            plt.figure(figsize=(1, 1))
            plt.axis('off')  # 设定为关闭坐标轴
            plt.imshow(img)
            plt.show()
            import time
            time.sleep(0.5)

            # 预测图像
            img = load_image(pathImg)
            results = infer_exe.run(inference_program,
                                    feed={feed_target_names[0]: img},
                                    fetch_list=fetch_targets)
            label_list = getClassList(os.path.join(dataRootPath, 'readme.json'))
            pred_label = np.argmax(results[0])
            pred_class = label_list[np.argmax(results[0])]
            print('---%s 的预测结果为: %s') % (pathImg, label_list[np.argmax(results[0])])
            print('\n')
            pred_label_list.append(pred_label)
            pred_class_list.append(pred_class)

    return pred_label_list, pred_class_list


##############################################################################
# 准备数据集
# 指定数据所在的路劲名
source_file = '/data/data20743/catVSdong.zip'
# 指定解压文件所在路径名
target_dir = 'data/'
# 调用unzipFile(source_file, target_dir)， 解压原始文件
unzipFile(source_file, target_dir)

# 查看解压后的文件，设置图片所在的路径
dataRootPath = 'data/catVSdog/train'
# 调用genDataList(dataRootPath)函数，划分训练集和测试集
classNumber = genDataList(dataRootPath)

# 调用函数createDataReader(BATCH_SIZE)构建读取器
train_reader, test_reader = createDataReader(BATCH_SIZE)
'''
函数 CNN(images, classNumber)
功能: 搭建具有两个卷积-池化层的卷积神经网络
返回值: 正向传播的预测值
'''


def cnn(image, classNumber):
    conv1 = fluid.nets.simple_img_conv_pool(
        input=image,
        num_filters=20,
        filter_size=3,
        pool_size=2,
        pool_stride=2,
        act='relu'
    )
    conv2 = fluid.nets.simple_img_conv_pool(
        input=conv1,
        num_filters=10,
        filter_size=5,
        pool_size=2,
        pool_stride=2,
        act='relu'
    )
    y_predict = fluid.layers.fc(input=conv2, size=classNumber, act='softmax', name='outpu_layer')
    print('两个卷积-池化层的卷积神经网络创建完成！')
    return y_predict


# 用三层神经网络进行训练并保存模型
image = fluid.layers.data(name='image', shape=[3, IMG_H, IMG_W], dtype='float32')
predict = cnn(image, classNumber)

# 调用setNN(predict, image, label, LEARNING_RATE)函数配置神经网络
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
avg_loss, acc, exe, feeder, test_program = setNN(predict, image, label, LEARNING_RATE)

# 模型训练
training(image, label, predict, avg_loss, acc, exe, feeder, test_program)

# 模型预测
# 解压待预测数据集
source_file = 'topredict.zip'
target_dir = 'data/'
unzipFile(source_file, target_dir)
# 生成待预测数据集的list文件(不带标签)
predPath = 'data/topredict/'
predList = getPredList(predPath)
# 模型训练 调用predImgs(preList, optimalEpoch), 输出预测结果
pred_label_list, pred_class_list = predImgs(predList, 10)