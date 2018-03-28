 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 22:18:40 2018

@author: Rorschach
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image, ImageDraw, ImageFont
#绘图
from skimage import transform as tf
#错切变化

#产生验证码
def create_captcha(text, shear=0, size=(100,30)):
    im = Image.new('L', size, 'black')
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(r'Coval-Black.otf', 22)
    draw.text((2, 2), text, fill=1, font=font)
    image = np.array(im)    #转化为 0，1 图像数组
    affine_tf = tf.AffineTransform(shear=shear)
    image = tf.warp(image, affine_tf)
    return image / image.max()  #归一化处理

from matplotlib import pyplot as plt
image = create_captcha('BENE', shear=0.5)
plt.imshow(image, cmap='gray')

#分割单词
from skimage.measure import label, regionprops

def segment_image(image):
    labeled_image = label(image > 0)    #image 是单色,取bool  1为白色，0为黑色
    subimages = []
    for region in regionprops(labeled_image):   #提取连续区域
        start_x, start_y, end_x, end_y = region.bbox
        subimages.append(image[start_x:end_x, start_y:end_y])
    if len(subimages) == 0:
        return [image,]
    return subimages
# =============================================================================
# #画出分割后的图
# image = create_captcha('Z', shear=0, size=(25, 25))
# subimages = segment_image(image)
# subimages
# =============================================================================
##随机产生字母
from sklearn.utils import check_random_state

random_state = check_random_state(14)
letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
shear_values = np.arange(0, 0.5, 0.05)  #[0,0.5) by 0.05

def generate_sample(random_state=None):
    random_state = check_random_state(random_state)
    letter = random_state.choice(letters)
    shear = random_state.choice(shear_values)
    return create_captcha(letter, shear=shear, size=(25, 25)), letters.index(letter)
             #返回字母图像 字母在letters中的位置

image, target = generate_sample(random_state)
plt.imshow(image, cmap='gray')
print('The target for this image is: {}'.format(target))

#创建数据集
dataset, targets = zip(*(generate_sample(random_state) for i in range(3000))) #每运行一次就产生一个随机，与i无关
dataset = np.array(dataset, dtype='float')
targets = np.array(targets)

##编码
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
y = onehot.fit_transform(targets.reshape(targets.shape[0], 1))
y = y.todense() #转化为密集矩阵，其实还是挺稀疏的

#重新调整大小,均为(25, 25)， train 中图像留黑太多, 用之前定义的函数切割图像
from skimage.transform import resize

datasets = np.array([resize(segment_image(sample)[0], (25, 25)) for
sample in dataset])

#segment_image(sample) 是一个list

#扁平化
X = datasets.reshape((datasets.shape[0], datasets.shape[1] * datasets.shape[2])) #3000个字母，每个字母 25*25 像素

#切分数据集
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)



#训练

##创建模型所需的特殊数据格式
from pybrain.datasets import SupervisedDataSet

training = SupervisedDataSet(X.shape[1], y.shape[1]) #625, 1
for i in range(X_train.shape[0]):
    training.addSample(X_train[i], y_train[i])
    
testing = SupervisedDataSet(X.shape[1], y.shape[1]) #625, 1
for i in range(X_test.shape[0]):
    testing.addSample(X_test[i], y_test[i])


#构建神经网络, 三层，625,100(暂定),26
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

net = buildNetwork(X.shape[1], 100, y.shape[1], bias=True)  #bias 偏置神经元

#训练，反向传播算法

trainer = BackpropTrainer(net, training, learningrate=0.01, weightdecay=0.01, 
                          verbose = True)
#, learningrate=0.01, weightdecay=0.01)
                                         #神经元各边权重的误差函数的偏导数和学习速率

##设置训练步数                                         
trainer.trainEpochs(epochs=20)

#预测结果  字母识别率

predictions = trainer.testOnClassData(dataset=testing)

real = []
data = y_test.argmax(axis=1)
for t in data:
    real = real + [int(t)]

n_t = 0
n_f = 0
for i in range(len(real)):
    if predictions[i] == real[i]:
        n_t += 1
    else:
        n_f += 1
accuracy = n_t / (n_t + n_f)
#print('Accuracy:', accuracy)    

from sklearn.metrics import f1_score
print('F-score: {0:.3f}'.format(f1_score(predictions, real, average='micro')))

from sklearn.metrics import classification_report
print(classification_report(real, predictions))

#test
def predict_captcha(image, net):
    predicted_word = ''
    subimages = segment_image(image)
    for subimage in subimages:
        subimage = resize(subimage, (25, 25))
        outputs = net.activate(subimage.flatten())  #激活神经网络, subimage展平 
        #print(outputs)
        prediction = np.argmax(outputs)  #找最大值的位置
        #print(prediction)
        predicted_word += letters[prediction]
    return predicted_word

##try
image = create_captcha('ZQAY', shear=0.3)
predict_captcha(image, net)


##count
def test_prediction(word, net, shear=0.2):
    captcha = create_captcha(word, shear=shear)
    prediction = predict_captcha(captcha, net)
    return word == prediction, word, prediction

#获得大量四字词
from nltk.corpus import words
valid_words = [word.upper() for word in words.words() if len(word) == 4]

#验证
num_correct = 0
num_incorrect = 0
for word in valid_words:
    correct, word, prediction = test_prediction(word, net, shear=0.2)
    if correct:
        num_correct += 1
    else:
        num_incorrect += 1
print('Number correct is {}'.format(num_correct))
print('Number incorrect is {}'.format(num_incorrect))
print('Accuracy:', num_correct / (num_incorrect + num_correct))

##字母识别率达到97%，单词识别率低 << 0.97**4 = 0.88

##混淆矩阵  (4, 2) = 5 代表 D 被识别为 B 5 次
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), predictions)

plt.figure(figsize=(20, 20))
plt.imshow(cm, cmap='gray')

tick_marks = np.arange(len(letters))
plt.xticks(tick_marks, letters)
plt.yticks(tick_marks, letters)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.imshow()


#调优

#1 查字典，预测值若不在字典，取与预测值相似的字典值返回
#通过列文斯坦编辑距离 衡量是否相似
from nltk.metrics import edit_distance
steps = edit_distance('STEP', 'STOP')
print('The number of steps needed is: {0}'.format(steps))

#低配版距离 每个位置字母是否一致
def compute_distance(prediction, word):
    return len(prediction) - sum(prediction[i] == word[i] for i in range(len(prediction)))

from operator import itemgetter
def improved_prediction(word, net, dictionary, shear=0.2):
    captcha = create_captcha(word, shear=shear)
    prediction = predict_captcha(captcha, net)
    if prediction not in dictionary:      #http://www.runoob.com/python/python-operators.html
        distances = sorted([(word, compute_distance(prediction, word)) for word in dictionary], key=itemgetter(1))
        best_word = distances[0]
        prediction = best_word[0]
    return word == prediction, word, prediction

#采用列文斯坦
def improved_prediction_pro(word, net, dictionary, shear=0.2):
    captcha = create_captcha(word, shear=shear)
    prediction = predict_captcha(captcha, net)
    if prediction not in dictionary:      
        distances = sorted([(word, edit_distance(prediction, word)) for word in dictionary], key=itemgetter(1))
        best_word = distances[0]
        prediction = best_word[0]
    return word == prediction, word, prediction

#结果
num_correct = 0
num_incorrect = 0
for word in valid_words:
    correct, word, prediction = improved_prediction(word, net, valid_words, shear=0.2)
    if correct:
        num_correct += 1
    else:
        num_incorrect += 1
        
num_correct_pro = 0
num_incorrect_pro = 0
for word in valid_words:
    correct, word, prediction = improved_prediction_pro(word, net, valid_words, shear=0.2)
    if correct:
        num_correct_pro += 1
    else:
        num_incorrect_pro += 1

print('Number correct is {}'.format(num_correct))
print('Number incorrect is {}'.format(num_incorrect))
print('Accuracy pro:', num_correct / (num_incorrect + num_correct))

print('Number correct is {}'.format(num_correct_pro))
print('Number incorrect is {}'.format(num_incorrect_pro))
print('Accuracy pro pro:', num_correct_pro / (num_incorrect_pro + num_correct_pro))








