# 人脸表情检测

### 环境依赖

- Python 3

- OpenCV(3.4.0)
- dlib(19.10)
- scikit-learn(0.19.1)
- scikit-image(0.13.1)

### 使用方法

```shell
python test.py -i 0
```

### 人脸库

项目使用两个库：人脸表情库、人脸库

- [The Cohn-Kanade AU-Coded Expression Database (CK+)](http://www.pitt.edu/~emotion/ck-spread.htm)

  训练好的人脸表情库，在`/models/emotions_model.dat`(约24.1MB)

- [The 10k US Adult Faces Database](http://wilmabainbridge.com/facememorability2.html)

  训练好的人脸库，在`/models/face_model.dat`中(约99.7MB)

### 步骤

1. 使用`gabor`滤波进行边缘检测和纹理特征提取。见`gabor.py`
2. 使用`dlib`进行脸部landmarks标记。见`faces.py`
3. 使用`sklearn`创建SVM classifier，`clf.predict_proba([features])[0]` 预测7个表情的概率

> 其中有7个表情：`'neutral', 'happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust' `

4. 最后在`test.py`调用camera进行实时表情检测，并显示出来。

### Result

#### 冷漠脸

<img src="https://ws2.sinaimg.cn/large/006tKfTcgy1fs1g64ag2oj31kw0wzqv8.jpg" width="650px">

#### 笑脸

<img src="https://ws3.sinaimg.cn/large/006tKfTcgy1fs1g65wazkj31kw0wzhdw.jpg" width="650px">

#### 惊讶脸

<img src="https://ws4.sinaimg.cn/large/006tNc79gy1fs1ghh1peqj31kw0wzu10.jpg" width="650px">

#### 厌恶脸

<img src="https://ws3.sinaimg.cn/large/006tKfTcgy1fs1g61xl5mj31kw0wzu10.jpg" width="650px">



### 发现的问题

在测试时发现，不管做什么表情，有很大的概率都会被归于“不笑”和“笑”，由此可以猜想：

在训练数据集中，训练样本可能大多数只有“笑”和“不笑”两类，且数量远远多于其他表情，这造成了SVM分类器的<u>**不平衡问题**</u>。

根据scikit-learn官网

> [`SVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) (but not [`NuSVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC)) implement a keyword `class_weight` in the `fit` method. It’s a dictionary of the form `{class_label : value}`, where value is a floating point number > 0 that sets the parameter `C` of class `class_label` to `C * value`.

可修改SVC参数`class_weight` 调整各个类别的权重。故修改代码如下：

```python
self._emotions = OrderedDict([
                             (0, 'neutral'), (1, 'happiness'), (2, 'sadness'),
                             (3, 'anger'), (4, 'fear'),  (5, 'surprise'),
                             (6, 'disgust')
                         ])
class_weight = {0: 1.,
                1: 1.,
                2: 50.,
                3: 50.,
                4: 50.,
                5: 50.,
                6: 50.}
        self._clf = svm.SVC(kernel='rbf', gamma=0.001, C=10,
                            decision_function_shape='ovr',
                            probability=True, class_weight=class_weight)
```

调整`class_weight`之后，`surprise`和`disgust`表情的识别率明显提高，但是`sadness`,`anger`,`fear`的识别率仍然不理想，说明修改此参数在**<u>一定程度上</u>**能减少不平衡训练样本的影响。

所以，要想得到更好的模型，必须从原始数据上下手，增加更多的`sadness`,`anger`,`fear`标签的训练数据。

### 参考

https://github.com/luigivieira/emotions













