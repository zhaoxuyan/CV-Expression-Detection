# Detection of Emotion

1. **利用sklearn自带的olivetti_faces库，手动为数据打上“笑”与“不笑”的标签，生成result.xml训练集文件。使用SVM读取xml，进行训练/预测。**

   <img src="https://ws3.sinaimg.cn/large/006tNc79gy1fs1sszhihzj30iq0f60u8.jpg" width="300px">

2. **利用dlib+gabor+SVM，使用**

- [The Cohn-Kanade AU-Coded Expression Database (CK+)](http://www.pitt.edu/~emotion/ck-spread.htm)

  训练好的人脸表情库，在`/models/emotions_model.dat`(约24.1MB)

- [The 10k US Adult Faces Database](http://wilmabainbridge.com/facememorability2.html)

  训练好的人脸库，在`/models/face_model.dat`中(约99.7MB)

- 使用`gabor`滤波进行边缘检测和纹理特征提取。见`gabor.py`

- 使用`dlib`进行脸部landmarks标记。见`faces.py`

- 使用`sklearn`创建SVM classifier，`clf.predict_proba([features])[0]` 预测7个表情的概率

> 其中有7个表情：`'neutral', 'happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust' `

- 最后在`test.py`调用camera进行实时表情检测，并显示出来。

<img src="https://ws2.sinaimg.cn/large/006tKfTcgy1fs1g64ag2oj31kw0wzqv8.jpg" width="450px">