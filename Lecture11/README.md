# Lecture 11

### Convolutional Networks for NLP

---

### Overview : NLP에서의 Convolutional Network 사용

|Index|Subtitle|
|--- | --- |
|11.1.| Intro to CNNs |
|11.2.| Simple CNN for sentence classifcaiton |
|11.3.| CNN for NLP |
|11.4.| Deep CNN for Sentence Classifcation |
|11.5.| Quasi-recurrent Neural Networks |

---

### 11.1. Intro to CNNs

- Convolution이란?
    - 주로 이미지에서 feature 추출을 위해서 고전적으로 활용하던 방법론이다.
    - 1D, 2D, ... 여러 dimension에서의 convolution이 가능하다.
    - 텍스트에서는 필터가 위아래로 이동할 수 밖에 없으므로 1D convolution을 수행한다.

- 텍스트 데이터의 CNN 도입 배경
    - RNN은 prefix context(the, of..)처럼 의미가 없는 단어들을 포함하고 있으며 마지막 단어 벡터의 영향을 많이 받는다.
    - 1D convolution을 통해 특정 k-gram의 정보를 capture하는 방법론을 도입한다.

- CNN에서 중요한 개념들 : **Convolution, Padding, Strdie, Dilation, Pooling** <br/>
    <img src = "https://user-images.githubusercontent.com/75057952/152970240-277c6315-61d0-46ff-86d3-be8781804899.jpg" width = "400dp"><br/><br/>
    <img src = "https://user-images.githubusercontent.com/75057952/152970303-bae61c9e-a8dd-4666-95a4-0f78f97597f5.jpg" width = "1000dp"><br/><br/>
    <img src = "https://user-images.githubusercontent.com/75057952/152970305-12cb4eab-9e34-4287-b443-da5a93128114.jpg" width = "300dp"><br/><br/>
    <img src = "https://user-images.githubusercontent.com/75057952/152970309-4c9e5329-a72b-4a8e-bfda-2ded0f40b49d.jpg" width = "500dp">

---

### 11.2. Simple CNN for sentence classifcaiton

- **Yoon Kim (2014):** [*Convolutional Neural Networks for Sentence Classfication. EMNLP 2014.*](https://arxiv.org/pdf/1408.5582.pdf)<br/>
     <img src = "https://user-images.githubusercontent.com/75057952/152970312-d6fec045-e778-4392-b32e-3bfe9209f375.jpg" width = "900dp"><br/>
     - Word2Vec, Glove 등으로 pre-train된 워드 임베딩 사용
     - Pretrained word vector로 초기화한 벡터를 두 개 복제하여 fine tuning한 것(static), 하지 않은 것(non-static) 둘 다 사용
     - ReLU activation function
     - 마지막 Softmax layer에서 Dropout(p=0.5), L2 norm constraint(||W||=3) 등의 regularization 기법 사용 <br/><br/>
        <img src = "https://user-images.githubusercontent.com/75057952/152970316-df438d2c-7e17-4d16-8f6f-3e8bc8dac0cc.jpg" width = "700dp"><br/>
        <img src = "https://user-images.githubusercontent.com/75057952/152970322-1b28966b-de89-4b93-bffb-559c5fe869d2.jpg" width = "300dp">

- Kim's work & 다른 classification model 비교
    - Regularization & Dropout 등의 방법론적 차이 존재로 성능 비교 문제점 지적
    - 간단한 CNN architecture로 유사하거나 조금 못 미치는 성능을 보였다는 것이 주된 contribution point라고 생각함

---

> **Remark**

- Gated Units used vertically
    - [ResNet](https://arxiv.org/abs/1512.03385) (He et al., ECCV 2016)
    - [HighwayNet](https://arxiv.org/abs/1505.00387) (Srivistava et al., NIPS 2015)
    - ResNet은 Residual block, Identity mapping을 적용함
    - HighwayNet은 Dynamic Gate로 구성된 block을 활용함
    - Residual Block : F(x) + x , Highway block :  F(x)T(X) + x.C(x)

- [BatchNorm](https://arxiv.org/abs/1502.03167) (Batch Normalization)
    - Convolution 연산의 output을 배치 단위로 평균이 0, 분산이 1이 되도록 정규화
    - 모델의 Parameter innitialization 값에 따라서 성능이 많이 달라지지 않도록 안정화하는 역할

- [1X1 Convolution](https://arxiv.org/abs/1312.4400)
    - 1X1 크기를 가지는 Convolution Filter를 사용한 Convolution Layer
    - 차원 축소, 파라미터 수를 줄임
    - Low dimensionial embedding의 효율 향상
    - Convolution 이후 ReLU 등의 비선형 함수를 추가해 모델 비선형성을 더함
    - 예시 : ResNet의 bottleneck architecture
    <br/>
    <img src = "https://user-images.githubusercontent.com/75057952/152970324-2648cd46-2f12-426e-b44f-f7c5005ebd6e.png" width = "500dp">
---

### 11.3. CNN for NLP

- NLP task에서 CNN 아키텍처의 의미
    - CNN은 기본적으로 low-level feature부터 high-level feature까지 학습해 나간다.
    - 문장을 구성하는 각각의 단어 토큰 - k-gram(multi word) - expression - phrase - sentence의 순서로 feature를 capture한다고 이해할 수 있다.
    - hierarchical feature capturing
    - CNN은 구현이 쉽고 간단한 아키텍처라는 장점이 있으나, CNN으로만 레이어를 구성하게 되면 많은 convolution layer가 필요하게 됨

- 지금까지 배운 모델
    - Bag of Vectors
    - Window Model
    - CNNs : easy to parallelize
    - RNNs with attention


---

### 11.4. Deep CNN for Sentence Classifcation

- VDCNN(Very Deep CNN) for Text
    - **Alexis Conneau (2017)** : *Very Deep Convolutional Networks for Text Classification*
    - 일반적으로 NLP에서 CNN을 사용하더라도 신경망이 매우 깊지 않음
    - 하지만 VGGNet, ResNet처럼 깊게 층을 쌓은 것이 VD-CNN이다.
    - BatchNorm & ReLU activation <br/>
    <img src = "https://user-images.githubusercontent.com/75057952/152970327-0affe897-d7ee-45b2-b50b-529d14a54067.JPG)" width = "200dp"> <br/>
    <img src = "https://user-images.githubusercontent.com/75057952/152970331-89418c7f-4973-4e91-ba24-f3eefd93cc77.JPG" width = "200dp"> <br/>
    - 대체로 29 Layer 이상 깊게 쌓았을 때 좋은 성능을 보임 <br/>
    <img src = "https://user-images.githubusercontent.com/75057952/152970334-63a89229-8086-49ac-b14c-159aa87637b2.JPG" width = "500dp"> <br/>

---

### 11.5. Quasi-recurrent Neural Networks 

- RNN은 Deep NLP를 위한 standard한 building blcok이지만, parallelize하기 어렵고 굉장히 느리다.
- 그래서 CNN과 RNN의 장점을 조합한 Quasi-recurrent Neural Network 제안
- [Q-RNN](https://arxiv.org/abs/1611.01576) <br/>
    <img src = "https://user-images.githubusercontent.com/75057952/152970339-8ad19bb3-d8aa-4449-b19a-71c78e6d029b.JPG" width = "500dp"> <br/>
    - QRNN의 경우 Convolution을 수행한 다음 CNN에서 Max-Pooling을 주로 수행하는 것고 달리 hidden state를 만들어 previous hidden state가 current hidden state에 영향을 미치게 됨
    - QRNN의 pooling layer는 개념적으로는 CNN의 max-pooling이나 average-pooling와 유사하게 하나의 값으로 합친다는 의미이지만, 그 과정은 fo-pool layer에서의 이전 state를 참조하는 RNN-fashion으로 이루어진다.
     
