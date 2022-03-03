# Word Vectors and Senses

# **Contents**

1. Introduction
2. Word2vec
3. Co-occurrence Matrix
4. GloVe

# 1. Introduction

## Vector Space

- Word Vector를 vector space에 위치시켰을 때, 유사한 의미를 보유하고 있으면 가까이 위치하고 있다는 것을 의미
- 벡터의 방향성에도 의미를 가지고 있음

ex) 코사인 유사도: 두 벡터 간의 코사인 각도를 이용하여 유사도를 구함

![Untitled](Word%20Vecto%20859eb/Untitled.png)

## Optimization

![Untitled](Word%20Vecto%20859eb/Untitled%201.png)

- 손실함수 $J(θ)$ 를 최소화해야함

$J(θ)$ 의 기울기(gradient)를 계산하고 반대편으로 이동을 하면서 minimum 최소점으로 이동하게 됨

![Untitled](Word%20Vecto%20859eb/Untitled%202.png)

- $alpha$ = learning rate

한번에 얼마큼 이동할지 결정하게 됨

- $▽θJ(θ)$ 는 손실함수의 기울기

- 하지만, parameter θ 를 계산하는 것은 비용이 매우 많이 듦 → Stochastic Gradient Descent

## Stochastic Gradient Descent

- 전체 중 일부 sample만 사용
    - 해당 샘플에 대해서만 gradient를 계싼해서 paramenter를 업데이트
    - 점진적으로 최소점으로 이동
    - 상대적으로 noise가 클 수 밖에 없음
- 여러개의 sample을 모아서 mini-batch라고 함
    - 보통 32개 or 64개의 sample
    - 여러개의 sample을 사용해서 noise를 줄이고 병렬화 가능 → 빠른 학습

# 2. Word2vec

- 텍스트 모델의 핵심은 텍스트를 컴퓨터가 인식할 수 있는 벡터로 변환하는 것
    - One Hot Encoding (Sparse representation)
    - Word Embedding (Dense representation)
    
- One Hot Encoding → 벡터가 0 또는 1로 구성되어 있어서 간단, but 단어의 유사성 파악은 반영하지 못함
- 이를 보완하기 위한 방법이 Word Embedding
    - Word2vec, GloVe, FastText

![Untitled](Word%20Vecto%20859eb/Untitled%203.png)

- 의미적으로 유사성을 가진 단어들은 서로 가까운 위치에 존재한다는 아이디어를 이용
- 중심 단어를 기준으로 양쪽으로 윈도우 크기만큼의 단어를 맥락 단어로 설정
- 원-핫 인코딩을 이용해 중심 단어 벡터와 맥락 단어 벡터를 생성한 후, 이를 입력 벡터와 출력 벡터로 사용
- **맥락 벡터가 입력, 중심 벡터가 출력**인 경우를 **CBOW** 방식
- **중심 벡터가 입력, 맥락 벡터가 출력**인 경우를 **Skip-gram** 방식

1) Continuous Bag of Words (CBOW)

: 모든 외부 단어(맥락벡터)로 가운데 단어(중심단어)를 예측하는 방식

![Untitled](Word%20Vecto%20859eb/Untitled%204.png)

- 위 그림을 보면, 입력 벡터는 맥락 벡터이기 때문에 입력 벡터의 수도 윈도우 크기의 2배
1. 가중치 U와 맥락벡터를 곱해서 여러 개의 결과 벡터가 생성됨
2. Hidden layer에서는 이 결과들을 요소별로 평균내어 hidden layer의 값으로 사용
3. 중심 벡터 사이에 있는 가중치 *V*와 hidden layer의 벡터를 곱하여 만든 output layer의 벡터에 softmax함수를 적용하여 이 값을 확률 값으로 만듦
4. 중심 벡터와 비교하여 모델의 Loss를 계산
5. Loss를 최소화하는 모델을 구하기 위해 Gradient Descent를 이용하여 가중치를 계속적으로 업데이트

2) Skip-grams (SG)

: 가운데 1개의 단어로 외부의 모든 단어를 예측하는 방식

![Untitled](Word%20Vecto%20859eb/Untitled%205.png)

- 출력 벡터는 맥락 벡터이기 때문에 출력 벡터의 수도 윈도우 크기의 2배

3) Word2vec의 효율성 높이는 법

- Stochastic Gradient Descent

: Loss를 최소화하는 parameter를 찾기 위해 Gradient Descent를 사용

- 하지만 계산량이 너무 많다는 단점
- 그래서 Stochastic Gradient Descent 사용 (mini-batch)

![Untitled](Word%20Vecto%20859eb/Untitled%206.png)

- 0에 해당되는 위치에서는 계산이 이루어지더라도 계속해서 0이기 때문에 실제로 gradient가 update되지 않는데, 불필요하게 계산이 이루어짐

→ solution negative sampling (skip-gram)

- Negative Sampling

![Untitled](Word%20Vecto%20859eb/Untitled%207.png)

- Skip-gram 예시 : Negative Sampling
    - 실제로 등장한 (=값이 0이 아닌) 행에 대해서만 gradient 계산하고 sparse한 matrix를 add, subtract 함으로써 gradient를 update하는 방식을 채택
- 이는 기존의 다중분류를 이진분류로 근사시켜 모델을 효율적으로 만드는 데에 기여

![Untitled](Word%20Vecto%20859eb/Untitled%208.png)

다음은 negative sampling의 목적함수 입니다.

*c* : 중심 벡터

*o* : 맥락 벡터

*k* : 노이즈 벡터 (랜덤하게 선택된 벡터. 실제 맥락벡터가 아님)

*u* : 중심 벡터와 hidden layer 사이의 가중치

*v* : 맥락 벡터와 hidden layer 사이의 가중치

- $u^Tv$는 중심 벡터와 맥락 벡터간 코사인 유사도를 의미
- True pair (중심 벡터, 맥락 벡터)의 경우 코사인 유사도가 클수록 확률이 높고, Noise pair (중심 벡터, 맥락 벡터)의 경우 코사인 유사도가 작을수록 확률이 높다고 해석할 수 있음

# 3. Co-occurrence Matrix

- Skip-gram은 중심 단어를 기준으로 맥락 단어가 등장할 확률을 계산
    - 그러므로 윈도우 개수를 아무리 크게 늘려도 global co-occurrence statistics(전체 단어의 동반출현 빈도수)와 같은 통계 정보는 내포할 수 없음
    - Why? 벡터의 값들은 중심 단어가 given일 때 각 값의 개별적인 등장 확률을 의미하기 때문
    
    → 그래서 등장한 것이 count-based의 Co-occurrence matrix
    

**1) Window based co-occurrence matrix (단어-문맥 행렬)**

- 한 **문장을 기준**으로 윈도우에 각 단어가 몇 번 등장하는 지를 세어 구성

![Untitled](Word%20Vecto%20859eb/Untitled%209.png)

**2) Word-Document matrix (단어-문서 행렬)**

- 한 **문서를 기준**으로 각 단어가 몇 번 등장하는 지를 세어 구성
- 문서에 있는 많은 단어들 중 빈번하게 등장하는 특정 단어가 존재한다는 것을 전제
- LSA (Latent Semantic Analysis; 잠재적 의미 분석)를 가능하게 하는 기법 (ex. 문서 간 유사도 측정)

![Untitled](Word%20Vecto%20859eb/Untitled%2010.png)

- 이와 같은 count-based matrix는 단어의 개수가 증가할수록 차원이 폭발적으로 증가함

→ **SVD** 또는 **LSA** 등을 이용하여 차원을 축소시킨 후 사용!

- 특정 크기의 window 내에 나타나는 단어들을 모두 count
    - A라는 단어가 B에 인접하게 몇 번 사용되었는지 확인해서 matrix를 구성한다.
- 유사한 쓰임/의미를 보유하고 있는 단어들끼리는 비슷한 벡터 구성을 보유하게 된다.
    - 왜냐하면 비슷한 단어들은 비슷한 환경/문맥에서 사용되기 때문에, 비슷한 단어들과 인접하게 된다.
- 유사한 형태의 단어들이 인접하게 위치
    - ex) take, taken, taking, took 등

- 문제점
    1. sparse matrix를 형성하게 된다.
    2. 단어 수가 많아지는 경우 matrix의 크기가 커진다.

# 3. GloVe (Global Vectors for Word Representation)

![Untitled](Word%20Vecto%20859eb/Untitled%2011.png)

- word2vec은 window 내의 정보만 사용한다는 점에서 전체적인 문장에 대한 이해가 떨어진다는 단점을 가지고 있다.
- Direct Prediction과 count based 방식을 합쳐서 적용
    - 임베딩된 단어벡터 간의 유사도 측정을 가능하게 하면서 corpus 전체의 통계정보도 함께 사용할 수 있도록 해보자

Glove의 기본 아이디어는 다음과 같음:

- 임베딩된 단어벡터 간 **유사도** 측정을 수월하게 하면서 (word2vec의 장점)
- 말뭉치 전체의 **통계 정보**를 반영하자! (co-occurrence matrix의 장점!)

- 임베딩된 두 단어벡터의 내적이 corpus 전체에서의 동시에 등장하는 확률의 로그값이 되도록 목적함수를 정의

![Untitled](Word%20Vecto%20859eb/Untitled%2012.png)

- objective function(목적 함수)는 다음과 같이 정의한다.

![https://blog.kakaocdn.net/dn/b4wy8H/btqGzXQGsg1/UQRhg63gaAzx2s6kABprhk/img.png](https://blog.kakaocdn.net/dn/b4wy8H/btqGzXQGsg1/UQRhg63gaAzx2s6kABprhk/img.png)

objective function for GloVe

- 장점
    - 작은 규모의 corpus에 대해서도 효과적으로 학습이 가능하다.
    - 자주 등장하지 않는 단어에 대해서도 효과적인 word vector를 생성할 수 있다.
    - 학습이 상대적으로 빠르다.

**Result**

![Untitled](Word%20Vecto%20859eb/Untitled%2013.png)

이처럼 frog와 형태적으로 또는 의미적으로 비슷한 단어를 잘 선택함

### + FastText, ELMo

추가적으로 Word2vec, GloVe 외의 단어 임베딩 모델을 소개해드리도록 하겠습니다.

1) FastText (2016)

- word2vec과 유사합니다. But 그러나 단어를 부분 단어(subword)로 표현한다는 차이가 존재합니다.
- word2vec은 단어를 쪼갤 수 없는 것으로 생각하지만, FastText는 하나의 단어도 부분 단어로 쪼갤 수 있다고 생각합니다.
- 학습하지 않은 단어에 대해서 유사한 단어를 찾아낼 수 있습니다.

2) ELMo (2018)

- 단어의 의미는 전에 오는지, 후에 오는지에 따라서도 변화한다는 특성을 반영한 모델입니다.
- 전체 문장을 한 번 훑고, 문장 전체를 고려한 word embedding을 진행같은 단어라도 문맥에 따라 뜻이 달라지는 동적인 모델입니다.
- 다의어로 인한 어려움을 해결할 수 있습니다.
