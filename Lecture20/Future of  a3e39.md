# Future of NLP + Deep Learning

# Why has deep learning been so successful recently?

![Untitled](Future%20of%20%20a3e39/Untitled.png)

- Neural networks의 발전
- Scale의 변화 → data size, model size

# 딥러닝이 아직 충분히 해결하지 못한 언어 과제

비지도 기계 번역(unsupervised machine translation)

- NMT(neural machine translation)의 첫 번째 모델인 seq2seq는 지도 학습 방식의 기계 번역
- 단어 의미와 단어순서, 문장 구조, 단어 간의 의존 관계 등의 모든 정보를 토대로 문맥을 이해하는 자연스러운 번역 결과를 냄

![Untitled](Future%20of%20%20a3e39/Untitled%201.png)

- 단어정보 시퀸스를 인코더에 입력
- 인코더는 이를 분석해 고정 길이의 백터 표현 (vector representation)을 추정
- 디코더는 이 벡터를 활용해 또 다른 단어정보의 시퀀스를 생성

BUT

- 양질의 large parallel corpus가 필요한 지도 학습에서는 데이터 양이 충분하지 않으면 학습효과 x
- 데이터가 부족한 상황에서는 어떻게 신경망 훈련?
    - monolingual corpora
    - 하나의 언어로 되어있는 말뭉치만으로도 학습할 수 있는 기계 번역 모델 제시
    - 최근에는 비지도 학습 교차언어 임베딩 (unsupervised cross-lingual embedding)이 가시적인 성과를 내고 있음

1) 언어 A와 언B에 대한 단일 언어 말뭉치(monoligual corpora)를 이용해 각각의 단어 임베딩을 생성

2) 가장 쉬운 단어들만 모아 대역(bilingual) 샘플 목록을 구축

3) 대역 말뭉치를 공유 임베딩 공간에 mapping하는 변환 네트위크로 학습시킴 → 그러면 나머지 다른 단어들도 자동으로 비슷한 위치에 매핑됨

![Untitled](Future%20of%20%20a3e39/Untitled%202.png)

영어와 독일어 말뭉치를 공유 공간에 매핑한 모습

근본적인 한계:

- 위드 임베딩 그 자체는 non-deterministic
- 어떤 알고리즘을 활용하는지에 따라 임베딩 결과는 천차만별로 달라질 수 있음
- 같은 의미를 나타내는 단어라도 각 언어라는 맥락에서 봤을 때 다른 word distribution을 가질 확률이 있음

![Untitled](Future%20of%20%20a3e39/Untitled%203.png)

같은 의미를 뜻하는 단어이지만, 언어에 따라 다른 분포를 보임

이러한 단점을 보완하고 training Large model을 할 수 있는 모델? 

→ GPT-2

# GPT-2

![Untitled](Future%20of%20%20a3e39/Untitled%204.png)

GPT-2는 WebText라 불리는 40GB 크기의 거대한 코퍼스에다가 인터넷에서 크롤링한 데이터를 합쳐서 훈련시킨 언어 모델

![Untitled](Future%20of%20%20a3e39/Untitled%205.png)

- BERT: transformer의 인코더 스택만 사용한 모델
- GPT-2: 디코더 스택만 사용한 모델

![Untitled](Future%20of%20%20a3e39/Untitled%206.png)

- Encoder는 단순한 self-attention 레이어를 사용
- Decoder는 **Masked Self Attention**을 사용

![Untitled](Future%20of%20%20a3e39/Untitled%207.png)

- gpt-2는 셀프 어텐션을 계산할 때 해당 스텝의 오른쪽에 있는 단어들은 고려하지 않음

### **Auto-Regressive**

- GPT-2는 자기 회귀 모델(auto-regressive model)

장단점:

- 평범한 Self-Attention을 사용하며 자기 회귀 능력을 포기한 BERT는 다음 단어의 예측 능력은 덜 하지만, 맥락 정보를 충분히 고려할 수 있습니다.
- 반면, Masked Self-Attention을 사용하는 자기회귀 모델인 GPT-2는 다음 단어의 예측 능력은 뛰어나지만, 해당 단어의 이후에 있는 맥락 정보들을 이용할 수 없습니다.

## **GPT-2 Architecture**

****Byte Pair Encoding (BPE)****

- gpt-2는 Byte Pair Encoding를 거친 토큰을 입력 단위로 사용
- BPE는 서브워드를 분리하는 알고리즘으로, 빈도수에 따라 문자를 병합하여 서브워드를 구성

![Untitled](Future%20of%20%20a3e39/Untitled%208.png)

- 단어를 문자(char) 단위로 쪼갠 뒤, 가장 빈도수가 높은 쌍을 하나로 통합하는 과정을 반복하여 토큰 딕셔너리를 만듦

![Untitled](Future%20of%20%20a3e39/Untitled%209.png)

- 앞으로 단어, 토큰이라고 불리는 것은 모두 BPE token을 의미

### **Input encoding**

- gpt-2도 입력으로 임베딩 행렬을 받음
- 그냥 이 행렬을 입력으로 넣는 것이 아니라, Positional Encoding을 추가적으로 거침

Positional encoding?

![Untitled](Future%20of%20%20a3e39/Untitled%2010.png)

- 단어에 순서 정보를 추가하는 것을 의미
- 이를 통해서 RNN의 최대 장점이었던 **순서 정보의 고려**가 트랜스포머에서도 가능해짐

### ****Decoder Stacks****

![Untitled](Future%20of%20%20a3e39/Untitled%2011.png)

- 입력 벡터는 각 디코더셀의 self-attention 과정을 거친 뒤, 신경망 레이어를 지남
- 이 과정은 스택에 있는 디코더셀, transformer block의 개수만큼 진행됨

### ****Model Output****

![Untitled](Future%20of%20%20a3e39/Untitled%2012.png)

- 모든 디코더 블럭을 거친 최종 결과물은 입력값에 대한 최종 셀프 어텐션값을 가지고 있음 (파란색)
- 이를 우리가 가진 임베딩 벡터와 곱해주면 (초록색), 각 단어가 다음 단어로 등장할 확률값이 나옴 (보라색)
- 이 중에서 가장 확률값이 높은 것이 출력값이 되며, 또 다음의 입력값이 됨

- GPT-2는 GPT-1 모델을 기반으로 하여 Unsupervised pre-training 작업을 극대화 시킨 pretrained language model
- 모델과 데이터셋의 크기가 성능에 큰 요인이 되긴 하였지만, FIne-tuning을 제외하고도 매우좋은 성능을 보였다는 점에서 중요한 Contribution이 있다고 생각됨