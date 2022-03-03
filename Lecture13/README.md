# Contextual Word Embeddings

## Contents

1. Reflections on word representations
2. Pre-ELMo and ELMo
3. ULMfit and onward
4. Transformer archietectures
5. BERT

# 1. Reflections on word representations

![Untitled](Contextual%20daf9c/Untitled.png)

- 2012년 이전
- 가장 성능이 높다고 보이는 State-of-the-art는 딥러닝 모델이 아닌 rule-based 혹은 일반적인 머신러닝 방법론을 의미

![Untitled](Contextual%20daf9c/Untitled%201.png)

- 세번째 모델은 word2vec과 같은 unsupervied NN를 supervised NN 방법론과 결합한 방법으로 피처 엔지니어링과 같은 전처리 작업을 조금 거치게 되면 보다 성능이 상승함을 볼 수 있음

![Untitled](Contextual%20daf9c/Untitled%202.png)

- 2014
- pre-trained된 word vector를 사용하는 방법이 일반적인 random initialize된 word vector보다 성능이 좋음을 보여
- pretained된 word vector: 가지고 있는 데이터보다 방대한 양의 데이터로 학습된 결과를 활용하기 때문에 성능 향상에 도움이 됨

![Untitled](Contextual%20daf9c/Untitled%203.png)

- pre-trained된 word vector를 사용하는 것은 또한 **unk** 토큰을 처리할 때 보다 효과적
- 하지만 단순히 횟수로 단어를 unk로 분류하게 되면 unk 토큰이 가지는 의미가 모호해지며 중요한 의미를 가지는 단어를 놓칠 수 있게 됨
- pre-trained word vector를 사용하는 방법이 가장 효과적

![Untitled](Contextual%20daf9c/Untitled%204.png)

- 단어를 표현하는 함에 있어서 word embedding이라는 방법을 사용할 때도 고려해야할 문제가 있음
- 바로 1개의 단어를 1개의 vector로 표현하는 것은 동음이의어와 같은 문제를 발생시킬 수 있음

![Untitled](Contextual%20daf9c/Untitled%205.png)

- 바로 이러한 context, 문맥을 고려하여 word vector를 생성할 수 있는 방법으로 Neural Language Model을 제시할 수 있다고 함
- LSTM 구조가 적용된 해당 language model은 문장의 sequence를 고려하여 다음 단어를 예측하는 모델로 sequence의 학습을 통해 문맥을 어느정도 유지할 수 있음
- 대표적인 사례가 ELMo 모델

# 2. Pre-ELMo and ELMo

- Pre-ELMo는 ELMo의 기본적인 구조가 굉장히 유사하며 RNN을 통해 context가 유지되는 word vector를 찾기 위한 가정이 적용되어 있음
- 또한 Neural Language Model로 word vector를 pre-trained한 뒤에 추가로 학습이 진행되는 Semi-supervied 접근법이 활용됨

![Untitled](Contextual%20daf9c/Untitled%206.png)

- 어떠한 input이 들어오게 되면 2가지 다른 방법으로 parallel하게 단어가 임베딩됨
- 한 쪽으로는 Word2vec과 같은conventional word embedding 모델, 다른 한쪽으로는 bi-LSTM과 같은 RNN이 적용된 word embedding 모델을 통과하게 됨

![Untitled](Contextual%20daf9c/Untitled%207.png)

- 그렇게 2가지 접근법으로 만들어진 word vector를 NER과 같은 small task-labeled에 사용하게 됨

![Untitled](Contextual%20daf9c/Untitled%208.png)

- 즉, context가 아닌 특징 자체를 고려한 word vector와 데이터의 문맥(context)를 고려한 word vector를 모두 사용하였고 의미를 반영했다고 볼 수 있음

![Untitled](Contextual%20daf9c/Untitled%209.png)

- pre-trained model로 학습한 word vector와 char-CNN model로 학습한 word vector를 concat하는 과정을 수식을 통해 확인할 수 있음

![Untitled](Contextual%20daf9c/Untitled%2010.png)

- ELMo의 특징: 모든 문장 전체를 사용해서 문맥이 반영된 word vector를 학습하는 모델
- 전체적인 흐름으로는 char-CNN과 같은 모델로 단어의 특징을 고려한 word vector를 구하고 해당 vector를 LSTM의 Input으로 사용해서 최종 word embedding 결과를 도출하는 과정을 가짐

![Untitled](Contextual%20daf9c/Untitled%2011.png)

- 순방향 Language Model과 역방향 Language Model이 모두 적용된 bidirectional Language Model
- 순방향 모델은 문장 sequence에 따라 다음 단어를 예측하도록 학습되며 역뱡향 모델은 뒤에 있는 단어를 통해 앞 단어를 예측하도록 학습
- ELMo는 모든 layer의 출력값을 활용해서 최종 word vector를 임베딩함
- 최종 layer의 값들로만 word vector로 사용했던 이전 모델들과의 차이점을 보임

![Untitled](Contextual%20daf9c/Untitled%2012.png)

- 첫번째 LSTM layer에서는 char-CNN의 결과에residual connection을 적용함
- residual connection을 적용함으로써, char-CNN으로 반영된 단어의 특징을 잃지않고 유지할 수 있고, 학습 시 역전파를 통한 gradient vanishing 문제를 완화시킬 수 있음
- 그렇게 첫번째 layer에서 얻은 결과로 다음 단어를 예측하는 LSTM layer를 통과하게 됨

![Untitled](Contextual%20daf9c/Untitled%2013.png)

- 그렇게 순방향 모델과 역방향 모델이 각각 그 다음과 그 전의 단어를 예측하기 위해 학습된 layer 정보다 다음과 같이 표시할 수 있음
- 해당 그림은 'read'의 다음 단어를 예측하는 순방향 모델, 그리고 'read'의 전 단어를 예측하는 역방향 모델의 'read' 부분만을 추출한 결과
- 즉, 'read'에 대한 각 layer별 출력값이 존재하며 확인할 수 있음

![Untitled](Contextual%20daf9c/Untitled%2014.png)

- ‘read’에 대한 각 layer의 출력값만 추출하고 두 모델의 데이터를 모두 활용하기 위해 concat할 수 있음

![Untitled](Contextual%20daf9c/Untitled%2015.png)

- 그렇게 concat된 layer별 출력에 정규화된 가중치를 곱해주며 학습을 통해 최적화함
- 최종적으로는 모든 layer의 벡터를 더해 하나의 임베딩 벡터라는 word vector를 듦

![Untitled](Contextual%20daf9c/Untitled%2016.png)

- 이렇게 되면, syntax 특징을 고려한 하위 layer의 정보와 semantics 특징을 고려한 상위 layer의 정보를 모두 활용함으로 효과적인 word vector라고 할 수 있음
- syntax 정보는 tagging, NER 등과 같은 단어 자체가 가지고 있는 특징을 의미하고 semantics 정보는 문맥이 고려된 특징

![Untitled](Contextual%20daf9c/Untitled%2017.png)

# 3. ULMfit and onward

- ELMo를 시작으로 contextual word representaion에 대한 관심이 증가하게 되면서 ULMfit이 등장하게 됨

![Untitled](Contextual%20daf9c/Untitled%2018.png)

- 먼저 앞서 언급한 transfer learning부분으로, Language Model이 대량의 corpus를 기반으로 학습하게 됨
- 다음으로 기학습된 언어 모델을 토대로 주어진 task에 맞게 추가 학습되면서 업데이트를 진행하게 됨
- 마지막으로는 최종적으로 classifier가 출력되는 부분으로 word vector가 결국 분류기로 출력됨을 의미

# 4. BERT

![Untitled](Contextual%20daf9c/Untitled%2019.png)

- BERT는 transformer의 encoder 부분을 차용하여 Bi-directional Language Model을 구축한 모델
- 보통 bi-direction은 ELMo에서도 확인할 수 있듯이, 순방향과 역방향이 모두 적용된 의미를 가짐
- 하지만 이러한 bi-direction의 순방향과 역방향은 각각 독립적으로 적용되기 때문에 순방향 혹은 역방향이라는 sequence를 가지게 됨

→ 결국 순방향과 역방향 모델을 모두 적용함에도 불구하고 단어를 예측함에 있어 전체 단어를 모두 활용할 수는 없다!

![Untitled](Contextual%20daf9c/Untitled%2020.png)

- 양방향 예측이 한번에 가능하다면 가운데 들어갈 어떤 단어를 예측함에 있어 보다 많은 정보를 기반할 수 있음

![Untitled](Contextual%20daf9c/Untitled%2021.png)

- BERT는 이러한 문제를 Masked Language Model을 적용함으로써 해결
    - 문장에 존재하는 단어 중 일부를 masked 처리한 뒤에, maksed 처리된 단어를 나머지 모든 단어로 예측하는 과정
    
    → 순방향과 역방향의 모든 문맥을 한번에 고려가능
