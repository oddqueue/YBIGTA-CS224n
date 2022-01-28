# Lecture 6

### Language Models and RNNs

---

### Overview : Language Models(언어모델) 그리고 RNN

|Index|Subtitle|
|--- | --- |
|6.1.| Language Modeling(LM) |
|6.2.| N-gram Language Model|
|6.3.| Neural Language Model|
|6.4.| RNNs |
|6.5.| Evaluation and Application of Language Model|

---
### 6.1. Language Modeling(LM)
- **어떤 단어가 빈칸에 등장할지** 예측하는 Task
- *the students opened their* ____ <br/>
    <img src = "https://user-images.githubusercontent.com/75057952/151372590-dcbbafae-ddcf-4319-8e81-798a6e85f101.svg" width = "200dp" /> <br/>
- LM은 각 term에 해당하는 조건부 확률을 도출하므로, 이를 곱하면 특정한 텍스트 x(1), ..., x(t)가 등장할 확률 역시 구할 수 있다.
- 자동완성에도 활용
---
### 6.2. N-gram Language Model
- N-gram LM의 경우 주어진 corpus에서의 statistical한 정보를 학습해 prediction을 수행한다.
- N개의 연속한 단어가 뭉친 덩이를 **N-gram**이라고 부른다.
- **Uni**gram, **Bi**gram, **Tri**gram, **4**-gram, ....
- N-gram LM의 학습은 특정 길이의 N-gram이 얼마나 빈번하게 등장하는지 계산해 prediction에 활용하는 방식이다.
- **Simplifying Assumption**
    - (t+1) 번째로 등장할 단어는 그 이전의 (n-1)개 단어에만 의존한다. <br/>
        <img src = "https://user-images.githubusercontent.com/75057952/151372597-b16d712c-5371-4218-a8d2-88ab5eb4cfb3.svg" width = "600dp" /> <br/><br/>
    - n-gram, (n-1)-gram probability는 large corpus에서 등장하는 빈도를 세어서 구할 수 있다.
    <br/>
    
    >***Application : "the students opened their ____"***
    
    - 이 예시에서 ____에 들어갈 단어를 찾으려면 아래 값을 구해야 한다.
    <img src = "https://user-images.githubusercontent.com/75057952/151372599-b893c916-5cf2-4e2a-af5d-b958e8ccbe3b.svg" width = "600dp" /> <br/>
    
    > **Sparsity problem 1**
    
    - 분자가 0인 경우(train corpus에 이러한 target이 없을 경우)
        - Prob = 0
        - Smoothing : Vocab 속의 모든 단어에 작은 값을 더해준다.
        <br/>
        
    > **Sparsity problem 2**
    
    - 분모가 0인 경우(train corpus에 이러한 feature가 없을 경우)
        - train 과정에서 학습하지 않은 조합을 예측할 때 발생하는 문제
        - Backoff : n-gram의 count가 없을 경우, 마지막 단어를 제외하고 (n-1) gram의 count로 대체하는 방법
        <br/>
        
    > **Storage problem**
    
    - 큰 n에서는, sparsity problem이 더욱 심화되고 모델이 지나치게 커진다.
    - 일반적으로는 n이 5보다 작아야 한다.
    <br/>
    
    > **Incoherence problem**
    
    - 하지만 작은 n에서는, 앞선 문맥을 반영하기 어렵다.
    <br/>
- Text Generation
    - N-gram LM으로 문장 생성도 가능하다.
    - 한 단어를 덧붙인 문장을 다시 input으로 넣어 다음에 올 단어를 예측하면서 재귀적으로 문장의 길이를 늘려 나간다.
    - 문법적으로는 맞지만, 잘 호응하지 않는 문장이 생성된다.
    - Lexical : Generally local, n이 작을 때에도 적절하게 학습한다.
    - Contextual : Generally global, n이 작으면 학습하기 어렵다.
        
---
### 6.3. Neural Language Model

- 단어들의 sequence를 input으로 전달하면 다음 단어에 대한 probability distribution을 도출하는 NN

> **Structure**

- NER(Named Entity Recognition) Recap :
    - 특정 window 내의 단어를 벡터로 임베딩하고, 간단한 Neural Net 통과하면 분류됨
- 마찬가지로 window-based Neural Model 적용하면 단어를 임베딩해서 input으로 전달하면, softmax를 거쳐 확률분포를 도출함
<img src = 'https://user-images.githubusercontent.com/75057952/151372614-b8e50079-e93e-4ef0-b298-4e8581a9066b.jpg' width = "300dp">

> **장단점**

- n-gram LM의 sparsity 문제를 해결할 수 있고, 모든 n-gram에 대한 관측을 저장할 필요가 없음
- word embedding, weight matrix의 부분행렬끼리 sharing이 없음 : 비효율적인 학습
<img src = 'https://user-images.githubusercontent.com/75057952/151372618-297ac4bb-1dc3-40e5-833a-d635a43018d5.jpg' width = "350dp">
- 그러나 고정된 window(예를 들어 4개 단어)의 크기가 너무 작고, window가 커지면 모델 크기와 Weight matrix W가 함께 커짐
- 어떤 input length라도 모델의 구조 변화 없이 처리할 수 있는 Network Architecture가 필요함 : *RNN*

---

### 6.4. RNNs

> **Architecture**

<img src = 'https://user-images.githubusercontent.com/75057952/151372622-de515063-31b9-4e75-be9e-fccfed2cca8d.jpg' width = "600dp">
<img src = 'https://user-images.githubusercontent.com/75057952/151372626-2d36490c-44d3-41bc-bd26-f0fa56fd9161.jpg' width = "600dp">

- Word Embedding 역시 학습할 수 있다.
    - Pre-trained E를 가져와서 word embedding을 fix
    - Pre-trained E를 가져오지만 custom data가 충분한 경우 fine-tuning
    - E를 random variable로 innitialize하고 scratch에서부터 re-train하는 방법도 있음
- RNN의 장단점
    - 임의의 길이를 가진 input을 처리할 수 있음
    - 이론적으로는 굉장히 멀리 떨어진 단어의 정보도 가져올 수 있음
        - 실제로는 어려움(Vanishing/Exploding Gradient)
    -  Model size가 input에 따라서 변하지 않음
    -  동일한 Weight matrix를 활용하므로(timestep마다), input이 process되는 과정에 대칭성이 확보됨
        - window-based neural net에서는 process의 대칭성이 확보되지 않고, 비효율적이라고 지적함
    - recurrent computation이 느리다는 문제점

> **Training**

- Corpus of text x(1), ..., x(T) : train dataset
- RNN-LM에 x(i)의 embedding을 input으로 넣어 y^(t)를 step t마다 계산함
- Loss function : true distribution과 predicted distribution 사이의 cross-entropy loss로 계산함 <br/>
    <img src = "https://user-images.githubusercontent.com/75057952/151372601-6c894e71-07f3-411a-b86b-15f7a1cc650b.svg" width = "400dp"> <br/>
- 최종 overall loss는 T개 step에 대한 Loss를 평균 내어 사용<br/>
    <img src = "https://user-images.githubusercontent.com/75057952/151372605-c4cdbe20-d1f1-4e35-bc21-c3f81caf9ae7.svg" width = "250dp"> <br/>

<img src = "https://user-images.githubusercontent.com/75057952/151372629-305f0f5d-dd07-4fb5-af16-b658903f06e9.jpg" width = "400dp"> <br/>
- 하지만 전체 corpus x(1), ..., x(T)에 대해 모두 계산하는 것은 비용적으로 어려움
    - 그래서 x(i)를 단어가 아닌 sentence/document 단위로 설정할 수 있음
    - SGD를 활용해 mini-batch 단위로 gradient descent을 수행할 수도 있음

> **Backpropagation**
- Backpropagation Recap(Lec 4) <br/>
    <img src = "https://user-images.githubusercontent.com/75057952/151372631-5f9b7d2b-6ff4-46ca-a32b-4338e56dfeb1.jpg" width = "400dp"> <br/>
- Backpropagation Through Time(BPTT) <br/>
<img src = "https://user-images.githubusercontent.com/75057952/151372634-c2c812a5-a973-46d8-bc84-8dc070c9d7a2.jpg" width = "600dp"> <br/>
> **Text Generation**

<img src = "ttps://user-images.githubusercontent.com/75057952/151372636-8f2a55d5-87d0-4708-b2be-26d865603c1e.jpg" width = "400dp"> <br/>
- 버락 오바마의 Speech / Harry Potter Text로 Train한 RNN-LM
    - n-gram LM에서는 정말 국소적으로만 말이 되는 문장을 만들어 냈음(Lexically Correct)
    - RNN LM은 보다 넓은 구간에서 말이 되는 문장을 생성, contextually correct sentence를 만들기도 함('gold price'에 대한 주제를 벗어나지 않았음)
    - 특정 'style'의 text generation
- 특정 color 색상에 맞는 paint color name genearation task
- Opening/Closing Quote
- Beam Search

---
### 6.5. Evaluation and Application of Language Model

> **Evaluating Language Model : Perplexity**

<img src = "https://user-images.githubusercontent.com/75057952/151372612-89a4c82a-05ec-42a3-b2a1-6c8d662d524e.svg" width = "400dp"> <br/>
- corpus 크기가 커질수록 P_LM term을 곱한 값이 커져 가기 때문에 1/T 제곱하여 normalize
- J(t)의 평균으로 구한 J의 exponential과 동일함
- perplexity가 낮을수록 좋은 언어모델
- 실제로 n-gram LM보다 complex RNN LM이 perplexity가 낮음

> **Why Language Model Matters?**
- Language Modeling은 특정한 언어에 대한 이해 척도를 가늠하라 수 있는 Benchmark task
- Language Modeling은 여러 NLP task의 subcomponent 역할을 하고 있음
    - Tasks
        - Generating Text
        - Estimating Probability of Text
    - Applications
        - Predictive typing
        - Speech recognition
        - Handwriting recognition
        - Spelling/grammar correction
        - Authorship identification
        - Machine translation
        - Summarization
        - Dialogue ...

> **RNN is useful**
- RNN은 language model build를 위한 주요 방법론
- LM 외에도 다양한 용도로 활용
- Tagging
    - Part-of-speech tagging(POS)
    - Named Entity recognition
- Classification
    - Sentiment classification
        - 문장의 단어들이 RNN input이 되고, sentence encoding을 도출(max, mean of word encoding can be a sentence encoding)
        - 이를 classifying network의 input으로 다시 활용
- Encoder module
    - Question answering
        - SQuAD challenge
        - Question-Context input의 encoder로 활용
    - Machine Translation
- Vanilla RNN to GRU, LSTM...

