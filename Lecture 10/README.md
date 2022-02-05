# Lecture 10

### Question Answering

*Contributor : Junha Park*

---

### Overview : Question Answering

|Index|Subtitle|
|--- | --- |
|10.1.| Motivation & History |
|10.2.| The SQuAD dataset |
|10.3.| Stanford Attentive Reader Model |
|10.4.| BiDAF |
|10.5.| Recent & Advanced Architectures |

---

### 10.1. Motivation & History

<p align="center">
<img src = "https://user-images.githubusercontent.com/75057952/152647936-813268af-e8a2-46ad-8eb9-0b85036785ed.JPG" width= "300dp">
</p>

- 이 그림은 사실 Question Answering의 예시는 아니다.
- 지금 화면은 단지 Question에 해당하는 특정 문서, 웹페이지를 보여주고 있다: *Simply Returning Relevant Documents*
- 실제로는 문서의 텍스트 속에서 대답에 해당하는 부분을 찾아내는 것이 핵심적이다.
- Question Answering Task는 크게 두 부분으로 구성된다.
    - *Finding Documents that might contain an answer*
        - Traditional Information retrieval(IR)/web search (cs276)
    - *Finding answer in paragraph or a document*
        - **Machine Reading Comprehension problem(MRC)**

- MRC Task에서 질문의 종류
    - Factoid type questions : 미국의 대통령은?
    - List type questions : 탄산음료 브랜드에는 뭐가 있는지?
    - Confirmation questions : Y/N 질문, 지금 1시인가요?
    - Causal questions : 왜 ~ 했는지?
    - Hypothetical questions : 특정한 정답이 없음, 북한이랑 전쟁하면 어떻게 될지?
    - Complex questions : 저출산을 어떻게 해결할 수 있을까?

> **History of Reading Comprehension**

- 1977, Yale A.I. Project : early NLP work attemptimg reading comprehension
- 2013, Chris Burges : Machine Comprehension Test(MCTest) 을 개념화하였다.
- 2015/2016, Large dataset 구축되면서 supervised neural system 이 많이 제안됨
    - DeepMind CNN/DM dataset
    - SQuAD dataset
    - MS MARCO, TriviaQA 등등...

- Reading Comprehension Task는 '배경 지식'의 여부에 따라서 다시 분류할 수 있다.
    - 특정 분야의 지식에 특화된 경우
    - Open-domain(Jeopardy, DeepQA)
    - Open-domain QA의 경우, 1964년 처음으로 dependency parsing을 수행한 다음 tree 형태의 parse matching을 통해서 answer를 도출하는 방법론을 사용함


- Traditional Landscape of NLP Questing Answering Task(circa 2003, architecture of LLC QA system)
    - Factoid Question, List Question, Definition Question의 세 종류 질문이 Question Processing을 거친다.
        - Question Parsing 과정을 거쳐 '무엇을 묻는 것'인지 확인함
    - Document는 Document Processing을 거친다.
    - Faction Answer, List Answer가 output으로 도출된다.
    - 구조를 보면 Complex Mutli-part로 구성되어 있다.
    - NER task의 결합으로 해석할 수 있었기 때문에, 단순 사실을 묻는 'factoid' question에서는 complex system이 나름 잘 작동했다.

---

### 10.2. SQuAD Dataset

> **SQUAD 1.0**

- Question, Passage, Answer 쌍으로 구성된 학습 데이터셋
- 100K Examples
- 반드시 Answer가 Passage안에 하위 시퀀스로 존재해야 함, **extractive question answering**
- *Gold Answers*는 3개까지 수집

> **Evaluation v1.1**

- SQuAD evaluation metric(2가지)
    - Exact Match Accuracy : 0/1 binarize, 3개의 gold answer와 맞는 것이 있는지 확인
    - F1-score : More reliable
    - 두 metric 모두 punctuation, article 무시하고 계산(a, an, the)

> **SQuAD 2.0**

- Defect of SQuAD 1.0 : 모든 질문에 대해 정답이 paragraph내에 존재한다. 그래서 기존 NLP system들은 암묵적으로 가장 그럴듯한 정답을 도출하도록 설계되어 있었다.
- SQuAD 2.0의 데이터셋에서 train 질문의 1/3은 정답이 없고, dev/text 질문의 1/2은 정답이 없도록 했다.
- NoAnswer를 도출해야 score 1이 추가된다.
- 그래서 SQuAD 2.0에서는 주어진 paragraph을 통해서 질문에 대답할 수 있을지 판단하는 task가 추가되었다.
- 이 Task에 대한 가장 일반적인 접근은, 각 정답별로 확률을 계산해 threshold를 넘기지 못하면 답이 없다고 판단하는 것이다.
- 혹은 answering confirm을 위한 또 다른 component를 설계할 수도 있다: *NLI(Natural Language Inference), "Answer validation"*
- 대체로 BERT 모델이 SOTA를 찍었다.

> ***SQuAD limitations**

- SQuAD는 *span-based* answer와의 쌍으로 구성되어 있어서 정답의 종류가 한정되어 있다.
- 문장의 참거짓을 판별하거나, 개수를 세거나, Implicit하게 인과관계를 묻는 질문은 없다.
- Passage를 보고서 정보를 학습해서 문제를 풀기보다는, lexical & syntactic matching을 통해서 answer를 span한다고 이해할 수 있다.
- Multi-fact/sentence inference는 어렵고, 단순한 coreference를 확인하는 수준에 그친다.
- 그럼에도 SQuAD는 well-targeted/structured/clean dataset이다.

---

### 10.3. Stanford Attentive Reader Model

- Stanford Attentive Reader Model은 Question Vector를 생성하고, Passage Vector를 생성해, Attention을 통해 정답이 있는 시퀀스의 시작 토큰과 종결 토큰 위치를 예측한다.

<p align="center">
<img src = "https://user-images.githubusercontent.com/75057952/152647937-ab4ebecc-494c-48a0-966a-018a73c28fd0.jpg" width= "700dp">
</p>

- **Stanford Attentive Reader++** 에서의 개선점
    - Single Bi-LSTM Layer >> 3 Layers
    - learn new parameter w : Bi-LSTM의 마지막 벡터를 단순 concat하기보다도 w와의 similarity score를 계산해 가중치 합산을 한 다음 concat하는 방식
    - GloVe emedding & POS/NER one-hot tags concatenated

---

### 10.4. BiDAF

> **Main Contribution of BiDAF**

- GloVe word embedding과 함께 character 수준의 embedding을 함께 수행, 이때 Char-CNN을 활용한다.
- Attention Flow Layer를 새롭게 도입하였다. (Query2Context, Context2Query)
- 마지막 LSTM Modeling 이후 Output을 도출할 때에도 단순하게 attention을 통해서 Start/End 토큰 위치를 결정하는 것이 아니라 Neural Network를 활용한다.
    - Attention은 단순한 가중치 선형결합이라면, 이를 NN으로 대체하여 보다 좋은 성능을 내고 있다.

> **BiDAF Architecture**

<p align="center">
<img src = "https://user-images.githubusercontent.com/75057952/152647939-4921ecdf-082c-4361-a7ad-0d73e8b864fb.jpg" width= "700dp">
</p> <br/>
<p align="center">
<img src = "https://user-images.githubusercontent.com/75057952/152647941-6c48df4a-3849-4433-a6cd-583efe5c350e.jpg" width= "900dp">
</p>

---

### 10.5. Recent & Advanced Architectures

- BiDAF 이후 Bidirectional Attention Flow를 활용하는 model variation들이 많이 생겨남
- Two-way attention between the context and question이 BiDAF의 중요한 Contribution
- [Dynamic Coattention Networks(DCN)](https://arxiv.org/pdf/1611.01604.pdf)
- [FusionNet](https://arxiv.org/pdf/1711.07341.pdf)
- Attention function의 Variation으로도 다양한 형태가 있음
    - MLP(Additive) form
    - Bilinear(Product) form
    - Nonlinearlity를 추가한 Attention Function 등등...

<p align="center">
<img src = "https://user-images.githubusercontent.com/75057952/152647942-c2202df6-9f1e-4025-9605-8c14641698e2.jpg" width= "500dp">
</p> 