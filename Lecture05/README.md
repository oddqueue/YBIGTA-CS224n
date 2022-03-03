# Dependency Parsing

## **Contents**

1. Introduction
2. Syntactic Structure : Constituency and Dependency
3. Dependency Grammar and Treebanks
4. Transition-based dependency parsing
5. Neural Dependency parsing

### 1. Introduction

- 문장의 정확한 이해를 위한 분석 방법 → Parsing

![Untitled](Dependency%2088878/Untitled.png)

- Parsing : 각 문장의 문법적인 구성 또는 구문을 분석하는 과정
- 주어진 문장을 이루는 단어 혹은 구성 요소의 관계를 결정하는 방법 : Constituency, Dependency

— Constituency parsing : 문장의 구성요소를 파악하여 구조를 분석

— Dependency : 단어간 의존 관계를 파악하여 구조를 분석

### 2. **Syntactic Structure : Constituency and Dependency**

1) Constituency parsing

- 문장을 구성하고 있는 구를 파악하여 문장구조를 분석하는 방법

→ Context free grammar이라고도 부름

![Untitled](Dependency%2088878/Untitled%201.png)

- 보통 각 단어들은 해당 단어의 문법적 의미를 가지고 있음
- 영어와 같이 어순이 비교적 고정적인 언어에서 주로 사용됨
- ex) ‘the’ → 관형사(Det), ‘cat’ → 명사(N)

: 단어들은 단어끼리 결합해서 어떠한 구를 구성

: 구성된 구는 구와 또 결합 → 문장 생성

![Untitled](Dependency%2088878/Untitled%202.png)

- 주어진 문장이 있을 때,
- 문장을 이루는 구(phrase)를 파악할 수 있음
- 문장 → 구 → 단어, 전과 역순으로 분류가능
- ‘The cuddly cat’ : NP
- ‘by the door’ : PP
- 최종적으로는 각 단어가 가지는 문법적 의미까지 분해

2) Dependency parsing

![Untitled](Dependency%2088878/Untitled%203.png)

- 문장에 존재하는 단어 간 의존 또는 수식 방향으로 관계를 파악하여 문장 구조를 분석 하는 방법
    - ‘head’, ‘governor’ : 수식을 하는 단어
    - ‘dependent’, ‘modifier’ : 수식을 받는 단어
    - 단어 간 관계를 정립하게 되면 → 트리구조로 표현 가능

### 3. Dependency Grammar and Treebanks

![Untitled](Dependency%2088878/Untitled%204.png)

- Sequence형태, Tree 형태
- 2가지 형태의 결과는 정확히 동일한 output을 가져야 함

![Untitled](Dependency%2088878/Untitled%205.png)

- Dependency structure은 2가지 형태로 표현 가능
- 가상의 노드 Root를 추가하여 모든 성분의 최종 HEAD를 Root로 설정
- Root는 모든 단어가 최소 1개 노드의 dependent가 되도록 해야함
- 화살표는 head에서 dependent로 향함
- 화살표 위 표시는 단어 간 문법적 관계(dependency)를 의미
- 화살표는 순환하지 않으며 중복의 관계가 형성되지 않음

**Dependency Conditioning Preferences (보편적 특징)**

- Bilexical affinities

: 두 단어 사이의 실제 의미가 드러나는 관계 

- Dependency distance

: dependency의 거리를 의미, 주로 가까운 위치에서 dependent 관계가 형성

- Intervening material

: 마침표, 세미클론과 같은 구두점을 넘어 dependent한 관계가 형성되지는 않음

- Valency of heads

: head의 좌우측에 몇 개의 dependents를 가질 것인가에 대한 특성

### Transition-based dependency parsing

![Untitled](Dependency%2088878/Untitled%206.png)

- Transition-based는 두 단어의 의존 여부를 차례대로 결정해나가면서 점진적으로 dependency structure을 구성해 나가는 방법
- 문장에 존재하는 sequence를 차례대로 입력하게 되면서 각 단어 사이에 존재하는 dependency를 결정해나가는 방법 → Deterministic dependency parsing
- 문장의 sequence라는 한 방향으로 분석이 이루어 지기 때문에 모든 경우를 고려하지는 못함
- 분석 속도는 빠름, but 낮은 정확도

![Untitled](Dependency%2088878/Untitled%207.png)

- Parsing과정에는 Buffer, Stack, Set of Arcs
- Input으로 문장이 입력되면, 3가지 구조를 거쳐서 output 도출

- 초기 상태 → Buffer에는 주어진 문장이 토큰 형태로 모두 입력됨
- Stack에는 Root 존재
- Set of Arcs에는 parsing의 결과물이 담김

![Untitled](Dependency%2088878/Untitled%208.png)

Parsing 과정:

- Buffer에 존재하는 문장의 토큰이 Stack으로 이동하게 되면서 어떠한 state를 형성하게 됨
- 해당 state를 기반으로 Decision이라는 결정을 내림
- Output으로 결과가 이동

a) BUFFER('Joe', 'ate', 'the', 'cake')

문장의 첫번째 토큰인 ‘Joe’가 먼저 Stack으로 이동

b) Stack

스택에는 ‘ROOT’와 ‘Joe’가 존재하게 되면서 어떠한 state를 형성 → 이때 state를 통해 Decision이라는 결정을 내림

(Decision 결정방법 : SVM, Neural Network등의 모델)

- Shift: BUFFER에서 STACK으로 이동하는 경우
- Right-Arc: 우측으로 dependency가 결정되는 경우
- Left-Arc: 좌측으로 dependency가 결정되는 경우

c) Set of arc

이러한 dependency statement가 모이면 이를 바탕으로 output을 제공 (tree 형태)

ex) ‘Joe ate the cake’

![Untitled](Dependency%2088878/Untitled%209.png)

(1) STACK에 ROOT만이 존재하는 state는 'shift'라는 decision이 내려지게 되면서 'Joe'가 BUFFER에서 STACK으로 이동하게 됨

(2) STACK에는 ROOT, Joe라는 state가 'shift'라는 decision이 내려지게 되고 'ate'이 BUFFER에서 STACK으로 이동하게 됨

(3) STACK에는 ROOT, Joe, ate라는 state는 'ate'이 'joe'를 수식하는 'Left-Arc'라는 decision이 내려지게 됨

![Untitled](Dependency%2088878/Untitled%2010.png)

(4) STACK에는 ROOT, ate라는 state가 'shift'라는 decision이 내려지게 되면서 'the'가 BUFFER에서 STACK으로 이동하게 됨

![Untitled](Dependency%2088878/Untitled%2011.png)

(5) STACK에는 ROOT, ate, the라는 state에서도 어떠한 관계가 형성되지 않는다고 판단했기에 'shift'라는 decision이 내려지게 되고 'cake'가 BUFFER에서 STACK으로 이동하게 됨

(6) STACK에는 ROOT, ate, the, cake라는 state에서 'cake'가 'the' 수식하는 'Left-Arc'라는 decision이 내려지게 되고 해당 결과(cake, det, the)는 Set of Arcs로 이동하게 됨

(7) STACK에는 ROOT, ate, cake라는 state에서 'ate'가 'cake'를 수식하는 'Right-Arc'라는 decision이 내려지게 되고 해당 결과(ate, dobj, cake)는 이동함

(8) STACK에는 ROOT, ate라는 state에서 BUFFER에 토큰이 존재하지 않기 때문에 'shift'가 발생할 수 없습니다. 하지만 모든 토큰은 하나의 dependent를 가진다는 dependency parsing의 특징으로 'ROOT'가 'ate'를 수식하는 'Right-Arc'라는 decision이 내려지게 되고 해당 결과(Root, root, ate)는 이동함

→ 트리형태로 표현이 가능해짐

![Untitled](Dependency%2088878/Untitled%2012.png)

- Decision을 결정하기 위해서는 SVM, NN, maxnet과 같은 모델이 적용됨
- 이 과정에 state를 모델이 unput으로 받기 위한 state 임베딩 과정이 필요함

![Untitled](Dependency%2088878/Untitled%2013.png)

해당 state에서 state가 임베딩과정을 보겠음

- 임베딩을 위한 feature을 표현하기 위해서 notation을 확인할 수 있음
- 이 때, 각 토큰의 tag를 활용

- s1W : Stack의 첫번째 단어 → the

Stack 구조 : 후입선출, 가장 늦게 들어간 것이 먼저 나옴

- b1W : Buffer의 첫번째 단어 → cake
- 해당 notation의 결과를 차자볼 수 없을 때 : NULL

![Untitled](Dependency%2088878/Untitled%2014.png)

- Notation을 기반으로 indicator feature라는 조건을 설정함으로써 state를 임베딩할 수 있음
- STACK의 첫번째 단어가 'the'이고 STACK의 첫번째 태그가 'DT'면 1 아니면 0 이라는 값이 부여지게 됩니다. 이러한 조건들로 하여금 해당 state를 10^6, 10^7 차원의 벡터로 표현하게 되는 것이 state를 임베딩하는 방법

![Untitled](Dependency%2088878/Untitled%2015.png)

- 1과 0인 binary로 표현 → sparse한 형태
- 10^6, 10^7 차원을 모두 계산하여야 하기 때문에 feature 연산이 대부분을 차지
- 이는 단어의 태그의 의미를 반영하지는 못하는 단점이 있음

### 5. Neural Dependency parsing

![Untitled](Dependency%2088878/Untitled%2016.png)

- Neural network 형태

![Untitled](Dependency%2088878/Untitled%2017.png)

Input

- words
- POS tag
- Arc labels

1) Words feature로 들어가게 되는 데이터는 총 18개

- STACK과 BUFFER의 TOP 3 words (6개)
- STACK TOP 1, 2 words의 첫번째, 두번째 left & right child word (8개)
- STACK TOP 1,2 words의 left of left & right of right child word (4개)

2) POS tags feature로 들어가게 되는 데이터는 words feature에서 들어가는 데이터의 태그를 의미하기 때문에 똑같이 18개

3) arc labels에서는 STACK과 BUFFER의 TOP 3 words 6개를 제외한 12개

![Untitled](Dependency%2088878/Untitled%2018.png)

words feature는 (18 x 단어의 총 개수)

POS tag feature는 (18 x POS tag 총 개수)

Arc-label feauture는 (12 x label 총 개수)

![Untitled](Dependency%2088878/Untitled%2019.png)

- 이렇게 포함된 데이터를 원핫으로 표현한 후에 word embedding matrix를 참고하여 해당 토큰의 벡터를 가져올 수 있음
- 각 토큰별로 벡터가 있는 상태에서 모두 concat한 뒤에 input layer에 들어가게 됨

![Untitled](Dependency%2088878/Untitled%2020.png)

- 각 feautre별로 임베딩된 벡터가 input layer를 입력된 이후에 hidden layer에서는 일반적인 feed forward network의 계산이 진행
- 활성화 함수: cube function
- cube function을 적용하게 되면 input으로 들어가는 3개의 feature인 word, POS tag, arc-label의 조합이 계산되면서 feature간의 상호관계를 파악할 수 있음

![Untitled](Dependency%2088878/Untitled%2021.png)

- softmax()를 통해 Decision으로 나타날 수 있는 모든 경우의 수의 확률을 구함
- 가장 높은 확률로 분류될 Decision이 선택되게 되며 해당 state에서는 conjucation의 관계를 가지는 Left-Arc의 경우가 Decision으로 선택됨

![Untitled](Dependency%2088878/Untitled%2022.png)

- UAS : Arc 방향만 예측
- LAS : Arc 방향과 관계 label까지 예측

### Additional Part

요즘 대세는 dependency parser로 굳어지고 있음 

- 언어에 상관없이 적용가능
- 문장의 의미 구조 파악 가능
- 어떤 단어를 표현함에 있어서 그것을 다른 단어와의 관계로 설명하려고 한다는 점에서 distributed word representation과 비슷한 맥락 → Good
