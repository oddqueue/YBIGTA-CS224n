# Lecture 1

### Introductions and Word Vectors

---

### 강좌 소개

- NLP에 사용되는 딥러닝의 기초에 대한 이해
- Basic Model → RNN → Attention → Transformers 순서
- ***Pytorch*** Implementation 수행

---

### 단어의 의미를 표현하는 방법

- **기의와 기표**
    - signifier(symbol) ↔ signified(idea or thing, concept)
    - denotational semantics
- **WordNet : 고전적인 NLP 해결책**
    - 동의어와 상의어 등의 list를 포함하고 있는 thesaurus(사전)
    
<aside>
💡 WordNet-like resources의 취약점

- 문맥에 따라 다름 : missing nuance
- 새로운 의미 등장(신조어)
- 주관적인 분류 기준, requires human labor
- word similarity의 정량적 개념 부재
</aside>
    
- **단어를 원핫 벡터로 변환하기**
    - all non-identical word vectors are orthogonal
    - vector dimension = vocabulary에 속한 모든 단어의 개수
    - **취약점**
        - word similarity의 정량적 개념 부재
- **분배 가설(distributional semantics, hypothesis)**
    - 단어의 의미는 인접하게 위치한 다른 단어들에 의해 결정된다.
    - 문맥(context) := 근처에 등장하는 words의 집합
    - ***context of w builds a representation of w***
    - 단어 벡터 → word vectors, word embeddings, neural word representations
    - n-dimensional  단어 벡터들의 distribution을 ***principal axis PC1, PC2***에 대해 projection
        - 2D scatter plot처럼 나타남
        - 의미적으로 유사한 단어들끼리 cluster되는 경향

---

### Word2vec (Mikolov et al. 2013)

<aside>
💡 Word2vec의 아이디어

- large corpus of text가 주어짐(typically large but rare words are truncated...)
- 고정된 vocabulary 내의 모든 단어는 vector로 표현 가능
- 텍스트의 position **t**를 순회할 때 중심 단어(center) **c**와 문맥 단어(outside) **o**를 정의
- **similarity of word vectors for c, o → P(c|o), P(o|c) 계산에 활용**
- 위의 Probability를 maximize하는 c, o word vector adjustment
</aside>

<aside>
💡 How Word2Vec Works

- Maximize likelihood, -log(L) = J
- Gradient Descent with Loss function
- SGD is strongly recommended due to expensive gradient calculation
- 하나의 window, with m=0 → Gradient = observed vector - expected vector
    - observation-expectation의 차이만큼 gradient가 나타나고, learning rate를 곱한 것만큼 parameter update가 일어남
</aside>

![PNG 이미지.png](https://user-images.githubusercontent.com/75057952/150629972-abe7bf0c-8e9f-4e40-b5c2-167c7c77badb.png)

![PNG 이미지.png](https://user-images.githubusercontent.com/75057952/150629971-ae53502e-a778-4436-8fa8-16411d03beb6.png)

**Analogy Task with Word2Vec**

- 2D plane 위에 projection
- Man : King ⇒ Woman : ❓
- Vector Composition으로 계산
