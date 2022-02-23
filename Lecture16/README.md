# Lecture 16

### Coreference Resolution

---

### Overview : Coreference Resolution

|Index|Subtitle|
|--- | --- |
|16.1.| Coreference Resolution Task |
|16.2.| Mention Detection |
|16.3.| Linguistics of Reference |
|16.4.| Coreference Models |
|16.5.| Evaluation |

---

### 16.1. Coreference Resolution Task

> **Coreference Resolution의 개념**

Barack Obama nominated Hillary Rodham Clinton as his secretary of state on Monday. He chose her because she had foreign affairs experience as a former First Lady.

- 이러한 텍스트가 주어졌을 때, 텍스트 내에는 "he, his, she"와 같은 대명사 그리고 대명사와 대명사가 실제로 지칭하는 대상 사이의 연결 관계가 존재합니다.
- Coreference Resolution은 이렇게 서로 다른 두 단어가 *같은 것을 지칭*한다는 관계를 밝혀내는 것을 의미합니다.
- 이때 이러한 관계를 Mention이라고 하고, Coreference Resolution = Identify all mentions that refer to the same entity라고 정리할 수 있습니다.

- **Applications**
    - Full text understanding
        - 정보를 추출하여 요약하거나, 질문에 대답하는 등의 Task에 활용할 수 있습니다. 
        - 예를 들어 QA Task에서 질문이 "그는 1961년에 태어났다"라면, Coreference Resolution을 통해 '그'가 무엇을 지칭하는지 파악하고 그 대상이 태어난 년도가 담긴 문장을 찾아낼 수도 있을 것입니다.
    - Machine Translation
        - 기계 번역 시에 누락된 정보가 있을 경우 번역이 어색할 수 있는데, Coreference Resolution을 통해 숨겨진 관계를 찾아내고 적절한 번역을 제시할 수 있습니다.
        - 예를 들어, he와 사람을 대응시켜서 누가 누구를 좋아하는 것인지 의미적으로 분명히 해석해 낼 수 있습니다.
    - Dialogue System
        - 챗봇 같은 대화 시스템에서도 어떤 대명사가 무엇을 지칭하는지 파악하는 것은 굉장히 중요합니다.

### 16.2. Mention Detection

> **2-Step Pipeline for Coreference Resolution** : 1st step = Mention Detection


- 초기 Coreference Resolution Task에 대한 접근은 두 단계 파이프라인으로 구성되어 있었습니다.
- Mention을 Detect하고, Clustering을 통해서 같은 대상을 지칭하는 것끼리 묶어내는 두 단계입니다.
    - Mention을 Detect하는 것은 비교적 쉽습니다(대명사는 한정되어 있고, 대응되는 명사 역시 찾기 쉬운 편입니다).
    - 반면에 Mention을 클러스터링하는 과정은 어려운 편입니다.
    <br/>
    <p align = "center">
    <img src = "https://user-images.githubusercontent.com/75057952/155056606-87841a81-1dc8-4821-99fd-b524e8613750.jpg" width = "500dp">
    </p>
    
    - Mention Detection의 2-Step 파이프라인입니다.
    - 기존의 2-step 파이프라인 중 POS Tagger, NER system, Constituency parser를 활용해 각각의 mention을 detect하는 과정 대신 Classifier를 Train하는 접근이 가능합니다.
    - 하지만, 최근에는 2-Step 파이프라인을 하나의 E2E Network로 대체하는 접근이 주를 이루고 있습니다.

### 16.3. Linguistics of Reference

> **Anaphora vs. Coreference vs. Cataphora**

- Anaphora는 개념적으로 독자적으로 의미를 참조하여 설명할 수 없는 경우를 의미합니다. 특히 뒤의 단어가 '전방 조응'을 통해 앞선 단어를 참조하면 Anaphora입니다.
- Coreference는 같은 대상을 가리키는 관계를 의미합니다. 
- Cataphora는 Anaphora와 역의 관계로, 앞의 단어가 뒤의 단어를 참조하면 Cataphora가 됩니다.
- 셋의 관계는 아래와 같습니다.
<br/>
    <p align = "center">
    <img src = "https://user-images.githubusercontent.com/75057952/155056608-f49074de-048d-4f5e-b238-463c799258d4.jpg" width = "700dp">
    </p>
<br/>
    <p align = "center">
    <img src = "https://user-images.githubusercontent.com/75057952/155056609-a39a3ced-4bb7-4e3f-85ed-45b87323aeb5.jpg" width = "600dp">
    </p>

- 위 벤다이어그램을 보면, Anaphora는 Briding Anaphora와 Pronomial Anaphora로 구분할 수 있습니다.

### 16.4. Coreference Models

- Coreference Model의 종류에는 네 가지가 있습니다.
    |4 Coreference Models|
    |---|
    |Rule-based|
    |Mention Pair|
    |Mention Ranking|
    |Clustering

> **Rule-based Model** : Hobb's naive algorithm

- Hobb's naive 알고리즘은 9단계로 구성되어 재귀적으로 탐색하는 알고리즘입니다.

<br/>
    <p align = "center">
    <img src = "https://user-images.githubusercontent.com/75057952/155056611-ddd43baf-8c5d-4e5f-8bfd-bcbc5fb7fd6f.jpg" width = "800dp">
    </p>
<br/>
    <p align = "center">
    <img src = "https://user-images.githubusercontent.com/75057952/155056613-c40ec6bb-fe71-4cb8-ade6-b26f51d4de02.jpg" width = "500dp">
    </p>

- Hobb's naive 알고리즘은 2010년 이전까지 Coreference Model로 나름 괜찮은 성능을 보였습니다.
- 하지만 결정적으로 Semantic-based 문장의 경우에는 전혀 정답을 찾아내지 못했습니다. 문장에 모든 정보가 내재된 것이 아니라, coreference를 알아내기 위해서는 real-world knowledge가 필요했기 때문입니다.
- 그래서 이 이후에는 Statistical, Neural 모델이 등장하게 되었습니다.

> **Mention Pair Model**

- Metnion Pair Model은 Candidate Mention에 대해 해당 Mention이 Coreference가 있는지, 없는지를 판별하는 분류기를 학습하는 방법입니다.
- Objective Function으로는 Cross-Entropy Loss를 활용하고 있습니다.
<br/>
    <p align = "center">
    <img src = "https://user-images.githubusercontent.com/75057952/155056615-0260cb5e-1073-4805-bf43-29f6a3291da5.jpg" width = "800dp">
    </p>
- Transitivity는 coreference link의 중요한 성질입니다.
- Mention Pair Model은 모든 Candidate에 대해서 예측을 수행하는데, 이것은 사실 지나치게 큰 클러스터를 만들 가능성이 있습니다.
- 예를 들어 굉장히 먼 위치의 두 단어는 확률적으로 Coreference일 가능성이 적지만, 우연히 두 단어를 연결하도록 분류했다면 Coreference 관계가 하나의 클러스터로 뭉칠 수 있습니다.
- 따라서 이 문제는 '최초'의 positive mention이 등장할 때까지만 가까운 것부터 탐색하는 방법론을 통해 개선할 수 있습니다.
- 아래의 Mention Ranking Model은 Score를 매겨서 Ranking을 구함으로써 위의 문제를 조금 더 합리적으로 해결할 수 있습니다.


> **Mention Ranking Model**
- softmax된 확률을 통해 score를 구하고, -log()를 취해 likelihood를 loss function으로 바꾸어 사용하고 있습니다.

<br/>
    <p align = "center">
    <img src = "https://user-images.githubusercontent.com/75057952/155056616-ffcf62b8-9a10-425a-acd7-5b07d29c1291.jpg" width = "500dp">
    </p>

- 위의 Mention Pair Model에서도 p(mi, mj)를 계산할 수 있다고 가정하고 있는데, 여기서도 마찬가지입니다.
- 따라서 이 probability를 어떻게 계산할 것인지가 굉장히 중요한 문제가 되었습니다.
- 확률 계산을 위해서는 크게 세 가지 방법을 활용할 수 있습니다.
    - Non-neural    statistical classifier
    <br/>
    <p align = "center">
    <img src = "https://user-images.githubusercontent.com/75057952/155056617-9710d6a5-3d91-4782-a645-ee67b001c22b.jpg" width = "350dp">
    </p>

    - Simple NN

    <br/>
    <p align = "center">
    <img src = "https://user-images.githubusercontent.com/75057952/155056596-399b8687-e623-40d0-b9ac-92c8f1fe9943.jpg" width = "500dp">
    </p>

    - More advanced model using LSTMs, attention

    <br/>
    <p align = "center">
    <img src = "https://user-images.githubusercontent.com/75057952/155056600-11fd0698-9586-41de-a116-6b18666bd5a4.jpg" width = "800dp">
    </p>

> **Clustering Model**

- Clustering 모델은 Coreference를 Clustering Task로 상정하고 군집화를 수행하는 아이디어이다.
- **Hierarchical Clustering**(계층적 군집화) 방식을 사용하며, 각각의 mention이 독자적인 cluster를 가지고 있는 상태에서 두 개의 cluster를 merge하면서 군집화한다.
- 두 개의 클러스터를 병합하는 것이 '좋은 군집화'인지 score를 매길 수 있도록 모델을 구성한다.
- 어떠한 s(ci, cj) = S를 상정하여 S score 값이 특정 threshold를 넘기면 Merge, 그렇지 않으면 Merge하지 않는 방식을 채택한다.
- Mention Pair를 Representation으로 바꾸고, Cluster-Pair Representation으로 변환하여 Score를 매기게 된다.
- 이때 각각의 Mention Pair Representation을 생성하는 인코더로는 Mention-Pair Model을 활용할 수 있다.
<br/>
    <p align = "center">
    <img src = "https://user-images.githubusercontent.com/75057952/155056596-399b8687-e623-40d0-b9ac-92c8f1fe9943.jpg" width = "500dp">
    </p>

- Mention-Pair를 Pooling Operation을 통해 처리하여 Cluster-pair representation을 생성하고, 다시 Linear Transform하여 최종 score를 산출하게 된다.
<br/>
    <p align = "center">
    <img src = "https://user-images.githubusercontent.com/75057952/155056601-a9649f62-31b3-4b4d-9816-724af654d4ef.jpg" width = "600dp">
    </p>
- Merge하는 Cluster pair는 이전 step에 이미 만들어진 cluster에 의존적이므로, 일반적인 Supervised Learning 대신 Reinforment Learning을 사용한다.
- Merge 결과를 바탕으로 Coreference Evaluation을 수행하고, 그 Metric의 변화를 Reward로 활용하는 방식이다.

### 16.5. Evaluation

<br/>
    <p align = "center">
    <img src = "https://user-images.githubusercontent.com/75057952/155056602-081b014e-f972-4496-a6bc-9c66ebe8e02e.jpg" width = "600dp">
    </p>
