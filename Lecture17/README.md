# Lecture17 – Multitask Learning   
## Motivation: Multitask Learning
task와 충분한 양의 dataset이 주어지면 현재로서도 충분한 performance의 model을 얻을 수 있음  
그러나 소수의 pre-trained model를 제외하면 **다시 random하게 처음부터 시작하는 것**이 대부분인 상황  
→ 하지만 pre-trained라는 concept는 굉장히 성공적이었기 때문에 이를 활용하는 것이 중요해짐  
(e.g. CV = pre-trained feature extractor, NLP = word vector representation(W2V, GloVe, BERT))  
→ **일부분이 아닌 model 전체 측면의 sharing을 시도할 수 없는가?**

현재의 NLP에서 이러한 문제에 대처할 때 가장 큰 문제점은 **task 별로 분리된 정도가 크다는 것**  
→ 일반적으로 unsupervised learning를 통해 해결할 수도 있지만 아직 자연어라는 특성 때문에 성공적이지 못했음  

그러나 분명히 Single-task model보다 unified model은 **knowledge transfer** 측면에서 강점을 가짐!  
(e.g. domain adaptation, weight sharing, transfer/zero-shot learning, continual learning)  

## The Natural Language Decathlon: Multitask Learning as Question Answering
### High-level Approach of decalNLP
**decaNLP** 저자인 강의자는 Multi-task learning을 question을 통해서 task를 추론하는 방식으로 실현  
context <img src="https://render.githubusercontent.com/render/math?math=x">, target <img src="https://render.githubusercontent.com/render/math?math=y">로 구성된 model <img src="https://render.githubusercontent.com/render/math?math=y=f(x)">가 아니라 task <img src="https://render.githubusercontent.com/render/math?math=t">도 필요한 <img src="https://render.githubusercontent.com/render/math?math=y=f(x, t)">로 이를 생각  
→ 이 때, <img src="https://render.githubusercontent.com/render/math?math=t"> 대신에 주어지는 question <img src="https://render.githubusercontent.com/render/math?math=q">를 통해서 task <img src="https://render.githubusercontent.com/render/math?math=t">를 연결시키는 것을 학습시키는 것! (i.e. <img src="https://render.githubusercontent.com/render/math?math=y=f(x, t(x, q))">)  

이를 통해서 다음과 같은 10가지 task에 대해 question으로 접근할 수 있도록 유도함
- Question Answering
- Machine Translation
- Summarization
- Natural Language Inference
- Sentiment Classification
- Sementic Role Labeling
- Relation Extraction
- Dialogue
- Semantic Parsing
- Commonsense Reasoning
  
→ 문제는 이런 task를 question <img src="https://render.githubusercontent.com/render/math?math=q">로 추론하는 것을 어떻게 유도할 것인가가 됨!
- task에 대한 직접적인 정보가 주어지지 않으므로 task-specific한 module은 사용 불가능
- inference 시 내부에서 task를 구분하고 조정할 수 있도록 학습 필요
- zero-shot inference/unseen tasks에 대해서도 대응할 수 있도록 training 되어야 함

따라서 decaNLP는 간단하게 보면 다음과 같은 방식으로 inference 하도록 구성됨  
1. context와 question을 입력으로 받음
2. softmax를 통해서 context/question/external vocabulary 중 어떤 것을 보아야 하는지 결정
3. Pointer Switch를 통해서 distribution을 통해 3가지 중 한가지 option 선택
4. 실제 answer로 사용할 word를 하나 결정

### Structure of decaNLP(MQAN)
<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924439-d6e0abc5-834f-4f37-a926-719d41dd0969.JPG" alt="1" width="600px" />
</p>

**Encoder**  

1. Independent Encoding  
단순히 fixed된 word vector를 사용하는 것은 unseen question word/context에 대해서 대응 불가  
따라서 GloVe에 없는 word에 대해서 training 및 실제 inference 시 대처를 위해 embedding도 training 필요  
fixed GloVe vector를 기본적으로 사용하고 unseen word에 대해서는 character n-gram 적용   
linear projection을 통해서 이후 layer의 차원에 맞게 <img src="https://render.githubusercontent.com/render/math?math=d"> dimension으로 만듦   
이렇게 projected representation은 BiLSTM을 거쳐 최종 embedding 결정   

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924444-485d5dbb-09fc-4d37-a090-e989826537cb.JPG" alt="2" width="300px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924446-08201822-8a7d-420c-9e4c-f28701a1f918.JPG" alt="3" width="500px" />
</p>

1. Alignment  
앞 단계에서 얻은 question/context 간의 관련성을 찾기 위해서 attention 수행  
이 때 서로 간에 완전히 관계가 없는 token일 경우를 찾아내기 위해 dummy token 추가  
attention을 계산하는 방식은 dot product로 수행

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924447-f804a80f-6961-41e0-beac-709c1d7c1a7c.JPG" alt="4" width="600px" />
</p>

1. Dual Coattention   
앞 단계에서 얻은 similarity score를 통해서 original input sequence의 각 token의 관련성을 계산  
이를 통해서 question/context에 따라 original seqeunce의 weighted summation을 얻을 수 있음  
각각 마다 얻은 similarity score를 공유하여 한 번 더 question/context에 attention 같이 계산  
이를 통해서 question/context 각각의 중요도에 따라서 다시 summation이 일어남("coattended")  
최종 output에는 Alignment 단계에서 추가된 dummy token의 representation(first column)이 추가되어 있음     
따라서 이는 이후 과정에서 무의미한 것이므로 이 단계에서 drop됨

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924448-f3547da8-663d-4dd8-9c32-3c970524f9d6.JPG" alt="5" width="400px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924450-e61dcb17-deef-4a7f-a343-6709c0626902.JPG" alt="6" width="400px" />
</p>

1. Compression  
Coattention으로 얻은 information을 사용하여 다시 layer 간 차원인 <img src="https://render.githubusercontent.com/render/math?math=d"> dimenstion으로 축소 필요  
따라서 두개의 indepedent BiLSTM을 이용하여 question/context 별로 축소  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924451-d5162231-9116-49b1-8823-3ecabb5376a5.JPG" alt="7" width="400px" />
</p>

1. Self-Attention/Final Encoding   
Transformer 구조를 택하여 Multi-head attention 수행  
Transformer output을 다시 2개의 BiLSTM을 거쳐 최종 encoder의 output으로 사용  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924452-aed89f4a-8858-4c6d-80c7-a3fb436e2133.JPG" alt="8" width="500px" />
</p>

> Encoder 구조에서 전체를 Transformer로 사용하지 않은 이유는 optimization이 쉽지 않았기 때문!  
> 분명히 Transformer가 병렬화에 유리한 건 맞지만 10 task를 모두 cover하기 위한 policy 조절이 쉽지 않았음  
> 10가지 task에 대해서 모두 robust하기 위해서는 어느정도 BiLSTM을 같이 사용할 수 밖에 없었음  

**Decoder**  

1. Answer Representation  
실제 decoder의 input으로 사용하기 전에 linear projection을 거침  
이후 layer가 Transformer 구조를 택하고 있기 때문에 이 때 Positional Encoding도 추가됨

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924453-251e5f1f-685f-4f8f-b9c5-8ceb5116af71.JPG" alt="9" width="150px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924454-45f1d927-358f-4107-b178-3f26606f314c.JPG" alt="10" width="500px" />
</p>

1. Multi-head Decoder Attention  
Encoder와 같이 Transformer 구조를 택한 Self-attention 수행  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924455-6c8963ea-7498-477e-9da6-47fe0a5b4edb.JPG" alt="11" width="400px" />
</p>

1. Intermediate Decoder State  
attention이 적용된 LSTM을 통과하여 앞선 step에서 생성한 word를 previous hidden state로 받게 함  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924456-fb35e753-050d-4883-a4ec-f59c8ca10da0.JPG" alt="12" width="300px" />
</p>

1. Context and Question Attention  
앞 단계의 LSTM의 output과 Encoder의 question/context encoding 간의 attention score 계산

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924457-9a4d01d1-5e05-46bf-beeb-20c9802e4e42.JPG" alt="13" width="500px" />
</p>

1. Recurrent Context State  
계산된 attention score를 기반으로 attention value 계산하여 feed-foward network 통과  
activation은 tanh를 사용하며, question/context마다 독립된 network로 존재  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924458-22d57d1f-1408-4060-922a-6f717cbc8ab2.JPG" alt="14" width="450px" />
</p>

1. Multi-Pointer-Generator  
실제 output word token을 결정하기 위해 question/context 또는 완전히 새로운 vocabulary인지 선택 필요  
따라서 Context and Question Attention 과정에서 얻은 softmax로 context/question distribution 생성  
이 때 새로운 vocabulary가 필요할 때를 위한 <img src="https://render.githubusercontent.com/render/math?math=v"> 종류의 token을 나타내는 새로운 <img src="https://render.githubusercontent.com/render/math?math=v">차원 vector 도입  
이후 이들 간의 switch 역할을 수행할 수 있는 2개의 learnable parameter <img src="https://render.githubusercontent.com/render/math?math=\gamma, \lambda"> 도입  
= Recurrent Context State를 통해 0 ~ 1 사이의 context/question의 중요도를 나타내는 parameter  
→ 이것을 negative log-likelihood loss(<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}=-\sum_t^T log p(a_t)">)를 통해서 학습

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924457-9a4d01d1-5e05-46bf-beeb-20c9802e4e42.JPG" alt="15" width="500px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924460-74ce55b2-8f94-4672-9966-b21d79289233.JPG" alt="16" width="400px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924462-423fc4e4-7e26-4782-9fc3-bdfb5cd00bdf.JPG" alt="17" width="250px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924464-be13ddb4-25c7-46e8-b784-c7d7af3c248e.JPG" alt="18" width="550px" />
</p>

### Optimization Technique: Anti-Curriculum Pre-training
Multitask model은 일반적인 task와 다르기 때문에 training strategy도 구성도 중요함  
→ 가장 간단하게 떠올릴 수 있는 방법은 **Fully-joint** 방식!  
= 각 task마다 mini-batch를 뽑아내고 task 별로 round-robin 방식으로 학습  
= 굉장히 쉽게 구성 할 수 있어서 Multitask라는 복잡한 환경에서도 쉽게 training 가능

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924465-8b91d9f4-0a41-4845-9203-4fe5d5f622a8.JPG" alt="19" width="400px" />
</p>

→ 매우 간단하지만 대부분의 strategy보다 우수한 performance를 보였음!  
→ 그러나 아직 완전히 우수한 수준의 performance를 보여주진 못함  

이에 대해 Neural Network training에서 존재하던 **Curriculum Learning**을 떠올릴 수 있음  
= 짧은 문장을 학습하면서 점점 긴 문장을 학습하는 방식  

그러나 Multitask Learning에서는 이 것을 역순으로 하는 **Anti-Curriculum Learning**이 효과적이었음!  
= 상대적으로 어려운 task부터 pre-training하고 쉬운 task로 fine-tuning  
→ Sentimental Classification과 같은 **쉬운 task에서 local optimum에 빠지는 것을 방지**하는 역할 수행  
→ 어려운 task에서 보다 general한 representation을 학습할 수 있었던 것!

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924468-296cdae4-55d4-4e31-a71d-e0747f2d6a16.JPG" alt="20" width="400px" />
</p>

### Performance of decaNLP
NLP task는 각 task마다의 고유한 metric이 존재함  
따라서 decaNLP는 같은 Architecture를 가지고 한 가지 task에 대해서만 집중했을 때의 결과를 비교군으로 선택  
이렇게 얻은 model 중에서 각 task에서 최고의 performance를 얻은 model을 선택하는 것을 "Oracle" model로 가정  
(i.e. question이 들어왔을 때 각 task가 어떤 task에 속하는지 알고 적절한 model을 골라내는 "Oracle")  
→ 이런 **이론적인 model의 performance와 Multitask model의 performance를 비교**하는 것!

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924470-1a5e8243-5afc-4c47-bdf4-676afac2f5ef.JPG" alt="21" width="550px" />
</p>

performance를 비교하는 과정에서 얻을 수 있는 결론은 다음과 같음
- Transformer 구조는 Single-task/Multitask 모두에서 좋은 performance 향상을 가져옴
- Question Answering과 Sementic Role Labeling는 performance 상 높은 상관관계를 보였음
- question에 pointing 해야 하는 것에 대해서는 좋은 performance를 보임 (i.e. MT = bad performance)
- Multitask Learning은 zero-shot task (i.e. QA-ZRE)에 대해서 performance 향상에 도움을 줌
- Anti-curriculum pre-training은 Fully-joint 방식보다 확실한 performance 향상을 보임  
- "Oracle"의 dacaScore은 '586.1'이므로 완전히 근접한 수준의 performance를 보여주지 못함  

→ 이후 word vector의 교체(CoVe), 부족한 dataset(IWSLT)의 oversampling 등을 거쳐 실험됨  
= 해당 경우에서 "Oracle" score는 618.2였으나 Multitask는 **616.8**까지 향상될 수 있었음!

현재 NLP의 여러가지 task에서 기존에 사용해왔던 metric의 정당성에 대해서 많은 논의가 있음  
(e.g. MST: "BLEU는 human evaluation과 유의미한 상관 관계가 없다")  
→ 그러나 다양한 metric에서 performance를 평가할 수 있다는 점에서 robust한 model이라는 강점이 있음 

또한 실제 Decoder의 dataset 간 Pointer의 distribution을 보면 task에 맞게 적절히 할당되고 있었음  
이는 Multitask Learning이 task 간의 confusion 없이 적절하게 이루어지고 있다는 것을 추론 가능함  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/155924472-683936d5-815c-42e5-8044-b7abbc35400c.JPG" alt="22" width="600px" />
</p>

> 현재 NLP task에서 중요하게 다루어지는 pre-trained BERT idea와는 다른 측면에서 장점을 갖고 있음!  
> 결국 BERT 또한 각 task마다 새로운 token, top-layer, 추가 dataset을 요구하는 **seperated** model임  
> 따라서 아직 BERT는 task-specific한 fine-tuning이 필요하므로 **General NLP model**이라고 하기에는 부족함

#
*Reference: https://youtu.be/M8dsZsEtEsg*  
*Writer: Sangkyu Lee*