# Lecture10 - Question Answering
## Motivation: Question Answering  
우리는 때때로 full-text document(e.g. Web)에서 간단히 필요한 document만을 찾고 싶을 수 있음  
특정한 **질문(question)** 에 대한 **정답(answer)** 을 필요로 할 때가 잦음  
이러한 task에 대해서 접근하는 방식을 두 분야로 나눌 수 있음  
>1. 정답을 포함하고 있는 document 자체를 찾아내기(= **Information Retrival**)
>2. 주어진 document에서 정답을 찾아내기(= **Reading Comprehension**)  
  → 이번 강의에서는 **2번에 해당하는 내용**을 주로 다루게 됨!

### Machine Comprehension: Challenging Problem for AI
NLP 분야에서 Reading Comprehension 자체는 굉장히 오래 전부터 다루어져왔던 주제임  
→ "text를 이해하는게 맞다면, text와 question이 주어지면 machine은 적절한 string을 내놓을 수 있어야 한다!"  

이런 Machine Comprehension을 test하기 위한 대표적인 dataset이 **'MCTest datset'** 이었음  
MCTest는 Passage(P), Question(Q)가 주어지면 Answer(A)로 이루어진 dataset  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153256695-812c6735-2abd-48e5-8a7f-a3d41632b3f1.JPG" alt="1" width="400px" />
</p>

Traditional NLP model은 굉장히 복잡했지만 Factoid question answering에서는 좋은 performance를 보였었음  
(i.e. specific entity와 관계를 맺는 사실에 대한 question일 때의 answering, Web 상에서 사용하기에는 충분한 수준)  
그러나 다른 question에 대한 개선을 위해 Neural Network model을 시도하기에는 dataset 자체가 부족했음  
→ **SQuAD**와 같은 dataset이 등장하면서 적극적으로 Neral Network model의 적용이 가능해짐  

## Standford Quenstion Answering Dataset(SQuAD)  

SQuAD는 일종의 "extractive question answering" dataset이었음
>1. Wikipedia를 통해서 생성
>2. 약 100k개의 example 
>3. question에 대한 answer는 passage의 subsequence임을 만족함("passage가 answer를 span")  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153256701-2b17adcb-84c6-47d5-9730-6e42c6412d28.JPG" alt="2" width="400px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153256705-2caaa09b-2e71-4835-831b-2a56452fb476.JPG" alt="3" width="400px" />
</p>


이렇게 생성한 question에 대해서 다음과 같이 performance를 평가했음
- 서로 다른 3명으로부터 question에 대한 answer를 sample  
- 두 종류의 평가 수단으로 평가  
  1. 3개의 답 중 하나가 일치하면 맞은 것으로 평가(Exact match)  
  2. 각 question마다 F1 score 계산 후 전체에서의 average F1 score 사용  
    → 사람들의 span이 일치하지 않는 경우에 조금 더 robust하다는 점에서 F1 score가 조금 더 우선됨
- 평가 시에는 punctuation과 articles은 제외하고 계산됨  

그러나 SQuAD는 모든 question에 대한 answer가 passage에 반드시 존재한다는 점에서 문제  
단순히 question에 대응되는 answer 후보 중 가장 적절한 것을 선택하는 것처럼 될 수 있음  
→ question에 대한 answer를 찾는 것이 아니라 가장 answer다운 것을 골라내는 것("ranking")

SQuAD 2.0에서는 train의 1/3, test의 1/2가 'No Answer'인 question으로 구성되게 바뀜  
→ 특정한 threshold를 사용하거나 다른 component(NLI, Answer validation)를 추가해서 'No Answer'를 답해야 함  

그러나 dataset의 개선에도 아직 Pattern Matching처럼 기능하는 것처럼 보이기도 함!  
(F1 score 상으로는 인간의 수준으로 올라와도 여전히 인간처럼 Comprehension한다고 보기는 어려움)

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153256710-dc4353d0-679e-43a3-8385-a8670dba171e.JPG" alt="4" width="400px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153256732-a82b452d-95f1-405c-9f6c-0b9e696d4902.JPG" alt="5" width="400px" />
</p>

→ 'No Answer'인 question에 대해서 유사한 단어(kill ↔ destroy)를 바탕으로 answer를 찾아냄  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153256741-c9bcfb73-fce7-45ad-af18-9fec3e62a90b.JPG" alt="6" width="400px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153256757-21cd0477-0fd4-48b9-a89f-df98eee33e81.JPG" alt="7" width="400px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153256760-0e13633e-f013-4c8a-ae9a-80879da48263.JPG" alt="8" width="400px" />
</p>

따라서 여전히 SQuAD dataset은 한계가 존재하는 dataset임
- span-based answers 형태의 question만 존재해 다양한 형태의 question에 대응 불가
- 오직 passages에서만 answer를 찾으므로 다양한 documentation 간의 참조가 없음  
  → 다양한 documentation에서 실제 answer에 가까운 answer를 찾아내는 것이 불가능  
  → syntactically stable한 passage에 대해서만 최적화되어 그렇지 않은 real-field에서 matching이 어려움
- 지시어(he, she) 외에 여러 문장을 동시에 참조해야하는 추론 문제가 없음  

**→ 그럼에도 불구하고, SQuAD는 Question Answering system의 starting point로 유용하게 쓰임!**

## Question Answering Models
### Stanford Attentive Reader
**Standford Attentive Reader**는 QA system에서 굉장히 simple한 Neural Network 구조로 좋은 performance를 보임  
LSTM을 Bidirectional하게 구성한 후 Question을 query로 보고 passage에 Attention을 적용한 구조!

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153256769-99ffcf47-2032-496d-946e-b9acad2ea869.jpg" alt="9" width="400px" />
</p>

>1. pre-trained GloVe를 통해 question과 passage를 embedding
>2. sigle-layer BiLSTM에 question을 통과
>3. 양방향의 end-point의 encoding을 concatenate해서 question vector 생성
>4. passage를 마찬가지로 BiLSTM으로 통과 
>5. 각 word의 encoding마다 concatenate해서 문장 길이 만큼의 passage vector 생성  
>6. start token과 end token 각각을 predict하기 위해 두 개의 weight matrix(<img src="https://render.githubusercontent.com/render/math?math=\mathbf{W}_s,\mathbf{W}^{\prime}_s">) 정의
>7. 각 weight matrix를 사용하여 softmax를 사용하여 attention weight 계산
>8. 계산된 attention weight를 기반으로 attention value 계산 후 prediction 수행

→ 당시 다른 model에 비해 simple했음에도 불구하고 performance 상 상당히 우수했음!

이후에 'Stanford Attentive Reader++(DrQA)'로 개선되면서 몇가지 특징이 추가됨
1. BiLSTM을 3-layer까지 stack했음
2. question vector 생성 시에 end-point만을 사용하는 것이 아니라 전체 output의 weighted sum 사용  
3. embbeding 된 GloVe vector 외에 다른 관련 feature도 같이 concatenate 후 input으로 사용  
  (e.g. POS tags, frequency, exact match on question, ···) 

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153256772-073023e2-0e6d-40aa-9099-b59834ff1817.jpg" alt="10" width="400px" />
</p>

이런 Neural Network model은 기존의 traditional한 model보다 우수한 performance를 보였음!  
→ 특히 일종의 **semantical matching**에 대해서 더 나은 performance를 보였음  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153256775-0f9cb4dc-1dc9-4da8-86a1-9bc3a593a88f.jpg" alt="11" width="400px" />
</p>

### BiDAF: Bi-Directional Attention Flow for Machine Comprehension
'Stanford Attentive Reader' 이후 attention을 **양방향으로 적용**하는 **BiDAF**와 그 variants가 등장  
question만을 query로 보는게 아니라 passage 또한 query로 활용하려는 노력이었음    
전체적인 concept는 **Query2Context**와 **Context2Query**로 구성되는 **Attention Flow layer**를 이해하는 것이 중요!  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153256777-095523e9-2450-4f40-90d8-4bbd29e77e35.jpg" alt="12" width="400px" />
</p>

양방향으로 attention을 계산해내기 위해서, similarity matrix <img src="https://render.githubusercontent.com/render/math?math=\mathbf{S}\in\mathbb{R}^{T\times J}">를 도입함  
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{H}\in\mathbb{R}^{2d \times T}"> = context matrix, <img src="https://render.githubusercontent.com/render/math?math=\mathbf{U}\in\mathbb{R}^{2d \times J}"> = query matrix   
similarity matrix의 <img src="https://render.githubusercontent.com/render/math?math=(t,j)"> component <img src="https://render.githubusercontent.com/render/math?math=\mathbf{S}_{tj}">는 <img src="https://render.githubusercontent.com/render/math?math=t">번째 context word와 <img src="https://render.githubusercontent.com/render/math?math=j">번째 query word의 유사도를 나타내도록 유도  
즉, 어떤 trainable scalar function <img src="https://render.githubusercontent.com/render/math?math=\alpha">를 통해서 context, query 간의 similarity를 계산해낼 수 있는 임의의 matrix  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{S}_{tj}=\alpha(\mathbf{h}, \mathbf{u})\in \mathbb{R}" height = "20px"> 
</p>  

→ 해당 <img src="https://render.githubusercontent.com/render/math?math=\alpha">로 저자는 
<img src="https://render.githubusercontent.com/render/math?math=\alpha(\mathbf{h}, \mathbf{u})=\mathbf{w}^T_{sim} \cdot [\mathbf{h} : \mathbf{u} : \mathbf{h} \circ \mathbf{u}]">로 선택한 것!(<img src="https://render.githubusercontent.com/render/math?math=\mathbf{w}_{sim}">: trainable weight parameter)  

**Context2Query**  
similarity matrix의 <img src="https://render.githubusercontent.com/render/math?math=t"> row에 대해 softmax 계산(<img src="https://render.githubusercontent.com/render/math?math=\mathbf{a}_{t}">)  
= **<img src="https://render.githubusercontent.com/render/math?math=t"> context word에 대해서 모든 query word들의 attention weight 계산하는 것과 같은 의미**  
→ 이를 각 <img src="https://render.githubusercontent.com/render/math?math=t">에 대해서 반복하여 query vector와 결합하면 각 <img src="https://render.githubusercontent.com/render/math?math=t"> context word에 대한 query vector의 attention value가 됨  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153256780-3be6c14e-7b04-4529-bf10-cf6b94a0bf30.jpg" alt="13" width="200px" />
</p>

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{a}_{t} = \text{softmax}(S_{t:}) \in \mathbb{R}^{J}" height = "20px"> 
</p>  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\tilde{\mathbf{U}_{:t}}=\sum_{j}\mathbf{a}_{tj}\mathbf{U}_{:j}\in \mathbb{R}^{2d}" height = "30px"> 
</p>  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\therefore \tilde{\mathbf{U}}\in \mathbb{R}^{2d \times T}" height = "20px"> 
</p>  

**Query2Context**   
similarity matrix의 각 row마다 가장 큰 값을 선택하고 추출된 최종 column vector에 대해서 softmax 계산(<img src="https://render.githubusercontent.com/render/math?math=\mathbf{b}">)    
→ 각 context word마다 가장 similarity가 큰 query word와의 similarity value가 뽑힘  
**= query들과 관계가 없는 context word였으면 하나라도 관계가 있는 word들에 비해 매우 작은 값!**  
= query word들과 관계가 있으면 있을 수록 해당 context word는 높은 값을 갖게 됨  
= **전체 query word에 대해서 모든 context word에 대한 attention weight를 계산하는 것과 같은 의미**  
→ 이를 context vector와 결합하면 query word들에 대한 각 <img src="https://render.githubusercontent.com/render/math?math=t"> context vector의 attention value가 됨  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153256782-5165c995-7c97-4225-bf76-77ddbfd585b7.jpg" alt="14" width="200px" />
</p>

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{b} = \text{softmax}(\text{max}_{col}(\mathbf{S})) \in \mathbb{R}^{T}" height = "20px"> 
</p>  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\tilde{\mathbf{h}}=\sum_{t}\mathbf{b}_{t}\mathbf{H}_{:t}\in \mathbb{R}^{2d}" height = "30px"> 
</p>  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\therefore \tilde{\mathbf{H}}\in \mathbb{R}^{2d \times T}" height = "20px"> 
</p>  

이렇게 구한 양방향에서의 attention weight를 context vector와 함께 임의의 함수 <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\beta}">를 통해 input <img src="https://render.githubusercontent.com/render/math?math=\mathbf{G}">로 활용  
이 <img src="https://render.githubusercontent.com/render/math?math=\mathbf{G}">를 <img src="https://render.githubusercontent.com/render/math?math=t"> column에서 보면 <img src="https://render.githubusercontent.com/render/math?math=t"> context vector와 양방향의 attention weight를 표현하게 됨  
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{G}_{:t} = \mathbf{\beta}(H_{:t}, \tilde{U}_{:t}, \tilde{H}_{:t})" height = "25px"> 
</p>   

이런 과정(Q2C, C2Q)에서 각각의 query word를 contexct vector를 모두 온전히 사용  
= **하나의 fixed size vector로 compression하는 과정이 생략되어 있음!**  
(e.g. Stanford Attentive Reader: question vector로 question을 fixed size vector로 encoding)  
→ compression 과정에서 발생하는 **정보 손실이 없음** (원문에서 언급하는 contribution 중 하나)  


이 때 원문에서는 아래와 같이 concatenation하여 사용하였으며 이를 two-layer BiLSTM의 input으로 사용  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\beta(\mathbf{h}, \mathbf{\tilde{u}}, \mathbf{\tilde{h}})=[\mathbf{h} : \mathbf{\tilde{u}} : \mathbf{h} \circ \mathbf{\tilde{u}} : \mathbf{h} \circ \mathbf{\tilde{h}}] \in \mathbb{R}^{8d \times T}" height = "25px"> 
</p>

> concatenate 시에 hadamard product를 더해서 concatenate하는 이유는 엄밀한 근거가 있는 것은 아님  
> Neural Network가 matching을 'good behaviour'로 학습하기 하도록 하는 유도해보는 'belief'에 가까움

## Further Works; How to Calculate Attention?
지금까지의 강의에서도 다양한 방법으로 attention을 계산하는 방법이 등장  
Google에 따르면 shallow net을 추가한 형태로 계산하는 attention이 조금 더 효과적이었음!  
그러나 Microsoft('FusionNet')은 더 많은 attention을 계산하기 위해서는 계산량을 줄여야 한다고도 주장함  
→ lower rank factorization을 통해 matrix multiplication을 줄이기, 비선형 함수 사용 등 다양하게 적용 가능

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153256784-41bfcbcb-397e-49b0-962c-0081490d421a.jpg" alt="15" width="400px" />
</p>

단순히 attention을 계산하는 방법 뿐만 아니라 복잡한 형태로 attention을 적용하는 것도 연구되었음! (2016 ~ 2018)

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153256787-bd52a088-0d61-4e4e-ad35-959aa26f5fbd.jpg" alt="16" width="400px" />
</p>

→ 그러나 최근에는 traditional word vector가 아닌 contextual word representation을 통한 attention model이 주류!  
(e.g. ELMo, BERT: language model을 통한 contextual embedding, Transformer: Self-Attention)

#
*Reference: https://youtu.be/yIdF-17HwSk*  
*Writer: Sangkyu Lee*