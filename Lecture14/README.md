# Lecture14 - Transformers and Self-Attention   
## Why We Need Self-Attention?
### Motivation: Self-Attention  
RNN 및 그 variant들은 Sequential Model의 primary workhorse로 존재해왔음   
그러나 여전히 몇가지 한계를 가지고 있기 때문에 개선의 여지가 필요함   
>1. Sequential하게 계산되는 구조는 병렬화를 어렵게 만듦  
>→ 전 단계의 time step의 계산에 dependecy가 발생하기 때문
>2. long/short-term effect에 대한 명시적인 관계를 유도하기 어려움    
>→ fixed-sized vector로 encoding되어 long/short-term effect를 동시에 구분하여 반영하기 어려움
>3. 언어에서 자주 관찰되는 hierarchy를 명시적으로 유도하기 어려움

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553948-eb88be2e-7c28-40c2-b0d1-8b4f098a4a06.JPG" alt="1" width="350px" />
</p>

이런 문제를 대처하는 한 가지 방안은 **CNN**을 같이 사용하여 그 장점을 가져오는 것임    
>1. 병렬화가 쉬움  
>→ 오직 layer 간의 dependecy만 존재하기 때문
>2. 명시적인 local dependecy를 반영할 수 있음  
>→ kernel을 통해 계산되는 local receptive field를 명시적으로 계산해낼 수 있음  
>3. 다양한 길이의 dependecy를 계산해낼 수 있음  
>→ dilation convolution을 통해서 linear만이 아닌 다양한 간격의 covolution 계산 가능 

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553950-18639b0a-ac5b-4721-827e-874cde033801.JPG" alt="2" width="350px" />
</p>

그러나 long-distance receptive field를 얻기 위해서는 **많은 layer의 stack이 요구됨!**  

DL model의 구조 자체는 결국 어떤 representation을 잘 catch 해내기 위한 것들임    
(e.g. RNN: directional context dependecy, CNN: local neighborhood dependecy → "inductive bias")  
따라서 task에 필요한 representation을 잘 catch해낼 수 있는 다른 구조가 있다면 이를 활용해도 문제가 없음  
**→ 그렇다면 NMT의 encoding-decoding을 위해 사용되던 attention을 representation으로 활용할 수 없을까?**  
= Self-Attention의 Motivation!

### Self-Attention is Self Re-Expression   
attention은 concecpt적으로 Encoder/Decoder output 사이의 관련성을 찾아내기 위한 시도  
그렇다면 input으로 주어지는 word embedding들의 관련성을 찾아내기 위해서도 응용이 가능  
**input word들 사이의 attention을 반복하여 문장 내에서의 각 word의 representation을 찾아내게 만드는 것!** 

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553951-acb98909-98f7-4cea-8489-8e84ed60e1ad.JPG" alt="3" width="350px" />
</p>

각 word마다 entire neighborhood에 대해서 attention 수행  
= neighborhood를 통해 weighted combination으로 표현되는 새로운 vector 생성   
= 각 word의 neighborhood를 반영하는 summurization를 통해 새로운 representation 획득  

> Self-Attention을 representation으로서 사용하는 근거를 **Message Passing**에서도 찾을 수도 있음  
> 특히 Transformer에 사용되는 Multi-Head Attention과 **Multiple Towers**는 concept적으로 유사

<p align="center">
<img src="https://user-images.githubusercontent.com/86907286/153553953-d7ac38d6-71e6-4786-b6e2-286f6ed2ea48.JPG" alt="4" width="350px" />
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/86907286/153553954-762104a0-6d72-4104-a2e1-9efb9f78e7c2.JPG" alt="5" width="250px" />
</p>

이렇게 Self-Attention을 사용하는 것은 RNN에 비교하면 또 다른 장점을 갖게 됨  
>1. 각 word 간의 "path length"는 모두 일정한 constanct가 됨  
>→ attention을 계산할 때는 RNN과 같이 time-interval 없이 직접 similarity를 계산하기 때문  
>2. 구조 상 자연스럽게 gating/multiplicative interaction이 발생함  
>→ softmax와 attention weight를 통해서 중요한 activation이 선택됨    
>3. 병렬화가 간단함  
>→ attention 계산 시에 matrix multiplication을 사용한다면 자연스럽게 유도되는 성질

그렇다면 Self-Attention이 **sequential data에 효과적으로 적용될 수 있는가?**  
→ RNN과 결합하여 이를 응용해보려는 여러 시도가 있었음  
→ 그러나 해당 접근으로는 경험적으로 NMT나 Text Generation에서 아주 효과적이지는 않았음  

### When Self-Attention is attractive? 
attention을 matrix로 계산하면 두 번의 matrix multiplication이 발생하므로 **length에 대해 quadratic**  
그러나 CNN/RNN은 **dimention에 대해서 quadratic**  
(e.g. CNN = 각 input vector를 flatten하여 linear transform하는 것과 연산 횟수 상 같아 dimension의 제곱)

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553956-2b6c8849-bc3c-401c-b3ef-3716c8870587.JPG" alt="6" width="450px" />
</p>

따라서 Self-Attention이 우위가 발생할 때는 **input dimension이 length보다 dominant한 경우**임  

## Transformer: Attention is All You Need
### Architecure of Transformer
**Transfomer**는 NMT task를 위해 제안된 모델이라 seq2seq와 같이 Encoder/Decoder 구조로 이루어짐  
각 Encoder/Decoder는 Multi-Head Attention과 feed-foward network를 묶은 block이 여러번 stack된 구조  
CNN/RNN Architecture 없이 **오직 Self-Attention만을 통해서 Encoder/Decoder를 구현**함!

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553958-849b7c8a-f46b-4773-a95c-21cc5ece9ab5.JPG" alt="7" width="350px" />
</p>

**Encoder**
1. word embedding vector에 **Postional Encoding**을 더해 postional information 추가    
2. 앞선 결과물로 **Multi-Head Attention**을 통해 **Scaled Dot-Product Attention** 계산  
3. Multi-Head Attention을 거치기 전 input에서 온 **Residual Connection**과 addition
4. Residual Connection이 더해진 후 **Layer Normarlization** 수행
5. Multi-Head Attention의 결과물로 feed-foward network에서 다시 representation 학습  
6. feed-foward network의 결과물도 마찬가지로 Residual Connection에 대해 addition  
7. Residual Connection이 더해진 후 Layer Normarlization 수행
8. 2~7의 과정을 layer를 stack하는 방법을 통해 원하는 수만큼 반복
9. 최종 output을 Decoder의 **2번째 Multi-Head Attention 시의 Key, Value**로 사용

**Decoder**
1. word embedding vector에 Postional Encoding을 더해 postional information 추가  
2. 앞선 결과물로 Multi-Head Attention을 통해 Scaled Dot-Product Attention 계산 
3. attention 계산 중에 **Look-ahead Masking**을 attention score를 Masking
4. Multi-Head Attention을 거치기 전 input에서 온 Residual Connection과 addition
5. Residual Connection이 더해진 후 Layer Normarlization 수행
6. **앞선 결과물을 Query**, Encoder의 output을 Key, Value로 보고 다시 Multi-Head Attention 수행
7. Multi-Head Attention을 거치기 전 input에서 온 Residual Connection과 addition
8. Residual Connection이 더해진 후 Layer Normarlization 수행  
9. Multi-Head Attention의 결과물로 feed-foward network에서 다시 representation 학습
10. feed-foward network의 결과물도 마찬가지로 Residual Connection에 대해 addition
11. Residual Connection이 더해진 후 Layer Normarlization 수행
12. 2~10의 과정을 layer를 stack하는 방법을 통해 원하는 수만큼 반복
13. 마지막 output을 통해 최종 feed-foward network로 output probability 게산 


### Main Concepts of Transformer
**Positional Encoding**  

Self-Attention을 seqential data에 바로 사용될 수 없는 이유는 **attention 자체는 position과 관련이 없기 때문!**  
순수하게 attention 자체를 계산 시에는 각 word의 position에 대한 정보가 전혀 반영되지 않음  
(i.e. 다른 word와의 similarity를 찾을 뿐 해당 word가 어떤 position에서 왔는지는 전혀 반영되지 않음)  
Transformer는 position에 대한 encoding을 추가하여 Self-Attention만으로 Encoder/Decoder 구조를 유도!  

따라서 Transformer는 position 정보를 word vector와 동시에 주기 위해서 추가적인 embedding을 사용함  
저자들은 이런 Positional Encoding으로 sinusoid를 통해서 각 position 별로 고유한 값을 가지게 유도함

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})" height = "28px"> 
</p>
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=PE_{(pos,2i %2B 1)}=cos(pos/10000^{2i/d_{model}})" height = "28px"> 
</p>

이렇게 sinusoid를 통해서 position을 할당하는 방식을 택한 이유는 **쉽게 relative distance를 찾을 수 있기 때문**  
sinusoid를 사용하면 position이 각도가 되어 linear transform <img src="https://render.githubusercontent.com/render/math?math=T">로 <img src="https://render.githubusercontent.com/render/math?math=\theta">와 <img src="https://render.githubusercontent.com/render/math?math=\theta %2B \phi"> position의 관계를 표현 가능   
이러한 <img src="https://render.githubusercontent.com/render/math?math=T">은 **rotation matrix**라는 것을 삼각함수의 덧셈정리로 유도 가능   

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553959-7b3e72bd-ab82-4789-9a5d-e1e43f389930.jpg" alt="8" width="400px" />
</p>

따라서 relative distance에 대해 대칭적으로 positional encoding이 적용될 수 있음  
→ 저자들은 이런 linear transform으로 관계가 표현될 수 있는 embedding이면 학습이 쉬울 것이라 생각하고 사용

**Scaled Dot-Product Attention**  

Transformer는 attention을 계산하기 위해 **Scaled Dot-Product Attention**을 사용함  
기본적인 attention을 위해서 dot product를 사용하면서 추가적인 scaling을 추가한 형태  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553961-d1b43979-1d09-4310-a41c-b2bda821d0e1.JPG" alt="9" width="250px" />
</p>

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T)}{\sqrt{d_k}})V" height = "50px"> 
</p>

key의 dimension인 <img src="https://render.githubusercontent.com/render/math?math=d_k"> factor로 scaling 해주는 이유는 softmax를 사용하기 때문임  
만약 <img src="https://render.githubusercontent.com/render/math?math=d_k">가 크지 않다면 dot product의 크기가 크지 않기에 scaling 하지 않는 것이 performance 상 유리함  
그러나 <img src="https://render.githubusercontent.com/render/math?math=d_k">가 커진다면  sigmoid를 통과할 때 gradient가 매우 작아지는 region이 발생  
→ 이를 회피하기 위해서 저자들은 <img src="https://render.githubusercontent.com/render/math?math=d_k">를 통해서 scaling하는 과정을 추가했음

이 때 attention을 위해 input vector를 실제 Query, Key, Value로 그대로 사용하지는 않음  
해당 vector의 Query, Key, Value로서의 적절한 representation을 projection하는 trainable weight matrix 사용   
예를 들어 Self-Attention으로 사용된다면 embedding vector들을 <img src="https://render.githubusercontent.com/render/math?math=E">로 표현할 때 다음과 같이 나타낼 수 있음  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=Q = EW^Q,K=EW^K, V=EW^V" height = "22px"> 
</p>

**Multi-Head Attention**   

CNN은 각 filter가 **특정한 linear transform을 학습**하여 서로 다른 representation에 특화되도록 유도될 수 있음  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553962-974205f5-a5f4-4fb0-86b5-1f2584054639.JPG" alt="10" width="400px" />
</p>

→ 한 filter는 'Who'에, 또 다른 filter는 'What', 또 다른 filter는 'Whom'에 특화될 수 있음!  

그러나 attention은 attention score를 통한 'average'이기 때문에 이런 특정 attribute에만 특화될 수 없음!  
따라서 한 번의 attention은 모든 correlation에 대한 'mix'된 similarity를 계산하는 역할을 수행하게 됨  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553965-90410dbf-8186-4843-b4e8-2dabcf3116d7.JPG" alt="11" width="400px" />
</p>

**이를 다양한 filter를 주는 것처럼 여러번의 attention을 수행하게 한다면?**  
→ 'Who'에 대한 attention, 'What'에 대한 attention, 'Whom'에 대한 attention이 독립적으로 존재하게 유도한다면?  
= CNN의 filter와 같이 각각 다른 역할을 하는 projection을 학습하게 유도할 수 있음!  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553969-9d05e02b-2b34-43dd-b4a4-57fcdd30d497.JPG" alt="12" width="400px" />
</p>

여러개의 attention은 서로 간의 dependency가 없어 동시에 계산할 수 있으므로 병렬화시키는 것도 간단함  
= CNN의 특징을 simulation하는 또 다른 대안이 될 수 있음  
→ 이 과정에서 많은 softmax 연산이 발생하기는 하지만 한번에 이루어지는 dimension이 상대적으로 큼  
→ CNN은 filter size가 작으므로 병렬화 시에 FLOPs 상 차이가 상쇄될 수 있어 여전히 병렬화에 유리! 

Transfomer에서는 attention을 수행하는 각각의 module을 **Head**라고 칭함  
각각의 Head에서의 output을 concat한 후, 추가적인 trainable weight matrix <img src="https://render.githubusercontent.com/render/math?math=W^O">를 사용  
이를 통해 모든 layer의 출력 차원으로 먼저 정의해둔 <img src="https://render.githubusercontent.com/render/math?math=d_{model}"> 크기로 만듦  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553970-0209c53a-7010-4890-b95a-cdc5cd76a62e.JPG" alt="13" width="250px" />
</p>

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1, \cdots, \text{head}_h)W^O" height = "20px"> 
</p>

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\rightarrow\text{head}_i=\text{Attention}(QW^Q_i, KW^Q_i, VW^V_i)" height = "25px"> 
</p>

하지만 Transformer에서는 Multi-Head Attention만을 사용하지 않고 다시 한번 feed-foward network를 통과  
일반적인 Fully-Connected Network를 거쳐 attention을 통해 얻은 결과로 다시 representation 학습

**Residual Connection/Layer Normarlization**  

Encoder/Decoder의 구조를 보면 Positional Encoding이 시작에만 입력되는 것을 확인할 수 있음  
이는 학습의 편의성을 넘어서 **Residual Connection을 사용할 시에 position의 정보가 계속 유지되었기 때문에 사용**  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553972-7c1352b8-d602-42fd-a4b7-24697cdf1e44.JPG" alt="14" width="500px" />
</p>

Residual Connection을 사용했을 때 attention distribution을 보면 diagonal이 활성화됨  
= **원래의 position에 해당하는 encoding vector에 대해서 focus가 유지될 수 있었다는 것**  
→ Residual Connection은 다음 layer에 Positional Encoding을 **"carry"** 해줄 수 있음!  

만약 Residual Connection을 사용하지 않고 매번 Positional Encoding을 넣어줘도 비슷한 효과가 나타났음  
그러나 Residual Connection을 사용했을 때에 비해 accuracy의 확보가 어려웠음  
→ Residual Connection은 positional information을 공급하기 위해 반드시 필요하다는 것!  

추가적으로 Residual Connection을 거친 output은 **Layer Normarlization**을 거쳤음  
이는 Batch Normalization과 유사하게 정규화후 trainable parameter <img src="https://render.githubusercontent.com/render/math?math=\gamma, \beta"> 를 통해서 output을 다시 표현하는 것  
그러나 mini-batch마다 수행하는 것이 아니라 **output의 차원 <img src="https://render.githubusercontent.com/render/math?math=d_{model}"> 마다** 수행하는 것이 차이

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\text{LayerNorm}(x_i)=\gamma\hat{z_i} %2B \beta, i = 1, \cdots, d_{model}" height = "25x"> 
</p>

**Look-ahead Masking**  

Transformer는 NMT를 위해서 제안된 모델이므로 target language를 위한 Decoder가 필요함  
이 때 RNN이 아니라 Self-Attention을 사용한다면 **Decoder는 Gold Answer를 한번에 입력 받아야 함!**  
그러나 "Teacher Forcing"을 위해 한번에 Gold Answer를 준다면 **word 예측을 위해서 미래 시점의 word를 참고 가능**  
→ 일종의 copy, 또는 "cheating"에 해당하는 학습 방법을 제공하게 되는 것!  

따라서 Decoder에 input을 제공할 시에 해당 시점에서 미래에 해당하는 word들을 참고하지 못하도록 유도 필요  
Transformer에서는 Decoder의 첫번째 Multi-Head Attention 시에 Masking을 추가하는 것으로 이를 구현  
→ Attention Score를 계산하기 위한 softmax 연산 이전의 <img src="https://render.githubusercontent.com/render/math?math=QK^T"> matrix에 대해서 Masking  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553974-c4086156-1733-4a7f-9ac2-dce0d5107abf.jpg" alt="15" width="250px" />
</p>

## Application: Generative Model from Self-Similarity
### Image Transformer
image의 영역에서는 **Self-similarity**라고 불리는 유사한 substructure의 반복이 자주 관찰됨  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553976-2bd72902-8af7-4d2b-8447-8ca9c5eec1ff.JPG" alt="16" width="300px" />
</p>  

이러한 Self-similarity를 활용하는 것은 image 영역에서 Classical한 접근법이었음!  
예를 들어 denoising 영역에서 사용되던 **Non-local Means**가 있음  
이는 한 patch에 대해 similarity function을 통해서 유사한 patch를 선택해 이들의 information을 활용하는 방법임  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553979-04f1af44-4900-45c6-b878-4c26a2b76483.JPG" alt="17" width="300px" />
</p>  

**Image Transformer** = 이러한 content-similarity concept을 통해 Image Generation/Super-Resolution에 적용   
→ word embedding에 해당했던 부분을 image patch로 교환하는 방식을 통해 patch 별 similarity를 계산  
→ Auto-regressive model(e.g. PixlCNN)와 같이 다음 pixel을 주변 patch의 similarity를 통해서 생성("raster-scan")  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553980-a9e9a67b-b3e3-461b-9a96-3909fc29cdf9.jpg" alt="18" width="400px" />
</p>  

단, image에서는 input vector length가 매우 길어지므로 CNN을 사용하는 것이 조금 더 "cheap" 함  
→ 이를 극복하기 위해서 CNN의 가정과 유사하게 attention의 범위를 local neighborhood로 줄이는 방법을 선택  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553983-951dec84-d309-440a-9be1-f5407f2f12fb.jpg" alt="19" width="450px" />
</p>  

아직 Image Generation에서 dominant인 GAN/CNN-based super-resoltion에 비하면 performance 상 부족함  
그러나 기존의 Auto-regressive model(e.g. PixlCNN)보다는 더 나은 compression rate를 가질 수 있었음!  


### Music Transformer
music의 영역에서도 일정한 distance를 두고 반복되는 Motifs처럼 Self-similarity가 관찰됨   
따라서 concept적으로는 similarity를 찾아내는 방식으로 Auto-regressive Music Generation이 가능함  
→ note-to-note / event-to-event 간의 similarity를 찾게 유도하면 됨!

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553984-0f07b4d8-3100-41d9-a10f-bd8cd80d3db2.jpg" alt="20" width="450px" />
</p>  

attention을 사용하면 RNN보다 time-step 상 멀리 떨어진 note에 대한 information도 효과적으로 반영 가능  
동시에 CNN처럼 **translation invariant**하기 때문에 학습된 length와 상관없이 일관적인 결과를 낼 수 있음!  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553985-cf9b5397-b1e6-4ada-93cb-136bd9b92617.jpg" alt="21" width="450px" />
</p>  

그러나 Transformer 자체로는 **periodic**한 relation을 반영하는 것이 부족하기 때문에 추가적인 변경이 필요함!  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553987-c207c908-d39e-4daf-9753-f87c67a88c6a.jpg" alt="12" width="400px" />
</p>  

따라서 **Music Transformer**는 attention 계산 시에 query와 key 간의 distance를 반영할 수 있도록 유도  
= 기존에 존재하던 **Relative Positional Self-Attention**을 사용 가능  
→ attention score 계산 시에 dot product 외에 따로 relative distance를 반영하는 position matrix를 추가하는 방법   
→ 해당 matrix를 통해 embedding 간의 relative distance를 나타내는 또 다른 key를 얻어 attention 계산

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553990-b933a7a5-31e7-4436-af26-f7a1691e5e8b.JPG" alt="23" width="450px" />
</p>  

그러나 music의 경우는 NMT에서의 문장의 길이보다 압도적으로 긴 seqeuence를 다루게 되는 문제가 있음!   
→ 이렇게 계산하는 방법은 너무나도 큰 memory capacity를 요구하게 됨

따라서 postion matrix를 통한 key에 대해서 계산한 attention을 적절히 축소하는 방법을 선택  
→ <img src="https://render.githubusercontent.com/render/math?math=QK^T">와 addition이 성립될 수 있도록 적절히 reshaping/padding/slise 과정을 추가  
→ 해당 방법으로 Music Transfomer는 많은 memory capacity의 절약이 가능했음!

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/153553991-ae73d1d2-4d43-4d3d-b4aa-323dc071e090.JPG" alt="24" width= "550px" />
</p>  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\text{softmax}(QK^T %2B \text{skew}(QE^T_{rel}))" height = "22px"> 
</p>

#
*Reference: https://youtu.be/5vcj8kSwBCY*  
*Writer: Sangkyu Lee*