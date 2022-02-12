# Lecture07 – Vanishing Gradients, Fancy RNNs 
## Vanishing/Exploding Gradient
RNN에서의 Backpropagation 시에 gradient는 어떻게 진행될까?
- 일반적인 Dense Neural Network와 같이 chain rule을 적용해보기
- 만약 <img src="https://render.githubusercontent.com/render/math?math=$t=4$">인 경우 Loss에서의 gradient를 생각해보면 다음과 같이 표현할 수 있음  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/151706356-fbc0b41d-4c9c-46e4-8c7b-773187213c33.JPG" alt="1" width="400px" />
</p>

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial J^{(4)}}{\partial\mathbf{h}^{(1)}}=\frac{\partial\mathbf{h}^{(2)}}{\partial\mathbf{h}^{(1)}}\times\frac{\partial\mathbf{h}^{(3)}}{\partial\mathbf{h}^{(2)}}\times\frac{\partial\mathbf{h}^{(4)}}{\partial\mathbf{h}^{(3)}}\times\frac{\partial J^{(4)}}{\partial\mathbf{h}^{(4)}}" height = "40px"> 
</p>  

Error로부터의 gradient는 대부분 인접한 hidden state 간의 gradient <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial\mathbf{h}^{(i)}}{\partial\mathbf{h}^{(i-1)}}">들의 곱으로 표현될 수 있음  
이를 일반화하면 chain rule을 통해 hidden state <img src="https://render.githubusercontent.com/render/math?math=\mathbf{h}^{(j)}">에 대한 step <img src="https://render.githubusercontent.com/render/math?math=$i$">에서의 gradient를 표현할 수 있음  
→ 만약 이런 **gradient들이 작다면 최종 gradient도 작게 되는 결과가 발생함!**  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{h}^{(t)}=\sigma({\mathbf{z_t}})=\sigma(\mathbf{W}_h\mathbf{h}^{(t-1)} %2B \mathbf{W}_x\mathbf{x}^{(t)} %2B \mathbf{b}_t)" height = "20px"> 
</p>  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\rightarrow\frac{\partial\mathbf{h}^{(t)}}{\partial\mathbf{h}^{(t-1)}}=diag(\sigma^{\prime}(\mathbf{W}_h\mathbf{h}^{(t-1)} %2B \mathbf{W}_x\mathbf{x}^{(t)} %2B \mathbf{b}_t))\mathbf{W}_h" height = "35px"> 
</p>  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\therefore\frac{\partial J^{(i)}(\theta)}{\partial\mathbf{h}^{(j)}}=\frac{\partial J^{(i)}(\theta)}{\partial\mathbf{h}^{(i)}} \mathbf{W}_h^{(i-j)}\prod_{j<t\le i} diag(\sigma^{\prime}(\mathbf{W}_h\mathbf{h}^{(t-1)} %2B \mathbf{W}_x\mathbf{x}^{(t)} %2B \mathbf{b}_t))" height = "35px"> 
</p>  

최종 gradient의 크기가 <img src="https://render.githubusercontent.com/render/math?math=\mathbf{W}_h^{(i-j)}">에 의해 결정되므로 만약 <img src="https://render.githubusercontent.com/render/math?math=\mathbf{W}_h"> 자체가 작다면 graident가 매우 작아질 것!  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\|\frac{\partial J^{(i)}(\theta)}{\partial\mathbf{h}^{(j)}}\| \le \|\frac{\partial J^{(i)}(\theta)}{\partial\mathbf{h}^{(i)}}\| \|\mathbf{W}_h^{(i-j)} \| \prod_{j<t\le i} \| diag(\sigma^{\prime}(\mathbf{W}_h\mathbf{h}^{(t-1)} %2B \mathbf{W}_x\mathbf{x}^{(t)} %2B \mathbf{b}_t))\|" height = "35px"> 
</p>  

→ <img src="https://render.githubusercontent.com/render/math?math=\mathbf{W}_h">의 가장 큰 singular value가 1보다 작으면 gradient는 shrink, 1보다 크면 gradient는 explode  
>원문([Pascanu et al.](http://proceedings.mlr.press/v28/pascanu13.pdf))에 따르면 activation function <img src="https://render.githubusercontent.com/render/math?math=\sigma">의 도함수의 boundness를 통해서 이를 유도함    
>즉, multivariate vector function일 경우 <img src="https://render.githubusercontent.com/render/math?math=\| diag(\sigma^{\prime}(\mathbf{z}_t)) \| \le \gamma">라면 <img src="https://render.githubusercontent.com/render/math?math=\|\mathbf{W}_h\| < \frac{1}{\gamma}">를 만족시켜야 한다는 것  
>이는 hidden state 간의 gradient가 <img src="https://render.githubusercontent.com/render/math?math=diag(\sigma^{\prime}(\mathbf{z}_t))\cdot\mathbf{W}_h">로 표현될 수 있기 때문에 나타나는 성질  
>만약 <img src="https://render.githubusercontent.com/render/math?math=\mathbf{W}_h">의 최대 singluar value가 <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{\gamma}">보다 작다면 <img src="https://render.githubusercontent.com/render/math?math=\|\mathbf{W}_h\|">는 <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{\gamma}">보다 클 수 없음 (operator norm, spectral radius)  
> <img src="https://render.githubusercontent.com/render/math?math=\therefore \|\diag(\sigma^{\prime}(\mathbf{z}_t))\cdot\mathbf{W}_h\|\le\|diag(\sigma^{\prime}(\mathbf{z}_t))\| \cdot \|\mathbf{W}_h\| < \gamma \cdot \frac{1}{\gamma} < 1"> (Caychy-Schwarz ineq.)  
> → 강의에서 **1을 bound로 언급**한 것은 원문에서 <img src="https://render.githubusercontent.com/render/math?math=\sigma">가 **identity function이었을 때 1**이라고 적어놨었기 때문!

### Why is Vanishing Gradient a Problem?
한 step에서의 loss에 대해서 멀리 떨어진 step에서의 hidden state로의 gradient가 잘 전달이 되지 않음    
→ 매 step에서 hidden state는 멀리 떨어진 step에 대한 **long-term effect에 대해서 제대로 배울 수 없음**   

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/151706358-9d15eeed-3bb3-4ad3-872f-4e48d4240c5d.JPG" alt="2" width="400px" />
</p>

한 step에서의 gradient는 해당 step보다 이전에서 발생한 과거의 해당 step으로의 effect라 볼 수 있음  
gradient가 step <img src="https://render.githubusercontent.com/render/math?math=n">을 넘어서 전달되지 못한다면 다음과 같은 경우를 구분할 수 없음  
> 1. step <img src="https://render.githubusercontent.com/render/math?math=n"> 이후에는 data 간의 dependency가 존재하지 않음  
> 2. dependecy가 존재하지만 gradient가 소실되면서 model이 이를 반영하지 않고 있음  

→ 두번째 경우를 NLP task에서 확인하지 못하는 것은 굉장히 치명적인 문제! (e.g. language model)
1. 만약 예측해야하는 word가 멀리 떨어진 대상일 경우 RNN으로 이를 예측하는 것이 어려움

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/151706359-734caa51-fa1a-4d52-9524-9633e5573773.JPG" alt="3" width="400px" />
</p>

1. 예측할 word를 위해 참조해야할 것이 문법적으로 관련이 있는 word 일 경우에 대해서 어려움  
  (Syntatic recency vs Sequential recency, Linzen et al, 2016) 

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/151706361-30e370e5-eba3-4b75-9e1d-688545debc1b.JPG" alt="4" width="400px" />
</p>

### Why is Exploding Gradient a Problem? 
SGD를 사용하여 update할 때 너무 큰 step을 진행하게 되어 좋지 않게 parameter가 변할 수 있음  
→ 최악의 경우 gradient가 Inf/NaN이 발생하게 되어 완전히 training을 다시 해야할 수도 있음

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\theta^{new} = \theta^{old} - \alpha \nabla_{\theta}J(\theta)" height = "20px"> 
</p>  

한가지 solution은 **gradient clipping** 
- SGD를 사용하여 update 시에 gradient가 일정 threshold를 넘어가면 scaling 실행  
- gradient의 방향은 유지하면서 step size만을 조정하는 concept  
- objective surface에서 "cliff"가 있을 경우 gradient가 순간적으로 발산하는 것을 방지  
- gradient clipping은 이런 cliff를 **올라가는** 것을 방지하는 역할을 하게 됨  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/151706362-466dc388-01fa-4efd-bb1f-81c568984d68.JPG" alt="5" width="400px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/151706364-f70c9b7f-25c8-48c9-8583-b5e6341b405b.JPG" alt="6" width="400px" />
</p>

**그러나 여전히 gradient vanishing은 RNN structure에서 해결하기 어려운 문제임!**  
→ 근본적으로 RNN은 매 step마다 hidden state가 완전히 rewritten된다는 것이 문제  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{h}^{(t)}=\sigma(\mathbf{W}_h\mathbf{h}^{(t-1)} %2B \mathbf{W}_x\mathbf{x}^{(t)} %2B \mathbf{b}_t)" height = "20px"> 
</p>  

일종의 **memory의 역할을 수행하는 영역을 분리**할 방법이 필요함  

## Fancy RNNS
### Long Short-Term Memory (LSTM)
1997년에 Hochreiter, Schmidhuber가 RNN의 vanishing gradient에 대한 soulution으로 제시한 RNN model  
- 하나의 step <img src="https://render.githubusercontent.com/render/math?math=t">에는 hiddent state <img src="https://render.githubusercontent.com/render/math?math=\mathbf{h}^{(t)}">와 cell state <img src="https://render.githubusercontent.com/render/math?math=\mathbf{c}^{(t)}">가 존재
- LSTM은 cell의 information을 erase, write, read operation을 통해 활용 
- 이러한 operation은 <img src="https://render.githubusercontent.com/render/math?math=n">의 길이를 가진 vector인 gate를 통해 이용  
- gate는 0과 1 사이의 값을 가지며 매 state마다 dynamic하게 계산됨  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/151706365-19bd1313-127f-446d-819e-08b60bd81360.JPG" alt="7" width="500px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/151706396-e6584ada-f730-4a15-b0b9-7f894c5876b6.JPG" alt="8" width="500px" />
</p>

이론적으로 LSTM의 forget gate가 모든 step에서의 정보를 기억하게 결정된다면 모든 step의 정보를 사용할 수 있음  
그러나 vanilla RNN에서는 모든 step의 information을 기억하도록 <img src="https://render.githubusercontent.com/render/math?math=\mathbf{W}_h">를 학습하기 어려움  
→ LSTM은 gradient vanishing을 완전히 회피하는 것을 보장하지는 않지만 RNN보다 쉽게 회피 가능!  

LSTM은 2013 ~ 2015년에서 다양한 NLP task에서 state-of-the-arts였던 model이었음   
그러나 2019년도 이후에는 Transformers 기반의 model이 지배적으로 사용됨  

### Gated Reccurent Units(GRU)
2014년에 Cho et al. 이 LSTM을 간단화한 구조로 제안한 model  
- LSTM과 달리 cell state <img src="https://render.githubusercontent.com/render/math?math=\mathbf{c}^{(t)}">가 없음  
- cell state의 information이 아니라 전 단계의 hidden state를 gate를 통해 참조하여 memory의 역할 수행     
- 현재 state에서 전 단계의 hidden state에서의 중요한 정보를 gate를 통해서 선택하는 방법
 
<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/151706380-6d3c5760-4ada-4694-8366-68b9ee2e7dd1.JPG" alt="9" width="500px" />
</p>

LSTM과 유사하게 이론적으로 update gate가 0으로 결정된다면 모든 step의 정보가 기억되게 됨  
→ GRU도 상대적으로 gradient vanishing을 RNN보다 쉽게 회피 가능!  

다른 많은 RNN 변종이 있지만, LSTM과 GRU가 가장 많이 사용됨  
그러나 LSTM과 GRU 사이에 우위관계에 대해서 확실한 근거가 존재가 없음  
- 주요한 차이점은 GRU가 더 적은 parameter를 사용하기 때문에 빠르다는 것  
- 그러나 data가 상대적으로 long dependency를 갖고 있다면 LSTM의 경우가 더 좋음  
  → 먼저 LSTM을 시도해보고 조금 더 time-efficient 해야할 경우 GRU로 switch하는 것이 좋음  

### Is Vanishing/Exploding Gradients Just a RNN Problem?
vanishing/exploding gradient 자체는 Neural Network를 사용하는 Architecture의 고질적인 문제  
layer가 깊어지면서 activation function의 chain rule로 인해 Backpropagation 시에 gradient가 작아짐  
Deep Neural Network일 경우 상대적으로 얕은 layer가 상대적으로 잘 학습되지 않는 문제로 파생됨  

feed-forward network의 경우 layer 간의 Direct Connection을 추가하는 방식으로 극복되고 있음  
(e.g. ResNet, DenseNet, HighwayNet: 특정 layer, 또는 이후 전체에 identity/transform connection 추가)  
→ 그러나 RNN은 같은 weight matrix가 여러번 곱해지는 형태라 특히 gradient가 불안정함! (Bengio et al, 1994)


### Bidirectional RNN  
Sentimental Classification task에서 단순한 RNN을 사용할 때, **left context**만이 반영될 가능성이 있음  
예를 들어 특정 word가 긍정적인 대상을 수식하는 부정어일 경우에 옳바른 semantic으로 해석되기 어려움  
이는 sentence에서의 word sequence대로 input을 받는 RNN에서 발생하는 문제점     
→ 특정 word의 step에서 발생한 hidden state output은 그 word의 **left context만을 반영한 encoding**이 됨!  

따라서 right context를 반영하기 위해 input sequece를 역순으로 받는 RNN을 따로 정의  
→ 역순으로 input을 반영하여 각 word의 encoding을 concatenate해 최종 hidden state output으로 사용  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/151706370-3edc7359-eca7-4b55-9b74-8e2712d76442.JPG" alt="10" width="500px" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/151706374-011d99a4-634e-446a-a686-c15d89066b16.JPG" alt="11" width="500px" />
</p>

그러나 Bidirectional RNN은 전체 input sequence가 **온전히 제공되는 경우에만** 사용 가능  
따라서 language model과 같이 left context만 주어지는 경우에는 사용이 불가능함  
→ 만약 input sequence가 전부 주어진다면 Bidirectinal하게 구성하는 것을 default로 하는 것이 좋음  
(e.g. BERT(**Bidirectional** Encoder Representation from Transformers)도 Bidirectional하게 구성됨)

### Multi-layer RNN  

RNN은 여러번 step을 거쳐 input을 받는다는 점에서 step을 펼쳐서 본다면 "deep"하다고 할 수 있음  
그러나 각 step의 output을 또 다른 RNN을 거치게 만드는 방법(stack)을 통해서도 "deep"하게 구성 가능  
이렇게 여러 층의 RNN을 거치는 것은 조금 더 복잡한 representation을 학습할 수 있도록 도움  
(lower RNN = lower-level features, higher RNN = higher-level features)  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/151706375-c8e9e2d3-91e9-405f-8462-a32c9c8cc175.JPG" alt="12" width="500px" />
</p>

feed-foward network 만큼은 아니더라도 RNN에서도 이런 방식은 high-performance에 도움이 됨  
(e.g Britz et al, 2017: Neral Machine Translation에서 2 ~ 4 layer RNN encoder, 4 layer RNN decoder가 best)  
→ 그러나 deeper RNN을 훈련시키기 위해 skip-connection이 요구됨  
→ Transformer-based network의 경우 24 layers까지 stack 될 수 있음

#
*Reference: https://youtu.be/QEw0qEa0E50*  
*Writer: Sangkyu Lee*