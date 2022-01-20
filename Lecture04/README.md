# Lecture04 
## Matrix Gradients
score에 대한 total weight의 Jacobian은 다음과 같이 나타낼 수 있음  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial s}{\partial\mathbf{W}} = \mathbf{\delta}\frac{\partial \mathbf{z}}{\partial \mathbf{W}} = \mathbf{\delta}\frac{\partial}{\partial \mathbf{W}}(\mathbf{Wx} %2B \mathbf{b})" height = "30px"> 
</p>

<img src="https://render.githubusercontent.com/render/math?math=$\mathbf{z}$">에 대한 total weight의 Jacobian(<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial \mathbf{z}}{\partial \mathbf{W}}">)을 조금 더 쉽게 이해하기 위해여 single weight로(<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial z_i}{\partial W_{ij}}">)부터 생각  
- 특정한 single weight <img src="https://render.githubusercontent.com/render/math?math=W_{ij}">가 기여하는 term은 오직 <img src="https://render.githubusercontent.com/render/math?math=z_i">임  
- 이는 신경망의 연결 형태로부터 쉽게 추론할 수 있음  
- 결과적으로는 <img src="https://render.githubusercontent.com/render/math?math=W_{ij}">와 <img src="https://render.githubusercontent.com/render/math?math=x_j">들의 선형결합으로 표현되므로 값은 단순히 <img src="https://render.githubusercontent.com/render/math?math=x_j">가 되는 것  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/150362641-57357af2-91ca-4579-841e-602ad2ab55d0.jpg" alt="1" width="250px" />
</p>

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial z_i}{\partial W_{ij}} = \frac{\partial}{\partial W_{ij}}(\mathbf{W_{i} \cdot x} %2B \mathbf{b}) = \frac{\partial}{\partial W_{ij}}(\sum_{k=1}^d{W_{i} \cdot x} %2B \mathbf{b}) = x_j" height="40px">
</p>

→ error signal <img src="https://render.githubusercontent.com/render/math?math=\delta">와 결합하여 score <img src="https://render.githubusercontent.com/render/math?math=s">에 대한 single parameter derivative도 다음과 같다고 할 수 있음  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial s_i}{\partial W_{ij}} = \delta_i x_j" height = "40px">
</p>

이를 목적에 해당하는 전체 parameter matrix <img src="https://render.githubusercontent.com/render/math?math=W_{ij}">로 확장  
→ 전 단계에서 발생하는 error signal <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\delta}">와 input vector <img src="https://render.githubusercontent.com/render/math?math={\mathbf{x}}"> 사이의 outer product(tensor product)와 동일  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial s}{\partial \mathbf{W}} = \mathbf{\delta} \otimes \mathbf{x}" height = "30px">
</p>

결과적으로 얻는 Jacobian의 크기는 <img src="https://render.githubusercontent.com/render/math?math=(n \times 1) \times (1 \times m ) = (n \times m)">  

이렇게 gradient를 유도할 때 유의해두면 좋은 점은 다음과 같이 정리할 수 있음  

>1. 변수들을 정의할 때 그들의 차원을 유의
>2. chain rule을 명심하고 activation function에 대해서도 유의
>3. softmax activation과 같은 경우 특정 class에 대해서 correct한 경우와 아닌 경우에 대해서 나눠서 생각  
>4. matrix calculus에 혼동이 올 경우 element-wise하게 먼저 시도
>5. error signal <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\delta}">의 차원은 hidden layer의 차원과 같다는 것을 기억

이 내용들은 일반적인 MLP에 대한 gradient 유도였지만, NLP에서는 바로 적용될 수 없음  
→ NLP에서는 input vector에 해당하는 <img src="https://render.githubusercontent.com/render/math?math=\mathbf{x}">가 단순히 fixed된 input이 아닌 embedding된 word vector이기 때문

최종적으로 gradient를 계산할 때 한 window에 등장하는 word vector에 대한 Jacobian을 계산할 수도 있음  
→ 이렇게 얻은 gradient로 word vector 자체도 update 해야하는 것인지 판단할 필요가 있음

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{\delta_{window} = } \begin{bmatrix} \nabla x_{museums} \\ \nabla x_{in} \\ \nabla x_{Paris} \\ \nabla x_{are} \\ \nabla x_{amazing} \end{bmatrix} \in \mathbb{R}^{5d}" height="120px"/>
</p>

word vector에 대한 gradient를 통해 update하는 것은 vector space 상에서 해당 word vector를 **이동시키는** 효과  
→ 그러나 다른 simillar word vector와 달리 dataset에 존재하지 않는 word의 경우만 변화가 누락될 수 있음  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/150362646-200d1de9-f87f-4597-b84d-4baa5f00a610.jpg" alt="2" width="350px" />
</p>

따라서 우리가 pre-trained된 word vector를 사용하는 경우에는 다음과 같은 사항이 고려되어야 함

> pre-trained된 word vector를 사용하는 것이 옳은가?  
> - 만약 그것이 가능하다면 대부분의 경우에 좋음
> - 우리가 얻을 수 있는 일반적인 dataset보다 더 좋은 embedding이 되어 있을 것  
> 
> 이렇게 사용하는 pre-trained word vector를 update(fine-tune) 해야하는가?
> - 만약 사용하는 dataset이 작다면 update하지 않고 fix해서 사용
> - 만약 dataset이 충분히 크다면 fine-tune하는 것을 고려 가능

## Backpropagation and Computation Graph
지금까지는 단순히 chain rule을 통해서 gradient를 도출해냈지만 이 것은 효율적인 방법이 아님  
각 hidden layer마다 그 다음 단계(successor)에서 발생하는 error signal <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\delta}">가 계속 등장함  
이를 효율적으로 활용하는 방법이 **Computation Graph**를 이용하는 것  

Computation Graph는 interior node가 필요한 operation을, edge에는 그에 해당하는 결과물을 갖는 Graph임  
이에 따라서 input을 통해 output을 도출하는 과정을 **Forward Propagation**이라고 칭함  
edge를 반대방향으로 탐색하여 각 node 별 gradients를 뒤로 전달하는 것을 **Backpropagation**이라 칭함  

Computation Graph의 각 node는 2가지 정보를 지니고 있음
>1. 해당 node에서의 operation
>2. 해당 node에서의 local gradient

이 때, 해당 node의 succesor로부터 전달된 gradient을 **upstream gradient**라고 칭함  
해당 node가 계산한 predecessor로 전달하는 gradient를 **downstream gradient**라고 칭함  
chain rule로 부터 **downstream gradient = local gradient × upstream gradient**라는 것을 쉽게 이해할 수 있음  

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/150362650-3be4de30-5e7d-4c91-88ff-74f7878f2f44.jpg" alt="3" height="150px" />
</p>

gradient의 계산에서 successor가 여러개일 경우 각각의 upstream gradient와 local gradient의 곱을 모두 더함  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial s}{\partial z} = \sum_{i=1}^n \frac{\partial s}{\partial h_i}\frac{\partial h_i}{\partial z}" height="40px">
</p>

이렇게 downstream gradient를 계속 유지하는 것을 통해서 gradient <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\delta}">를 여러번 계산하지 않도록 만듦

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/150362717-e37ce93e-e346-4cdb-b64f-e05b5ac9e701.jpg" alt="4" height="150px" />
</p>

대표적인 연산에 대한 downstream gradient의 계산을 다음과 같은 직관으로 기억해둘 수 있음  
>1. +는 upstream gradient를 'distribute'
>2. max는 upstream gradient를 'routes'
>3. ×는 upstream gradient를 'switch'

실제로 Backpropagation을 진행하는 것은 다음과 같은 step을 따름
>1. 주어진 Computational Graph를 topolgical sorting
>2. 계산된 topological order를 통해서 Forward Propagation 진행
>3. topological order를 역순으로 사용하여 Backpropagation 진행

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/150362725-3848ada3-1b8f-420f-a714-bc843f3414c2.jpg" alt="5" height="400px" />
</p>

이렇게 진행할 경우 Forward Propagation과 Backpropagation은 동일한 running time을 갖게 됨     
→ gradient를 더 이상 수학적 계산 없이 자동적으로 계산해주는 일종의 Automatic Differentiation algorithm이 됨  

현재는 Pytoch, Tensorflow와 같은 framework가 자동적으로 수행
- 과거에는  gradient가 잘 계산되었는지 직접 확인할 필요가 있었음  
- 따라서 수치미분을 통해서 각 parameter 별로 gradient가 잘 계산되었는지 직접 확인할 필요가 있었음  
- 이 때 two-side인 경우가 one-side인 경우보다 numerically stable하므로 이를 주로 사용함  
- 만약 특정 node를 직접 implement할 경우에는 debugging을 위해 필요한 task이므로 알아둘 필요가 있음  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=f^{\prime}(x) \approx \frac{f(x %2B h) - f(x-h)}{2h}, \,\,h \approx 0.0001" height="30px">
</p>

## Some Tips to Know
### Regularization 
- 과거의 고전적인 통계학에서는 data의 수보다 많은 parameter를 갖는 model은 좋지 않다고 판단함  
- 이는 paramter가 너무 많으면 각 data를 **외워버리는** 효과가 쉽게 발생하기 때문  
- 그러나 Neural Network는 over-parameterazation에서 더 좋은 성능을 보이는게 경험적으로 확인됨  
→ parameter가 너무 training data에 최적화되지 않도록 유도해야 함  
→ objective에 weight에 대한 regularization term을 추가할 수 있음(e.g. L2 regularization)

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=J_{L2}(\theta) = J(\theta) %2B \lambda \sum_k \theta^2" height="25px">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/86907286/150362736-474f9852-3304-416c-9f64-adc40167485b.jpg" alt="6" width="350px" />
</p>

### Vectorization
- vector 연산시에 각 vector마다 loop를 통해서 dot product를 사용 가능
- vector를 하나의 matrix로 묶어서 matrix operation으로 바꾸는 것보다는 느림  
→ loop의 사용보다 matrix로 연산하는 것을 지향해야하며, 이는 GPU를 사용할 때 더욱 두드러지는 특징임  

### Nonlinearities
- affine activation을 사용하는 것은 layer의 증가를 통한 model의 설명력을 증가시키는 효과를 얻지 못함  
(i.e multiple matrix product = linear transform = single matrix)  
→ node마다 비선형 activation을 추가하는 것이 반드시 필요  
여러 가지 activation function이 연구됨(sigmoid, tanh, hard tanh, ReLU, Leaky ReLU)  
→ 계산의 효율성을 단순화된 activation도 효과적이라는 점에서 ReLU, Leaky ReLU가 자주 사용됨 

### Parameter Initailization
- 학습 시작 시에 parameter의 초기값을 small random value로 사용하는 것이 매우 중요함  
- model에서 weight들의 symmetries를 깨서 각 unit이 특정 representation에 대한 specialization을 잘 할 수 있도록 유도할 수 있음  
- 일반적으로는 uniform distribution를 통해서 초기화되며, 특히 주로 사용되는 방법은 Xavier Initization임  
→ activation function의 domain에 중간에 weight 값이 위치하도록 fan-in/fan-out으로 분산을 정규화하는 방법임  

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=Var(W_i) = \frac{2}{n_{in} %2B n_{out}}" height="35px">
</p>

### Optimizers
- SGD를 사용해도 어느정도 잘 학습이 되지만, 좋은 performance를 위해서는 learning rate의 hand-tuning이 필요  
- 만약 model이나 situation이 복잡한 경우에 이를 쉽게 대응할 수 있도록 다양한 optimizer가 연구되었음  
  (e.g. Adagrad, RMSprop, Adam, SparseAdam)  
→ 이전 단계의 gradient를 기억하면서 parameter마다 gradient에 비례하여 learning rate를 결정하는 방법임  

### Learning rates
- 학습이 진행될 수록 learning rate를 줄이는 것은 좋은 performance를 달성하는데에 도움이 됨  
- 주로 10의 제곱을 곱하면서 감소시키거나 수식을 통해서도 적용될 수 있음
- local optimum의 회피를 위해 cyclic한 learning rate를 줄 수도 있음  
-  SGD에서 발전된 optimizer에 처음으로 제공하는 learning 값은 inital value  
   → training stage에 따라서 자동으로 shrink됨

#
*Reference: https://www.youtube.com/watch?v=yLYHDSv-288*  
*Writer: Sangkyu Lee*