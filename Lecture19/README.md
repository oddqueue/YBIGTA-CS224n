# Lecture 19

### Bias in AI

---

### Overview : Bias in AI (CV, NLP)

|Index|Subtitle|
|--- | --- |
|19.1.|What is Bias|
|19.2.|Is Bias Bad?|
|19.3.|Evaluating Algorithmic Bias|
|19.4.|Long Trip of single AI Model|

---

### 19.1. What is Bias

- Human Reporting Bias
    - AI 모델의 편향성은 주로 데이터의 편향성에서 비롯됩니다.
    - NLP Task의 경우에는 Corpus가 그 데이터가 될 것이고, Corpus를 마련하는 과정에서 필연적으로 고정관념 등이 반영될 수 있습니다.
    - Corpus에서 단어가 등장하는 빈도만으로 Real-word Natural Language를 파악하려고 하다 보니 Bias가 발생합니다.
    - 보통, Neutral word보다 Toxic/Extreme word들이 corpus에서의 빈도가 높다고 합니다.

- Accumulated Bias
    - Trained Model의 Bias는 대개 악순환을 초래합니다.
    - Training data가 collected/annotated되어 Model train에 활용되고, 그 모델은 다시 Media filtering, ranking, generating, aggregating 등에 활용됩니다.
    - 이 과정에서 Bias는 대개 강화되는 경향이 있습니다.

- Types of Bias

    <br/>
    <p align = "center">
    <img src = "https://user-images.githubusercontent.com/75057952/155057545-5535466f-c385-4616-87c8-f94b9afc4360.JPG" width = "500dp">
    </p>

    - Bias in data
        - Reporting bias : 대체로 치료제는 효과가 있다는 보도가 우세하고, Rebuttal article은 드물다.
        - Selection bias : 국가별로 가난한 정도를 설문조사했는데 google form으로 받으면 안 되는 이유
        - Out-group homogeneity bias : 마치 미국인이 볼 때 한/중/일의 사람들이 비슷해 보이는 현상

    - Bias in data Representation
        - Biased Labels
        <br/>
        <p align = "center">
        <img src = "https://user-images.githubusercontent.com/75057952/155057547-6cb3aa03-0843-487d-a3e1-f4d7e70daeec.JPG" width = "500dp">
        </p>
    - Bias in Interpretation
        - Confirmation Bias : pre-existing belief와 부합하는 정보만을 recall한다.
        - Overgeneralization : 사람은 다리가 두 개이다. 새는 다리가 두 개이다. 나는 새이다.(Overfitting과도 연관)
        - Automation Bias : Automated decision making system에서 추출된 정보를 더 신뢰하는 경향(AI 만능주의)
        <br/>
        <p align = "center">
        <img src = "https://user-images.githubusercontent.com/75057952/155057548-bce16689-9161-47de-b0e2-88ae66de6ef4.JPG" width = "500dp">
        </p>

    - Biased Human data >> Biased Representation >> Biased ML Model >> Biased Network... >> gets into a infinite feedback loop

### 19.2. Is Bias Bad?

- Bias can be *good, bad, neutral*
    - Bias in statistics in ML
    - Neural network의 single neuron에도 bias term 존재

- Cognitive bias
    - 위에서 언급한 human bias
- Algorithmic bias

    <br/>
    <p align = "center">
    <img src = "https://user-images.githubusercontent.com/75057952/155057550-47e7fa4d-32fb-4f24-aec0-8865ed8078d4.JPG" width = "500dp">
    </p>

    - Why it is a big deal? **Amplifying Injustice**
    - Criminal 에측, Policing algorithm에서 인종적 편향 보고
    - Terror, White-Collar 등등의 Bias가 facial image로부터 학습되었다는 보고

### 19.3. Evaluating Algorithmic Bias

- Checking for TP, TN, FP, FN counts
    - Labeled된 데이터(주로 classify)를 바탕으로 TP, TN, FP, FN을 나눈 다음, 각각의 subgroup별로 분포가 다른 feature가 있는지 찾습니다.
    - 해당 feature를 interrupt했을 때 실제 모델 성능에 영향을 미치지 않는다면, 해당 feature는 algorithmic bias에 의한 것으로 판단할 수 있습니다.
- 실제로는 FP가 FN보다 낫다는 보고가 있습니다.
    - 대개 False Positive는 "처리되지 말아야 할 것이 처리되는 것"으로, ML 모델은 대개 유용한 쪽으로 개발되기 때문에 유용한 처리가 추가되는 것은 '비효율성의 문제'를 야기합니다.
    - 하지만 FN는 처리되어야 할 것이 처리되지 않은 것이므로 '비합리성의 문제'를 야기한다고 알려져 있습니다.
    - 예를 들어 모자이크 처리를 한다고 했을 때, 카드 번호를 모자이크하지 않는 것보다는 그냥 카드 회사까지 같이 모자이크하는 것이 나을 것입니다.


### 19.4. Long Trip of single AI Model

<br/>
<p align = "center">
<img src = "https://user-images.githubusercontent.com/75057952/155057552-396542c1-9e15-4d4a-a238-1e256087160b.JPG" width = "500dp">
</p>

- Single AI Model은 Long-term Impact가 더 크다고 보고되어 왔습니다.
- AI Model을 Human-Friendly하게 설게하는 것은 굉장히 중요한 일이 되었습니다.
> **Data Really, Really Matters**

- Data의 skew, correlation을 파악하는 것이 중요합니다.
- 비슷한 distribution으로부터 single training set/testing set을 확보하기보다는, multiple source로부터 input을 확보하는 것이 바람직합니다.
    
- **ML Techniques for Bias Mitigation and Inclusion**
    - Remove the signal for problematic output
        - 고정관념, -ism 류의 output을 도출할 수 있는 가능성을 model 수준에서 차단
    - Adding signal for 'desired' varibles
        - still controversial : 'what is desirable results?'

- [Benton et al.](https://arxiv.org/pdf/1712.03538)
    - Internal Data
        - EMR records
        - Mental health diagnoses, suicide attempts, completions data
        - Social media data of 'patients'
    - Proxy Data
        - Twitter media data
        - Proxy mental health diagnoses
            - 'I've been diagnosed with X'
            - 'I tried to commit sucicide'
> **Multi-task Learning**
<p align = "center">
<img src = "https://user-images.githubusercontent.com/75057952/155057556-37137683-3005-4eb0-874d-0ac00f50e82f.JPG" width = "500dp">
<img src = "https://user-images.githubusercontent.com/75057952/155057535-5c1cf250-9292-44b0-b1bf-950524d63564.JPG" width = "500dp">
<img src = "https://user-images.githubusercontent.com/75057952/155057538-f6b1d9ab-aeff-4ca1-8cb5-9fc71086e95b.JPG" width = "500dp">
</p>
        
- Cormorbidity detection에 더욱 효과적
- 동반 질환을 detect하여 subgroup간 perforamce 비교 시 MTL(Multi-task learning) group에서 예후 개선

>  **Multitask Adverserial Learning**
<p align = "center">
<img src = "https://user-images.githubusercontent.com/75057952/155057540-16d3a5a1-6a11-442c-85b2-f6b69e10ee37.JPG" width = "500dp">
</p>

### 19.5. Jigsaw

- Kaggle에서 Jigsaw 대회가 열린 적이 있었습니다.
- 문장의 Toxicity를 분류하거나, Model score를 내어서 Regression할 수도 있습니다.
- 여기서는 False Positive Bias가 문제가 되었습니다.
    - "I'm a pround gay/lesbian person" : model score 0.6
    <p align = "center">
    <img src = "https://user-images.githubusercontent.com/75057952/155057542-1f0bc5c7-9484-4993-997e-e1a1889597c0.JPG" width = "500dp">
    </p>
- Unintende Bias를 제거하는 것이 쉽지 않았던 이유는, 각각의 사례들은 대개 'unique'했기 때문입니다. 
- 유사한 문장들과의 비교를 통해 Toxicity score를 학습해야 하는데, 그것이 쉽지 않다 보니 pretrain된 model의 Bias가 반영된 것이라고 보고했습니다.
- 그래서 대회 참가자들은 의도적으로 bias를 negate할 수 있는 데이터셋을 추가하여 학습하기도 했습니다.
- Metirc 자체도 AUC로 할지, Toxicity 분류에 따른 score distribution을 그려서 distribution의 겹친 면적으로 할지 대회에서 많이 고민했다고 합니다.