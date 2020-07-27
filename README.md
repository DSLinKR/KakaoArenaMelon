# Team MelonMelody's KakaoArenaMelon Recs competition

## 재현 방법
1. main에 있는 python 파일들과 train, test 데이터셋을 같은 경로에 저장합니다.
2. 해당 경로에서 다음 코드를 terminal에 시행합니다.

<pre>
<code>
python inference.py
</code>
</pre>

3. 파일들이 포함된 경로에 결과파일이 *results.json* 으로 저장되어있을 것입니다.
  



## ALS
<a href="https://www.codecogs.com/eqnedit.php?latex=R=UV" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R=UV" title="R=UV" /></a>

ALS(Alternative Least Squares)은 CSR 형태로 나타내지는 행렬 R을 User 행렬과 Item 행렬의 곱으로 나타내는 방법.

## User CF
기존에 알려진 User-based KNN 기법을 이용하되, user간(즉 playlist간) 코사인 유사도를 계산할 때 추가적으로 각 노래의 inverse song frequency를 곱해서 song score를 구하도록 만들었습니다.
이때, 마지막에 추출하는 각 노래의 song score는 normalization 과정을 거쳐서 normalized score로 return하도록 만들었습니다.

## Item CF

## Gated CNN

## Blending

## XGBoost

## FastText Weighted Tf-Idf
playlist title은 존재하지만 songs, tags가 모두 전혀 주어지지 않은 플레이리스트들을 cold start라고 규정하였습니다.
이러한 플레이리스트들의 경우, FastText 모델과 tf-idf vectorizer를 이용하여 playlist title간 유사도를 계산하여, 가장 유사한 플레이리스트들 100개(혹은 150개)를 후보군으로 두고,
후보군 플레이리스트들에서 나타나는 songs와 tags 전체를 빈도 순으로 정렬하여 가장 자주 나온 song 100개, tag 10개를 return하도록 만들었습니다.

> 주석
>> Maksims Volkovs, Himanshu Rai, Zhaoyue Cheng, Ga Wu, Yichao Lu, Scott Sanner. 2018. Two-stage Model for Automatic Playlist Continuation at Scale.  
>> Yann N. Dauphin, Angela Fan, Michael Auli, David Grangier. 2016. Language Modeling with Gated Convolutional Networks
