# ML/DL STUDY
## ML 
- 머신러닝을 위한 수학: 퍼셉트론, KNN 모델, 나이브 베이즈, 정보획득량(지니계수, 엔트로피) 🗒️ [노트](https://changeable-yacht-8d2.notion.site/00-75cd73fc8bf24bdbbadb73a11d8c5d1c?pvs=4)
- KNN 🗒️ [노트](https://changeable-yacht-8d2.notion.site/089-1-knn-24da543dc20a46f190ae286747d749c5?pvs=4)
- 나이브 베이즈 🗒️ [노트](https://changeable-yacht-8d2.notion.site/090-2-d27890027a104351921ba695abd2c998?pvs=4)
- 의사결정트리, 랜덤포레스트 🗒️ [노트1](https://changeable-yacht-8d2.notion.site/091-3-e4b3d119dead4e788f1e02c788c98eca?pvs=4) 🗒️ [노트2](https://changeable-yacht-8d2.notion.site/097-9-2c90382113be442ea7e058773bf865da?pvs=4)
- 신경망 🗒️ [노트1](https://changeable-yacht-8d2.notion.site/092-4-e3ca2e8b725f446289cc05a41dd67dd2?pvs=4) 🗒️ [노트2](https://changeable-yacht-8d2.notion.site/096-8-ff86f5fde20a4b7fb2bbc1f60a6cf256?pvs=4)
- 로지스틱 회귀, SVM 🗒️ [노트](https://changeable-yacht-8d2.notion.site/093-5-813cfc3afde441eaa2b1686720f654b8?pvs=4)
- 다중 회귀 🗒️ [노트](https://changeable-yacht-8d2.notion.site/094-6-6b01fba32f1a437db984ad05de03b1ef?pvs=4)
- 단순 회귀 🗒️ [노트](https://changeable-yacht-8d2.notion.site/095-7-9f144e7ee835416e80c1634602ded779?pvs=4)

---

## DL 
### mnist 필기체 데이터 분류 | Tensorflow | CNN
- Data: Tensorflow에 내장된 mnist 필기체 데이터
- model 1: FC3층/SGD -> acc: 0.9346 - loss: 0.2425
- model 2: FC3층/SGD/BatchNormalization/Dropout/Earlystop -> acc: 0.9894 - loss: 0.0341 </br>
🔗 [코드](DL/DL_mnist_CNN.ipynb)


### 닮은꼴 배우 조우진 vs 김병철 분류 | Tensorflow | VGG19
- Data: 구글 이미지 스크래핑으로 배우 별 100장 수집
- model 1: VGG19+FC3층 -> acc: 0.9290 - loss: 0.1860
- model 2: VGG19+FC3층/검증데이터분리/데이터증강/합성곱층 고정/뉴런(128,64)/Adam학습률o -> acc: 0.6940 - loss: 0.6225
- model 3: VGG19+FC3층/검증데이터분리/데이터증강/뉴런(128,64)/Adam학습률x </br>
🔗 [코드]()


### 유동인구 카운트 YOLO 영상 파일럿 프로젝트
- **프로젝트 설명**: YOLO 모델을 사용하여 영상 속 유동인구를 실시간으로 감지하고, 입장 및 퇴장 수를 카운트하는 시스템.
- **주요 기능**: 사람 객체 탐지, 입장 및 퇴장 카운팅, 누적 인원 계산 </br>
🔗 [코드](https://colab.research.google.com/drive/1g1eJ_ly3gkeQ5JVrr_86XQi1GimkRnyH?usp=sharing)
- **시연 이미지**:  
![시연 이미지](https://github.com/goguma999/pilot/blob/main/count/sjk.jpg?raw=true)



