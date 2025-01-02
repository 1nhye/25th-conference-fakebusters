
## Fakecatcher

#### Probabilistic Video Classification
1. fakeforensics metadata csv파일 만들기
```
cd 25th-conference-fakebusters/model/fakecatcher/data
python fakeforensics.py
```
2. Feature set 만들기
```
cd 25th-conference-fakebusters/model/fakecatcher/svr
python preprocess_feature.py
```
misc에 featureset에 대한 pkl파일이 저장됩니다. 

