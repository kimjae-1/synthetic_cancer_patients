# 전달사항
* 데이터의 경우, clrc_pt_bsnf.csv(대장암_환자_기본정보), clrc_ex_diag1.csv(대장암_검사_진단), clrc_ex_diag2.csv(대장암_검사_진단) 이 3개의 파일을 사용합니다.


  해당 데이터들을 RTSGAN-github/data/CLRC/ 에 넣어주시면 되겠습니다.(아래 파일구조 참고)


# 파일구조(참고)
```
├──RTSGAN-github
    ├──utils
    ├──aegan.py
    ├──autoencoder.py
    ├──basic.py
    ├──gan.py
    ├──main_2012.py
    ├──tstr.py
    ├──missingprocessor.py
    ├──physionet2012.py
    ├──preprocess.py
    ├──stdprocessor.py
    ├── data(폴더만 존재하는 상태)
    │       ├── CLRC
    │            ├──  clrc_pt_bsnf.csv
    │            ├──  clrc_ex_diag1.csv
    │            └──  clrc_ex_diag2.csv
    └── requirements.txt
  ``` 

# 가상환경 설정

* conda create -n 가상환경이름 python=3.7
* 
* pip3 install -r requirements.txt

# 실행방법
1. git clone
```
git clone https://github.com/bmiskkuedu/synthetic_cancer_patients.git

```

2. 가상환경 설정 및 필요 패키지 설치
```
conda create -n 가상환경이름 python=3.7

pip3 install -r requirements.txt
```

3. preprocess.py 실행(전처리)
```
python3 preprocess.py
```
학습에 필요한 connect_clrc.pkl이 rtsgan-connect-data 폴더 내부에 생성됩니다.

4. 모델 학습
```
python3 경로/main_2012.py \--dataset 경로/rtsgan-connect-data/connect_clrc.pkl \--epoch 11 \--iterations 1 \--log-dir ./result \--task-name test \--python-seed 42 \
```
./result/test/ 경로에 train_replaced_with_syn.pkl(합성데이터)이 생성됩니다.

5.합성 데이터 평가
```
python3 경로/miss2012.py \--dataset ./result/test/train_replaced_with_syn.pkl \--task-name miss \--impute zero
```
Examine the quality of synthetic data under the TSTR setting


/miss/test_info 경로에 info.log가 생성됩니다.
