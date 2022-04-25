# 전달사항
* 데이터의 경우, clrc_pt_bsnf.csv(대장암_환자_기본정보), clrc_ex_diag1.csv(대장암_검사_진단), clrc_ex_diag2.csv(대장암_검사_진단) 이 3개의 파일을 사용합니다.


  해당 데이터들을 data 폴더 - CLRC 폴더에 넣어주시면 되겠습니다.(아래 파일구조 참고)
* 코드가 정상적으로 동작하는지 확인하는 게 목표이기 때문에, Training의 경우 epoch을 1로 설정하였습니다.

# 파일구조(참고)
```
        ├──backup
        ├──.gitignore
        ├──data_pipeline_extract.py
        ├──data_pipeline_ref.py
        ├── data_pipeline_v1.py
        ├── data_pipeline_v2.py
        ├── main_v1.py
        ├── models.py
        ├── recon.py
        ├── utils.py
        ├── data(폴더만 존재하는 상태)
        │       ├── CLRC
        │            ├──  clrc_pt_bsnf.csv
        │            ├──  clrc_ex_diag1.csv
        │            └──  clrc_ex_diag2.csv
        └── requirements.txt
``` 

# 가상환경 설정

* conda create -n 가상환경이름 python=3.7.13
* conda install -c anaconda tensorflow-gpu==2.2.0
* pip3 install -r requirements.txt

# 실행방법
1. git clone
```
git clone https://github.com/bmiskkuedu/synthetic_cancer_patients.git

```

2. 가상환경 설정 및 필요 패키지 설치
```
conda create -n 가상환경이름 python=3.7.13

conda install -c anaconda tensorflow-gpu==2.2.0

pip3 install -r requirements.txt
```

3. main.py 실행(서버 기준)
```
python3 main_v1.py
```
