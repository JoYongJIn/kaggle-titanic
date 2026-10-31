# Kaggle Titanic — PyTorch MLP

[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) 대회를 신경망(MLP)으로 푼 프로젝트입니다.
승객 정보(객실 등급, 성별, 나이, 가족 수 등)로 생존 여부를 예측하는 이진 분류 문제입니다.

이 저장소는 [ml-workbench](https://github.com/JoYongJIn/ml-workbench) 템플릿으로 만들어졌으며, 학습 실행·지표 시각화·Kaggle 제출을 웹 대시보드에서 관리합니다.

## 결과

| 실험 | 모델 | 검증 정확도 | Kaggle 공개 점수 |
|---|---|---|---|
| 성별 기준선 (Kaggle 제공) | 여성=생존, 남성=사망 | – | 0.76555 |
| baseline 64-32 | MLP 64→32, dropout 0.3, lr 1e-3 | 0.8101 | **0.76794** |

검증 세트(178명)와 공개 리더보드 점수 차이가 큰데, 이는 데이터가 작아 분할에 따라 점수가 크게 흔들리고 모델이 학습 데이터에 과적합된 영향입니다. 다음 실험 방향은 아래 "배운 점"에 정리했습니다.

## 접근 방법

**전처리 및 파생 변수** (`train.py` 의 `add_features`)

- 이름에서 호칭(Mr / Mrs / Miss / Master / Rare) 추출 — 나이·성별·신분 정보를 함께 담음
- 가족 크기(`SibSp + Parch + 1`)와 혼자 탑승 여부
- 객실 번호 유무 (상위 등급일수록 기록이 남아 있음)
- 나이 결측치는 호칭별 중앙값으로, 승선 항구 결측치는 최빈값으로 채움
- 고유값이 많은 문자열 열(이름, 티켓, 객실번호)은 제거, 나머지 범주형은 원-핫 인코딩, 숫자형은 표준화

**모델**

```
입력(43) → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
        → Linear(32) → BatchNorm → ReLU → Dropout(0.3)
        → Linear(1)  (BCEWithLogitsLoss)
```

Adam(weight decay 1e-4), ReduceLROnPlateau, 검증 손실 기준 조기 종료(patience 30). 최고 검증 손실 시점의 가중치로 test 예측.

## 재현 방법

```bash
/opt/homebrew/bin/python3.12 -m venv .venv        # Python 3.12 이상이면 어떤 것이든 가능
.venv/bin/pip install -r requirements.txt
kaggle competitions download -c titanic -p data && unzip -o data/titanic.zip -d data
.venv/bin/python train.py --run-dir runs/repro --lr 0.001 --epochs 300 --seed 42
```

`runs/repro/` 에 다음 파일이 생깁니다.

| 파일 | 내용 |
|---|---|
| metrics.jsonl | 에폭별 train/val loss, accuracy, 학습률 |
| summary.json | 최고 에폭과 검증 점수 |
| submission.csv | Kaggle 제출 파일 (`PassengerId, Survived`) |
| model.pt | 가중치와 feature 열 목록 (git 제외) |

하이퍼파라미터는 전부 CLI 인자로 받습니다. `python train.py --help` 로 목록을 볼 수 있습니다.

Kaggle 노트북 버전은 `notebooks/titanic-nn.ipynb` 이며, 같은 `train.py` 를 `/kaggle/input/titanic` 경로로 실행하도록 구성되어 있습니다.

## 구조

```
titanic/
├── train.py            # 전처리 + MLP + 학습 + 예측 (단일 파일, Kaggle 노트북과 동일 코드)
├── project.json        # 대회 slug, 타깃 열, 대시보드에 노출할 하이퍼파라미터 정의
├── requirements.txt
├── data/               # Kaggle 데이터 (재배포 금지라 git 제외)
├── runs/<id>/          # 실험별 설정·지표·제출 파일 (가중치와 로그만 git 제외)
└── notebooks/          # Kaggle 노트북 + kernel-metadata.json
```

## 배운 점

- 호칭(Title) 파생 변수가 나이 결측치 처리와 생존 예측 양쪽에 가장 효과가 컸다. `Master` 가 어린 남자아이를 분리해 주는 것이 성별 기준선 대비 개선의 핵심.
- 891행짜리 데이터에서 검증 정확도 81% 는 신뢰 구간이 넓다. 시드를 바꿔 여러 번 돌려 평균을 봐야 하고, 단일 분할 점수로 제출 여부를 결정하면 안 된다.
- 리더보드 점수가 검증 점수보다 4%p 낮았다. 다음 실험: 은닉층 축소(32→16), dropout 0.5, 교차 검증 평균, 그리고 트리 모델과의 비교.
