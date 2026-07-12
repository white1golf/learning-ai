# LLM과 딥러닝 직접 구현 학습

이 저장소의 목표는 완성된 라이브러리를 사용하는 데 그치지 않고, 작은 구성 요소부터 직접 만들면서 딥러닝과 LLM의 작동 원리를 이해하는 것입니다.

## 전체 학습 로드맵

아래 순서를 유지하며 앞 단계의 개념과 코드를 다음 단계에서 활용합니다.

### 01. 작은 신경망 직접 구현

- Python과 NumPy로 뉴런 구현
- 여러 뉴런으로 완전연결층(Dense Layer) 구현
- 활성화 함수 구현
- 손실 함수 구현
- 수치 미분과 경사하강법 구현
- 역전파가 실제로 가중치와 편향을 바꾸는 과정 확인
- 작은 2층 신경망 학습

폴더: `01-numpy-neural-network/`

### 02. PyTorch로 딥러닝 모델 학습

- MNIST 숫자 분류기 제작
- 데이터셋과 DataLoader 이해
- 배치 단위 학습
- 학습 및 평가 루프 작성
- 과적합의 발생과 대응 방법 확인

예정 폴더: `02-pytorch-mnist/`

### 03. 언어 모델의 출발점 만들기

- 문자 단위 Bigram 언어 모델 구현
- 문장을 숫자 토큰으로 변환
- 다음 글자 예측
- 학습한 모델로 새로운 텍스트 생성

예정 폴더: `03-bigram-language-model/`

### 04. Transformer 직접 구현

- Embedding
- Positional encoding
- Self-attention
- Multi-head attention
- Feed-forward network
- Causal masking

예정 폴더: `04-transformer-from-scratch/`

### 05. 미니 GPT 완성

- 작은 텍스트 데이터로 모델 학습
- temperature와 top-k를 이용한 문장 생성
- 모델 크기에 따른 결과 비교
- 데이터 품질에 따른 결과 비교

예정 폴더: `05-mini-gpt/`

### 06. 실제 LLM 활용

- 사전 학습 모델 실행
- 프롬프트 작성
- RAG와 임베딩 구현
- LoRA를 이용한 소규모 파인튜닝

예정 폴더: `06-practical-llm/`

## 폴더와 파일 작성 규칙

학습 과정을 나중에 처음부터 복기할 수 있도록 다음 규칙을 지킵니다.

1. 큰 학습 단계마다 별도 폴더를 만든다.
2. 폴더 이름 앞에는 학습 순서에 따라 `01-`, `02-`와 같은 두 자리 번호를 붙인다.
3. 각 폴더에는 해당 단계만의 `README.md`, 의존성 및 실행 환경을 둔다.
4. 폴더 안의 실습 파일에도 `01_`, `02_`와 같은 두 자리 번호를 붙인다.
5. 새로운 개념을 배울 때 이전 실습 파일을 덮어쓰지 않고 새 파일을 추가한다.
6. 각 파일에는 개념 설명, 계산 과정, 실행 가능한 예제와 관찰할 내용을 함께 남긴다.
7. 폴더의 `README.md`에는 파일별 학습 내용과 실행 명령을 계속 기록한다.
8. 예제를 실행해 예상 결과가 나오는지 확인한 뒤 다음 단계로 넘어간다.

예상 구조는 다음과 같습니다.

```text
learning-ai/
├── README.md
├── 01-numpy-neural-network/
│   ├── README.md
│   ├── main.py
│   ├── 01_weights_and_bias.py
│   ├── 02_dense_layer.py
│   └── 03_activation_functions.py
├── 02-pytorch-mnist/
│   ├── README.md
│   ├── 01_dataset.py
│   └── 02_training_loop.py
└── ...
```

`main.py`는 해당 폴더의 환경 확인 또는 전체 실행 진입점으로 사용할 수 있습니다. 개별 학습 과정은 번호가 붙은 파일에 보존합니다.

## 현재 진행 상황

- 현재 단계: **01. 작은 신경망 직접 구현**
- 완료: Python·uv·NumPy 환경 구성 및 검증
- 완료: 단일 뉴런의 가중치와 편향
- 완료: 여러 뉴런을 묶은 완전연결층(Dense Layer)
- 완료: 여러 입력 샘플을 한 번에 처리하는 배치(Batch)
- 완료: 비선형성을 만드는 활성화 함수
- 다음 실습: 두 개의 완전연결층을 연결한 작은 신경망의 순전파

현재 폴더로 이동해 다음과 같이 실행합니다.

```bash
cd 01-numpy-neural-network
uv sync
uv run python 01_weights_and_bias.py
```
