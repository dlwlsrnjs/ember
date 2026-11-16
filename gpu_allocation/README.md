# GPU 실행 패키지: 고정 LLM 피드백의 배분 × 음성 선택

현재 실험을 다른 GPU 서버에서 실행하기 위한 코드입니다. 저장소 루트의 기존
EMBER 구현과 구분되는 독립 실행 디렉터리입니다. 이 코드는 새로운 성능 결과나
논문 채택을 주장하지 않습니다.

같은 teacher 응답, 유지한 양성, 학습 예산에서 다음 네 조건을 비교합니다.

| 음성 배분 | LLM 거절 쌍 선택 T | 같은 queried pool에서 무작위 선택 R |
|---|---|---|
| 기존 배분 O | O_T | O_R |
| capacity 안에서 max-min 배분 U | U_T | U_R |

warm-only NoOp를 포함해 **5조건 × 3 cold settings × 3 paired 초기화 = 45회**
학습합니다. validation overall Recall@20으로 checkpoint와 조건을 선택하고,
선택한 checkpoint를 test와 확장 후보 집합에서 평가합니다. 기존 historical
test를 사용하는 탐색 실험이며 미사용 cohort 확증이 아닙니다. U를 무조건 우월한
방법으로 가정하지 않으며 max-min 배분 자체를 새 알고리즘으로 주장하지 않습니다.

## 1. 환경

로컬에서 확인한 환경은 Python 3.10.12, PyTorch 2.11.0+cu128,
NumPy 1.26.4, SciPy 1.11.4, H100입니다. CUDA 드라이버와 호환되는 torch를
설치해야 합니다. 다음은 로컬과 같은 CUDA 12.8 wheel 예시입니다.

```bash
cd gpu_allocation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

## 2. 입력 파일 전송

**코드만 clone하면 동일 실험의 입력은 아직 없습니다.** 데이터와 기존 LLM 판정,
학습 캐시는 GitHub에 올리지 않습니다. 원 서버에서 별도로 제공하는
`allocation_private_inputs_20261117.tar.gz`를 승인된 서버로 전송하고 이 디렉터리에
풀어야 합니다. 약 98MB의 입력 파일 21개이며, 압축 파일의 크기는 별도 확인합니다.
원본 MovieLens 이용 조건은 그대로 적용됩니다.

```bash
# 원 서버의 제공 경로에서 scp 등으로 이 파일을 먼저 가져옵니다.
tar -xzf /path/to/allocation_private_inputs_20261117.tar.gz -C .
python verify_inputs.py
```

`PRIVATE_INPUT_MANIFEST.json`도 별도 전송 묶음에서만 제공하며 GitHub에는 포함하지 않습니다.
압축을 푼 뒤 그 manifest의 SHA256을 모두 확인합니다. 저장된 입력은
MovieLens-1M 원본 3파일, corrected Q12K 판정 3개, 기존 T/R retained rows 6개,
T/R/NoOp 기준 캐시 9개입니다. 판정 생성에 사용한 역사적 retriever/invocation
전체를 복원하는 패키지는 아닙니다. 새로운 LLM 호출이나 모델 다운로드는
이 실험의 학습 단계에 필요하지 않습니다.

## 3. GPU에서 실행

두 GPU가 있을 때:

```bash
nohup python -u run_gpu.py --gpu-a 0 --gpu-b 1 --wait-for-gpu > launch.log 2>&1 &
python status.py
```

한 GPU에서 두 개의 작은 학습 worker를 돌릴 때:

```bash
nohup python -u run_gpu.py --gpu-a 0 --gpu-b 0 --wait-for-gpu > launch.log 2>&1 &
```

동시에 모델 worker를 최대 2개 실행하며, CPU/BLAS/torch 스레드는 각 1개입니다.
지정한 GPU에서 여유 메모리 8GiB 이상, GPU utilization 30% 이하를 세 번 확인한
뒤 시작합니다. 다른 사용자의 프로세스를 종료하거나 메모리를 선점하지 않습니다.
GPU 인덱스는 CUDA 논리 인덱스이며 숫자로 지정한 CUDA_VISIBLE_DEVICES도 지원합니다.
UUID/MIG mask는 현재 availability check에서 지원하지 않습니다.

같은 디렉터리에 두 번 launch할 수 없습니다. 이미 시작한 모델을 다른 서버에서
이어 학습하는 checkpoint resume 기능도 제공하지 않습니다. 다른 서버의 실행은
**독립된 새 45회 panel**이며, 결과를 보고 좋은 서버의 결과만 선택하지 마십시오.
같은 공유 파일시스템에서 중복 실행하는 용도로 이 명령을 사용하지 마십시오.

실행 순서는 자동으로 이어집니다.

1. 입력 hash 및 T/R/U/NoOp 공통 조건 검증과 분석 코드 고정.
2. 45회 validation 선택 학습. 선택 checkpoint와 hash를 저장한 뒤 test 추론.
3. 선택 checkpoint를 기존 100후보, 같은 후보의 ID tie breaking,
   전체 eligible case pool의 고정 target 및 모든 관측 positive target에서 평가.
4. 고정된 모든 대비와 setting-t interval, paired hierarchical bootstrap 수집.

예외가 나면 실패 기록을 보존하고 자동 재시도하지 않습니다. launch shell이
종료되어도 시작한 worker와 후속 평가 driver는 별도 process session에서 계속됩니다.

## 4. 결과와 진행 상황

```bash
python status.py
tail -n 8 www_short/wsdm_recenter_20261116/owned_r18_gpu_allocation_factorial/worker_gpu_a.log
tail -n 8 www_short/wsdm_recenter_20261116/owned_r18_gpu_allocation_factorial/worker_gpu_b.log
```

진행 보고는 개수와 실행 상태만 읽으며 중간 test 수치를 열지 않습니다.
모든 학습과 평가가 끝난 뒤 다음 파일을 확인합니다.

- `www_short/wsdm_recenter_20261116/owned_r18_gpu_allocation_factorial/PIPELINE_COMPLETE.json`
- 같은 폴더의 `run_v1/COLLECTED.json`, `run_v1/VALIDATION_POLICY.json`
- `run_v1/ALL45_GPU_FIRST_FITS.csv`, 각 fit의 `SELECTION.json`와 `selected.pt`

모든 강제 조건을 보고하고 validation 선택 정책에는 NoOp도 포함합니다.
R도 teacher-derived positive 및 공통 support/capacity를 유지하므로 전체 LLM 제거
대조가 아닙니다. R에서 teacher-positive인 쌍을 뽑아도 negative training target은
0입니다. 배분을 바꾸면 상대 warm 개체의 degree와 context 결합도 바뀔 수 있어,
전체 효과를 coverage만의 매개 효과로 해석하지 않습니다. 통계 단위는 shared
corpus의 3 settings이며 9 learning realizations를 9개의 독립 데이터셋처럼 세지 않습니다.

모든 source 및 입력은 준비 단계에서 hash로 고정됩니다. 실행 중 원본 코드를
바꾸거나 결과에 맞춰 재학습하지 마십시오. 전체 candidate 평가는 재학습이나
모델 재선택을 수행하지 않습니다. 원고와 과거 외부 리뷰는 이 업로드에 포함하지 않습니다.
