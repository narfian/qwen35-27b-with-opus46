# qwen35-27b-with-opus46

`Qwen 3.5 27B` 모델을 **Unsloth + LoRA (4bit)** 로 SFT 파인튜닝하기 위한 저장소입니다.
Colab 노트북 (`Qwopus3-5-27b-Colab.ipynb`) 의 파이프라인을 **`uv run` 기반 CLI 프로젝트**로 재구성했습니다.

주요 구성 요소:

- **모델**: `unsloth/Qwen3.5-27B` (4bit 양자화 로드)
- **어댑터**: LoRA (rank=64, α=64)
- **데이터 믹스** (노트북과 동일):
  - `nohurry/Opus-4.6-Reasoning-3000x-filtered` – 3,900개
  - `Jackrong/Qwen3.5-reasoning-700x` – 700개
  - `Roman1111111/claude-opus-4.6-10000x` – 9,633개
- **채팅 템플릿**: `qwen3-thinking` (`<think>...</think>` reasoning 포맷)
- **추적**: Weights & Biases (선택)

---

## 1. 사전 준비

- Python 3.10 ~ 3.12
- NVIDIA GPU + 최신 CUDA 드라이버 (27B + 4bit 로드 기준 24GB↑ VRAM 권장, 32K 컨텍스트라면 더 필요)
- [uv](https://docs.astral.sh/uv/) 설치:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## 2. 저장소 클론 & 의존성 설치

```bash
git clone <this-repo>.git
cd qwen35-27b-with-opus46

uv sync
```

`uv sync` 는 `pyproject.toml` 의 의존성(`torch`, `unsloth`, `trl`, `transformers`, `datasets`, …)을 `.venv/` 에 설치합니다. Unsloth 관련 패키지는 최신 기능이 필요하므로 GitHub 소스에서 직접 가져오도록 `[tool.uv.sources]` 에 지정되어 있습니다.

### 2-a. (권장) 빠른 CUDA 커널 설치 — `--extra flash`

Qwen 3.5 는 하이브리드 아키텍처(attention + SSM 레이어)이기 때문에
[`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) /
[`causal_conv1d`](https://github.com/Dao-AILab/causal-conv1d) 가 없으면 순수 torch 경로로
**fallback** 됩니다 (학습/추론 모두 수 배 느려짐). 설치는 `torch` 가 이미 깔린 뒤에 해야 하므로 2-step 입니다.

```bash
uv sync                  # torch / unsloth / trl 등 기본 환경 구축
uv sync --extra flash    # torch 위에서 causal_conv1d + FLA 커널 빌드
```

`pyproject.toml` 의 `[tool.uv].no-build-isolation-package` 설정 덕분에 `--no-build-isolation`
플래그를 손으로 넘기지 않아도 uv 가 알아서 격리 없이 빌드합니다.

빌드가 실패하면 대부분 다음 중 하나입니다.

- `nvcc` 가 PATH 에 없음 → `nvcc --version` 확인, 필요 시 해당 CUDA toolkit 설치
- 빌드 툴 부재 → `sudo apt install -y build-essential ninja-build`
- wheel 을 못 찾는 경우 → `CAUSAL_CONV1D_FORCE_BUILD=TRUE uv sync --extra flash`

확인:

```bash
uv run python -c "import causal_conv1d, fla; print(causal_conv1d.__version__, fla.__version__)"
```

## 3. 비밀 토큰 설정 (git 에 올라가지 않음)

1. `.env.example` 을 복사해 `.env` 파일을 만듭니다.

   ```bash
   cp .env.example .env
   ```

2. `.env` 에 실제 토큰을 채워 넣습니다.

   ```dotenv
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

3. `.env` 는 `.gitignore` 에 포함되어 있어 저장소에 **절대 커밋되지 않습니다**. 팀원과 공유할 때는 반드시 별도 채널(비밀번호 관리자, 사내 secrets 저장소 등)을 사용하세요.

- `HF_TOKEN`: 모델을 Hugging Face Hub로 업로드할 때 필요합니다 (게이티드 모델을 받을 때도).
- `WANDB_API_KEY`: `SFTConfig.report_to == "wandb"` 일 때 로그인에 사용됩니다. W&B 를 쓰지 않으려면 `--train-report-to none` 으로 끄면 됩니다.

`.env` 대신 셸 환경 변수로 export 해도 동일하게 동작합니다 (`os.environ` 이 우선).

## 4. 사용법

모든 명령은 `uv run` 으로 실행합니다. `uv run` 이 자동으로 가상환경을 구성/활성화합니다.

### 데이터 파이프라인만 먼저 점검

GPU 없이도 돌릴 수 있는지 확인하기 위한 dry-run 용도로 사용합니다 (모델/토크나이저는 로드됨).

```bash
uv run qwen-finetune prepare-data
```

### 학습 실행

```bash
uv run qwen-finetune train
```

기본값 외의 프리셋을 쓰려면 `--config-preset`을 지정합니다:

```bash
uv run qwen-finetune train --config-preset q35-9b-opus
```

주요 옵션은 CLI 플래그로도 덮어쓸 수 있습니다:

```bash
uv run qwen-finetune train \
  --config-preset q35-27b-opus \
  --train-output-dir ./checkpoints/qwen35-27b-run1 \
  --train-num-train-epochs 2 \
  --train-per-device-train-batch-size 6 \
  --train-gradient-accumulation-steps 6 \
  --train-learning-rate 2e-4 \
  --train-report-to wandb
```

W&B 를 끄고 싶다면:

```bash
uv run qwen-finetune train --train-report-to none
```

학습이 끝나면 LoRA 어댑터가 프리셋의 `train.lora_save_dir` 에 저장됩니다 (`--train-lora-save-dir` 로 변경 가능).

### Hugging Face Hub 로 푸시

- **Merged 16bit 모델**:

  ```bash
  uv run qwen-finetune push-merged
  ```

  → `https://huggingface.co/<your-username>/Qwen3.5-27B-opus`

- **GGUF (q4_k_m / q8_0 / bf16)**:

  ```bash
  uv run qwen-finetune push-gguf
  ```

  → `https://huggingface.co/<your-username>/Qwen3.5-27B-opus-GGUF`

레포 이름을 바꾸려면 `--push-merged-repo-suffix`, `--push-gguf-repo-suffix` 플래그를 사용하세요.
학습 때 사용한 프리셋과 같은 값을 지정하면 해당 LoRA 경로와 Hub suffix를 그대로 사용합니다.

## 5. 프로젝트 구조

```
.
├── pyproject.toml                # uv 프로젝트 정의 (의존성 + 스크립트)
├── .env.example                  # 토큰 템플릿 (실제 값은 .env 에)
├── .gitignore                    # .env / 체크포인트 / wandb 등 제외
├── Qwopus3-5-27b-Colab.ipynb     # 원본 Colab 노트북 (참고용)
└── src/qwen_finetune/
    ├── __init__.py
    ├── cli.py                    # `qwen-finetune` CLI 엔트리
    ├── config.py                 # 공통 config dataclass + preset loader
    ├── config_presets/           # default / 모델별 preset / template
    ├── data.py                   # HF 데이터셋 로드·정규화·필터링
    ├── model.py                  # Unsloth 모델 + LoRA 로드
    ├── train.py                  # SFTTrainer 실행
    ├── push.py                   # push_to_hub_merged / push_to_hub_gguf
    └── secrets.py                # .env 로드 & 토큰 조회 헬퍼
```

## 6. 커스터마이징

기본값은 Colab 노트북과 동일합니다. 영구적인 변경이나 모델별 조합은 `src/qwen_finetune/config_presets/` 에 새 preset 파일을 추가해 관리하고, 모든 수정 가능 항목을 한 번에 보고 싶으면 `template.py` 를 복사해 시작하면 됩니다. 1회성 실험은 CLI 플래그로 조절하면 됩니다. 예:

- 다른 베이스 모델 사용: `--model-model-name unsloth/Qwen3-14B-unsloth-bnb-4bit`
- 더 짧은 학습: `--train-max-steps 200`
- 8bit 로드: `--model-load-in-4bit false --model-load-in-8bit true`

## 7. 라이선스 / 크레딧

원본 파이프라인 저자: [Jackrong](https://huggingface.co/Jackrong) — *Jackrong-llm-finetuning-guide*. 이 저장소는 해당 노트북을 재구성한 러너이며, 모델/데이터셋 라이선스는 각 원본 소스를 따릅니다.
