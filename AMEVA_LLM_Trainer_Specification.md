# AMEVA-LLM-Trainer: 로컬/오프라인 CPU 특화 소형 LLM Fine-tuning 파이프라인 구축 명세서

본 명세서는 **AMEVA-STT-Trainer**의 성공적인 Headless API + CLI 아키텍처를 계승하여, 인터넷이 단절된 환경에서도 $0.5\text{B} \sim 1.5\text{B}$급 초경량 거대 언어 모델(LLM)을 로컬 CPU 환경에서 안정적으로 파인튜닝하고 배포할 수 있는 **AMEVA-LLM-Trainer**의 구축 명세서이다.

다른 AI 개발 에이전트(Antigravity 등)가 본 명세서를 입력받아 프로젝트 초기화부터 최종 빌드까지 완벽히 수행할 수 있도록 설계 패턴, 폴더 구조, API 스펙, 핵심 소스코드 템플릿, 그리고 운영 스크립트까지 극단적으로 상세하게 기술한다.

---

## 1. 핵심 아키텍처 철학 (Core Philosophy)

1. **로컬라이징 (Localizing)**:
   - 모든 AI 모델 캐시 경로(`HF_HOME` 환경변수)를 `C:\ameva\models\llm`으로 고정하며, 모든 데이터셋과 체크포인트는 로컬 디스크 내에 보존한다. 외부 서버나 클라우드로의 민감 데이터 유출을 원천 방어한다.
2. **오프라인 환경 보장 (Offline Environment)**:
   - 초기 셋업(모델 다운로드) 단계 이후에는 랜선이 뽑힌 완전 폐쇄망 오프라인 환경에서도 데이터 정제, 토큰화 학습, 양자화 배포가 100% 정상 작동하도록 설계한다.
3. **기능 우선 중심 (Feature-first Focus)**:
   - 불필요한 GUI 렌더링 오버헤드를 원천 차단하고, CPU 환경에서 훈련 손실(Loss/Perplexity)을 모니터링하며, 정확한 Instruction-Response 템플릿 정렬 및 토큰화 패딩 등 연산 코어의 효율에 리소스를 집중한다.
4. **안정적인 구동 환경 (Stable Execution Environment)**:
   - CPU/RAM 16GB 한계 극복을 위해 메모리 사용량을 수평 고정하는 **IterableDataset Streaming**과 가상 메모리 스왑(Threashold) 오버플로우 방지 전략을 적용한다.
   - 예외 상황에서도 프로세스가 죽지 않는 **Exception Guard 데코레이터**와 싱글톤 설정 패턴을 강제한다.

---

## 2. 전체 디렉토리 구조 (Repository Layout)

새로 구축할 `AMEVA-LLM-Trainer` 프로젝트는 아래의 폴더 트리를 엄격히 준수한다.

```text
AMEVA-LLM-Trainer/
├── setup.py                # 루트 통합 크로스플랫폼 셋업 진입점 (OS 자동 라우터)
├── setup/                  # 셋업용 스크립트 격리 보관
│   ├── setup_env.ps1       # Windows PowerShell 셋업
│   └── setup_env.sh        # Unix/Linux/macOS Bash 셋업
├── requirements.txt        # 가볍게 정돈된 의존성 명세 (PyQt, FFmpeg 등 그래픽/오디오 제거)
├── run_server.bat          # FastAPI 백엔드 단독 실행기 (포트 8001)
├── run_cli.bat             # Premium CLI 클라이언트 단독 실행기
├── configs/
│   └── train_config.yaml   # 하이퍼파라미터 및 경로 설정
├── cli/                    # CLI 클라이언트 패키지
│   ├── __init__.py
│   ├── cli.py              # CLI 메인 진입점
│   ├── client/
│   │   ├── __init__.py
│   │   └── api_client.py   # REST API 통신 모듈 (AMEVA_API_URL 바인딩)
│   └── views/
│       ├── __init__.py
│       ├── tasks.py        # 태스크 목록 및 신규 태스크 생성 화면
│       ├── monitor.py      # plotext 기반 실시간 아스키 차트 모니터
│       ├── explorer.py     # 원격 텍스트 파일 브라우저
│       ├── db_view.py      # SQLite DB 테이블 뷰어
│       └── sysinfo.py      # 서버 시스템 리소스 뷰어
├── src/                    # 백엔드 엔진 패키지
│   ├── __init__.py
│   ├── backend/            # FastAPI 애플리케이션
│   │   ├── __init__.py
│   │   ├── main.py         # 백엔드 메인 엔트리
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   └── database.py # SQLite DB 커넥터 및 스키마 초기화
│   │   └── routers/
│   │       ├── __init__.py
│   │       ├── tasks.py    # 태스크 제어 라우터
│   │       └── system.py   # 시스템 리소스 조회 라우터
│   ├── core/               # 공통 코어
│   │   ├── __init__.py
│   │   ├── config.py       # 싱글톤 설정 제어 및 로컬 경로 고정
│   │   └── exceptions.py   # Exception Guard 데코레이터
│   ├── data/               # 데이터 엔지니어링 레이어
│   │   ├── __init__.py
│   │   └── processor.py    # ChatML 변환 및 IterableDataset 스트리밍 토크나이저
│   ├── models/             # AI 모델 레이어
│   │   ├── __init__.py
│   │   └── loader.py       # AutoModelForCausalLM + PEFT LoRA 설정
│   └── training/           # 학습 제어 레이어
│       ├── __init__.py
│       ├── trainer.py      # Transformers SFTTrainer 구동 스크립트
│       └── callbacks.py    # 학습 지표 실시간 DB/Log 기록용 커스텀 콜백
├── scripts/                # 실행 가능한 커맨드라인 엔트리
│   ├── 01_prepare_data.py  # 학습용 raw 텍스트 파일 검수 및 정제
│   ├── 02_run_tuning.py    # 백중단 백그라운드 학습 스크립트 (FastAPI 스폰 대상)
│   └── 03_export_gguf.py   # 모델 병합 및 GGUF 변환/양자화 빌더
├── dataset/                # 정제된 JSON/TXT 데이터 보관
├── outputs/                # 학습 결과물 (LoRA 가중치, 병합 모델) 저장
└── logs/                   # 실행 로그
```

---

## 3. 설정 관리 및 에러 가드 템플릿 (src/core/)

### 3.1. 싱글톤 설정 로더 (`src/core/config.py`)
HuggingFace 캐시 경로인 `HF_HOME`을 물리적 외부 로컬 공통 디렉토리(`C:\ameva\models\llm`)로 강제하여 완전한 오프라인 고립을 달성한다.

```python
import os
import yaml

# 프로젝트 루트 결정
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 로컬 고립형 HuggingFace 캐시 강제 고정
os.environ["HF_HOME"] = r"C:\ameva\models\llm"

# 태스크 전용 출력 경로 관리
ACTIVE_TASK_ID = os.environ.get("CURRENT_TASK_ID")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")

OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
LORA_DIR = os.path.join(OUTPUTS_DIR, "lora_adapter")
MERGED_DIR = os.path.join(OUTPUTS_DIR, "merged_model")

if ACTIVE_TASK_ID:
    OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs", ACTIVE_TASK_ID)
    LORA_DIR = os.path.join(OUTPUTS_DIR, "lora_adapter")
    MERGED_DIR = os.path.join(OUTPUTS_DIR, "merged_model")

GGUF_DIR = os.path.join(ROOT_DIR, "models", "gguf")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "train_config.yaml")

# 기본 하이퍼파라미터 정의 (0.5B ~ 1.5B CPU LoRA 특화 처방)
DEFAULTS = {
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",  # 디폴트 타겟 모델
    "learning_rate": 1e-4,                      # 과적합 및 가중치 소실 방지 비율
    "max_steps": 500,
    "batch_size": 1,                            # CPU 16GB 환경을 위한 극단적 1배치 고정
    "gradient_accumulation_steps": 8,           # 실질 배치 8 확보
    "max_seq_length": 512,                      # CPU VRAM 스왑 방지를 위한 단축 문맥 규격
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
}

def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return DEFAULTS.copy()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = DEFAULTS.copy()
        cfg.update(user_cfg)
        return cfg
    except Exception:
        return DEFAULTS.copy()

CFG = load_config()
```

### 3.2. 글로벌 데코레이터 예외 가드 (`src/core/exceptions.py`)
메모리 부족(OOM)이나 파일 시스템 차단 등 어떠한 예외 상황이 발생해도 시스템 크래시로 이어지지 않게 로그를 안전하게 적재하고 복구를 시도한다.

```python
import functools
import logging
import os
import traceback
from datetime import datetime

# 로그 파일 핸들러 생성
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "..", "logs"), exist_ok=True)
log_file = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "exception.log")
logging.basicConfig(filename=log_file, level=logging.ERROR,
                    format="%(asctime)s [%(levelname)s] (%(location)s) %(message)s")

def log_exception(e: Exception, location: str):
    tb = traceback.format_exc()
    logging.error(f"Error: {str(e)}\nTraceback:\n{tb}", extra={"location": location})

def exception_guard(location: str = None, reraise: bool = False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            loc = location or f"{func.__module__}.{func.__name__}()"
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_exception(e, loc)
                if reraise:
                    raise
                return None
        return wrapper
    return decorator
```

---

## 4. 데이터셋 구축 및 토큰화 스트리밍 (`src/data/`)

### 4.1. ChatML 변환 및 CPU 최적화 스트리밍 (`src/data/processor.py`)
CPU 학습 시 RAM 폭발을 원천 차단하기 위해 **IterableDataset** 아키텍처를 도입하여 하드디스크의 텍스트 토큰을 실시간으로 1개씩 읽어 피딩하는 구조를 갖춘다.

```python
import json
import os
import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from src.core.config import CFG
from src.core.exceptions import exception_guard

class LLMIterableDataset(IterableDataset):
    """
    대규모 데이터를 메모리에 적재하지 않고, 파일 한 줄씩 실시간 토큰화하여 반환하는
    Flat-Memory 스트리밍 데이터셋.
    """
    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_length: int = 512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found at: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # ChatML 형식 준수: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
                    messages = data.get("messages", [])
                    if not messages:
                        continue

                    # 토크나이저 템플릿 적용 (허깅페이스 자동 포맷팅)
                    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    
                    encodings = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )

                    input_ids = encodings["input_ids"].squeeze(0)
                    attention_mask = encodings["attention_mask"].squeeze(0)
                    
                    # CausalLM 학습을 위해 label은 input_ids를 그대로 복제
                    labels = input_ids.clone()
                    
                    # 패딩 토큰은 학습 Loss 계산 시 제외하기 위해 -100 처리
                    labels[labels == self.tokenizer.pad_token_id] = -100

                    yield {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    }
                except Exception as e:
                    # 손상된 개별 라인은 로깅 후 무시하고 계속 진행
                    continue

@exception_guard("src.data.processor.prepare_dataset")
def get_dataset_generator(file_path: str, model_id: str, max_length: int = 512) -> LLMIterableDataset:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return LLMIterableDataset(file_path, tokenizer, max_length)
```

---

## 5. 학습 엔진 및 로더 (`src/models/`, `src/training/`)

### 5.1. 가중치 모델 로더 (`src/models/loader.py`)
CPU 환경에서 양자화 옵티마이저가 불가함을 명확히 전제하여 FP32로 정밀하게 베이스 가중치를 고정 로드하고 LoRA 어댑터를 주입한다.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from src.core.config import CFG
from src.core.exceptions import exception_guard

@exception_guard("src.models.loader.load_peft_model", reraise=True)
def load_peft_model(model_id: str):
    # CPU 학습에서의 가용성 극대화를 위해 FP32(float32)로 강제 정밀도 설정
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    lora_config = LoraConfig(
        r=CFG["lora_r"],
        lora_alpha=CFG["lora_alpha"],
        target_modules=CFG["lora_target_modules"],
        lora_dropout=CFG["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer
```

### 5.2. 트레이너 엔진 구동 (`src/training/trainer.py`)
`transformers.Trainer`를 래핑하여 CPU 환경에 맞춤화된 학습 프로세스를 구동하고, 백엔드 DB에 메트릭을 동기화하기 위한 콜백을 부착한다.

```python
import os
import sys
from transformers import Trainer, TrainingArguments
from src.core.config import CFG
from src.core.exceptions import exception_guard
from src.models.loader import load_peft_model
from src.data.processor import get_dataset_generator
from src.training.callbacks import DBProgressCallback

@exception_guard("src.training.trainer.run_fine_tuning", reraise=True)
def run_fine_tuning(task_id: str, data_path: str):
    model_id = CFG["model_id"]
    output_dir = os.path.join(CFG["OUTPUTS_DIR"], "checkpoints")
    
    # 모델 및 토크나이저 로드
    model, tokenizer = load_peft_model(model_id)
    
    # 스트리밍 데이터셋 로드
    dataset = get_dataset_generator(data_path, model_id, CFG["max_seq_length"])
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=CFG["max_steps"],
        per_device_train_batch_size=CFG["batch_size"],
        gradient_accumulation_steps=CFG["gradient_accumulation_steps"],
        learning_rate=CFG["learning_rate"],
        logging_steps=10,
        save_steps=100,
        fp16=False,               # CPU 환경이므로 반정밀도 가속 차단
        bf16=False,
        use_cpu=True,             # CPU 강제 설정
        report_to="none",
        dataloader_pin_memory=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[DBProgressCallback(task_id)]
    )
    
    trainer.train()
    
    # 학습 완료 후 최종 LoRA 어댑터 저장
    lora_save_path = CFG["LORA_DIR"]
    model.save_pretrained(lora_save_path)
    tokenizer.save_pretrained(lora_save_path)
    print(f"LoRA Adapter saved successfully at: {lora_save_path}")
```

### 5.3. 실시간 DB 트래킹 커스텀 콜백 (`src/training/callbacks.py`)
학습 진행 중 매 Step마다 Loss 및 Step 카운터를 SQLite DB에 기록하여 CLI 클라이언트에서 실시간 플로팅 및 모니터링을 가능하게 한다.

```python
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from src.backend.core.database import db_manager

class DBProgressCallback(TrainerCallback):
    def __init__(self, task_id: str):
        self.task_id = task_id

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss", None)
            step = state.global_step
            max_steps = state.max_steps
            progress = (step / max_steps) * 100 if max_steps > 0 else 0
            
            if loss is not None:
                # DB의 태스크 메트릭 테이블 및 작업 진행 상태 업데이트
                db_manager.update_task_progress(self.task_id, progress, status="RUNNING")
                db_manager.insert_log_metric(self.task_id, step, loss)
```

---

## 6. 백엔드 데이터베이스 및 API 서버 명세 (`src/backend/`)

### 6.1. SQLite 스키마 설계 (`src/backend/core/database.py`)
로컬 훈련의 텔레메트리 관리를 위한 가벼운 로컬 SQLite 데이터베이스를 생성한다.

```python
import sqlite3
import os

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs", "ameva_llm.db"))

class DatabaseManager:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        with self.conn:
            # 태스크 관리 테이블
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    tsk_nm TEXT NOT NULL,
                    status TEXT NOT NULL,         -- WAITING, RUNNING, COMPLETED, FAILED
                    progress REAL DEFAULT 0.0,
                    start_time TEXT,
                    end_time TEXT,
                    data_path TEXT
                )
            """)
            # 실시간 손실율 트래킹 테이블
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    step INTEGER,
                    loss REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                )
            """)

    def insert_task(self, task_id: str, name: str, data_path: str):
        with self.conn:
            self.conn.execute(
                "INSERT INTO tasks (task_id, tsk_nm, status, data_path) VALUES (?, ?, 'WAITING', ?)",
                (task_id, name, data_path)
            )

    def update_task_progress(self, task_id: str, progress: float, status: str):
        with self.conn:
            self.conn.execute(
                "UPDATE tasks SET progress = ?, status = ? WHERE task_id = ?",
                (task_id, status, task_id)
            )

    def insert_log_metric(self, task_id: str, step: int, loss: float):
        with self.conn:
            self.conn.execute(
                "INSERT INTO metrics (task_id, step, loss) VALUES (?, ?, ?)",
                (task_id, step, loss)
            )

    def get_metrics(self, task_id: str):
        cursor = self.conn.cursor()
        cursor.execute("SELECT step, loss FROM metrics WHERE task_id = ? ORDER BY step ASC", (task_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_all_tasks(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tasks ORDER BY start_time DESC")
        return [dict(row) for row in cursor.fetchall()]

    def get_task_details(self, task_id: str):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

db_manager = DatabaseManager()
```

### 6.2. FastAPI 엔트리포인트 (`src/backend/main.py`)
통합 통제 플레인을 지원하는 비동기 API 엔드포인트 명세이다.

```python
import subprocess
import uuid
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import psutil
import os

from src.backend.core.database import db_manager
from src.core.config import ROOT_DIR

app = FastAPI(title="AMEVA-LLM-Trainer Headless API Server", version="1.0.0")

class TaskCreate(BaseModel):
    name: str
    data_path: str

@app.get("/")
def read_root():
    return {"status": "healthy", "service": "AMEVA-LLM-Trainer API"}

@app.post("/tasks")
def create_task(task_in: TaskCreate, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    db_manager.insert_task(task_id, task_in.name, task_in.data_path)
    
    # 02_run_tuning.py 스크립트를 백그라운드 서브프로세스로 안전 격리 격벽 호출 실행
    def launch_training():
        db_manager.update_task_progress(task_id, 0.0, "RUNNING")
        cmd = [
            os.path.join(ROOT_DIR, "venv", "Scripts", "python.exe"),
            os.path.join(ROOT_DIR, "scripts", "02_run_tuning.py"),
            "--task-id", task_id,
            "--data-path", task_in.data_path
        ]
        # 환경변수 상속 및 현재 구동 컨텍스트 전달
        env = os.environ.copy()
        env["CURRENT_TASK_ID"] = task_id
        
        try:
            subprocess.run(cmd, env=env, check=True)
            db_manager.update_task_progress(task_id, 100.0, "COMPLETED")
        except Exception:
            db_manager.update_task_progress(task_id, 0.0, "FAILED")

    background_tasks.add_task(launch_training)
    return {"task_id": task_id, "status": "WAITING"}

@app.get("/tasks")
def list_tasks():
    return db_manager.get_all_tasks()

@app.get("/tasks/{task_id}/metrics")
def get_task_metrics(task_id: str):
    return db_manager.get_metrics(task_id)

@app.get("/sysinfo")
def get_system_info():
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_available_gb": psutil.virtual_memory().available / (1024**3),
        "disk_free_gb": psutil.disk_usage("/").free / (1024**3)
    }
```

---

## 7. CLI 클라이언트 및 모니터 (`cli/`)

### 7.1. API 클라이언트 (`cli/client/api_client.py`)
```python
import os
import requests

class APIClient:
    def __init__(self):
        # AMEVA-STT-Trainer(8000) 포트와 충돌하지 않도록 디폴트 8001로 격리
        self.base_url = os.environ.get("AMEVA_API_URL", "http://127.0.0.1:8001")

    def post(self, path: str, json_data: dict = None):
        try:
            res = requests.post(f"{self.base_url}{path}", json=json_data, timeout=5.0)
            return res.json()
        except Exception as e:
            return {"error": str(e)}

    def get(self, path: str, params: dict = None):
        try:
            res = requests.get(f"{self.base_url}{path}", params=params, timeout=5.0)
            return res.json()
        except Exception as e:
            return {"error": str(e)}

    def check_health(self) -> bool:
        try:
            res = requests.get(self.base_url, timeout=2.0)
            return res.status_code == 200
        except Exception:
            return False

api_client = APIClient()
```

### 7.2. plotext 기반 실시간 아스키 차트 모니터 (`cli/views/monitor.py`)
UI 대시보드 없이도 터미널 내에서 그래프를 그릴 수 있는 `plotext`를 사용하여 실시간으로 Loss 감소 추이를 시각화한다.

```python
import time
import os
import plotext as plt
from rich.console import Console
from cli.client.api_client import api_client

console = Console()

def watch_logs(task_id: str):
    console.clear()
    console.print(f"[bold cyan]태스크 모니터 가동 (ID: {task_id})[/bold cyan]\n")
    
    while True:
        try:
            metrics = api_client.get(f"/tasks/{task_id}/metrics")
            if not metrics or "error" in metrics:
                time.sleep(2)
                continue
            
            steps = [m["step"] for m in metrics]
            losses = [m["loss"] for m in metrics]
            
            if len(steps) > 0:
                plt.clear_data()
                plt.clear_terminal()
                # 아스키 플롯 속성 빌드
                plt.plot(steps, losses, label="Train Loss", color="cyan")
                plt.title("Real-Time Training Loss (CPU LoRA)")
                plt.xlabel("Steps")
                plt.ylabel("Loss")
                plt.theme("dark")
                plt.show()
            else:
                console.print("[yellow]대기 중: 메트릭 수집을 기다리는 중...[/yellow]")
                
            time.sleep(5)
        except KeyboardInterrupt:
            console.print("\n[bold green]실시간 모니터링 종료.[/bold green]")
            break
```

---

## 8. 모델 병합 및 GGUF 변환 최적화 (`scripts/03_export_gguf.py`)

LoRA 어댑터 튜닝이 끝나면, 원본 뼈대 모델과 결합하여 `llama.cpp` 에코시스템과 즉각 호환되는 **GGUF 4비트 양자화 모델**을 내보낸다.

```python
import os
import argparse
import subprocess
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from src.core.config import CFG

def merge_and_save_gguf():
    # 1. Weights Merger
    print("[INFO] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        CFG["model_id"],
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_id"])
    
    print("[INFO] Loading LoRA adapters & merging weights...")
    # CFG["LORA_DIR"]로부터 LoRA 가중치 결합
    model = PeftModel.from_pretrained(base_model, CFG["LORA_DIR"])
    merged_model = model.merge_and_unload()
    
    merged_dir = CFG["MERGED_DIR"]
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"[SUCCESS] Merged model saved to: {merged_dir}")
    
    # 2. GGUF 변환 (third_party/llama.cpp 유틸리티 연동 가정)
    # llama.cpp의 convert_hf_to_gguf.py 또는 convert.py를 호출하여 ggml 포맷 변환 진행
    print("[INFO] Converting to GGUF basic FP16 format...")
    gguf_output_dir = CFG["GGUF_DIR"]
    os.makedirs(gguf_output_dir, exist_ok=True)
    
    raw_gguf_path = os.path.join(gguf_output_dir, "model-unquantized.gguf")
    
    # 레포에 내장되거나 설치된 llama.cpp의 변환용 python 스크립트 실행
    # (일반적으로 llama.cpp 패키지 내 python 스크립트를 subprocess로 구동)
    convert_script = os.path.join("third_party", "llama.cpp", "convert_hf_to_gguf.py")
    if os.path.exists(convert_script):
        cmd = ["python", convert_script, merged_dir, "--outfile", raw_gguf_path]
        subprocess.run(cmd, check=True)
        print(f"[SUCCESS] HF weights converted to unquantized GGUF at: {raw_gguf_path}")
        
        # 3. K-Quant 4비트 양자화 수행
        quantized_path = os.path.join(gguf_output_dir, "model-q4_0.gguf")
        quantize_bin = os.path.join("third_party", "llama.cpp", "quantize")
        if os.name == "nt":
            quantize_bin += ".exe"
            
        if os.path.exists(quantize_bin):
            print("[INFO] Run K-Quantization (q4_0) via llama.cpp quantize utility...")
            q_cmd = [quantize_bin, raw_gguf_path, quantized_path, "q4_0"]
            subprocess.run(q_cmd, check=True)
            print(f"[SUCCESS] Quantized GGUF model exported to: {quantized_path}")
        else:
            print("[WARN] llama.cpp quantize binary not found. Please compile it via setup.py.")
    else:
        print("[WARN] convert_hf_to_gguf.py script not found. GGUF conversion skipped.")

if __name__ == "__main__":
    merge_and_save_gguf()
```

---

## 9. 자동 구동 및 자율 왓치독 스크립트 (Launcher)

### 9.1. 백엔드 상태 감지 및 자동 기동 런처 (`run_cli.bat`)
서버가 먼저 실행되어 있지 않더라도, CLI 구동 시 스스로 Uvicorn 포트의 상태를 점검하여 꺼진 백엔드 서버를 백그라운드로 띄워주는 안정성 부트스트랩 스크립트이다.

```batch
@echo off
echo ======================================================================
echo AMEVA-LLM-Trainer Premium CLI Launcher
echo ======================================================================

IF NOT EXIST "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Please run setup.py first.
    pause
    exit /b
)

echo [INFO] Checking if backend API server is running on port 8001...
netstat -ano | find "8001" >nul
if %errorlevel% neq 0 (
    echo [WARN] API server is not running on 8001. Automatically starting the server...
    start "AMEVA LLM API Server" run_server.bat
    echo [INFO] Waiting 5 seconds for the server to initialize...
    timeout /t 5 /nobreak >nul
) else (
    echo [INFO] API server on 8001 is already running!
)

echo Starting AMEVA-LLM-Trainer CLI...
venv\Scripts\python.exe cli\cli.py
pause
```

---

## 10. 설치 자동화 구성 (`setup.py` & `requirements.txt`)

### 10.1. 라이트급 의존성 명세 (`requirements.txt`)
메모리를 크게 차지하는 GUI 모듈(`PyQt6`, `matplotlib`)과 오디오 모듈(`librosa`, `soundfile`, `pydub`, `scipy`)을 완전히 배제하고, 순수 CLI 연산 및 LLM MLOps에 필수적인 패키지 위주로 고립화한다.

```text
# Core AI Stack
torch
transformers
peft
accelerate
datasets
sentencepiece
numpy

# SQLite & Backend API
fastapi
uvicorn
requests
psutil
plotext

# MLOps Telemetry
pandas
pyyaml
tqdm
rich
win10toast
gguf
python-docx
```

### 10.2. 크로스플랫폼 통합 빌더 (`setup.py`)
사용자 OS 환경을 감지하여 가상환경 생성, 의존성 설치, 그리고 C++ `llama.cpp` 에코시스템 빌드까지 논스톱으로 수행하는 단일 셋업 진입점이다.

```python
import os
import sys
import subprocess
import shutil

def run_command(cmd, shell=False):
    print(f"[EXEC] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    subprocess.run(cmd, shell=shell, check=True)

def main():
    print("======================================================================")
    print(" AMEVA-LLM-Trainer Unified Setup Launcher")
    print("======================================================================")
    
    # 1. 가상환경 생성
    if not os.path.exists("venv"):
        print("[INFO] Creating virtual environment (venv)...")
        run_command([sys.executable, "-m", "venv", "venv"])
    
    # 2. pip 업그레이드 및 의존성 패키지 설치
    print("[INFO] Installing python dependencies...")
    pip_path = os.path.join("venv", "Scripts", "pip.exe") if os.name == "nt" else os.path.join("venv", "bin", "pip")
    run_command([pip_path, "install", "--upgrade", "pip"])
    run_command([pip_path, "install", "-r", "requirements.txt"])
    
    # 3. llama.cpp 서드파티 빌드 프로세스 (GGUF 양자화용)
    llama_dir = os.path.join("third_party", "llama.cpp")
    if os.path.exists(llama_dir):
        print("[INFO] Compiling llama.cpp binaries...")
        # Windows 환경: cmake 또는 빌드 유틸리티 확인
        if os.name == "nt":
            # Windows용 사전 빌드된 바이너리 체크 혹은 C++ make 컴파일러 부재 시 안내 우회
            print("[INFO] Windows detected. Local setup will check precompiled binaries.")
        else:
            # macOS / Linux 환경: Makefile make 빌드 가동
            print("[INFO] Unix-like OS detected. Starting native compile...")
            run_command(["make", "-C", llama_dir])
            
    print("\n[SUCCESS] Setup process completed successfully!")
    print("To run the application, execute: run_cli.bat")

if __name__ == "__main__":
    main()
```

---

이 명세서는 **AMEVA-LLM-Trainer**의 완벽한 청사진입니다. 이 파일을 읽는 다른 AI 에이전트는 본 문서의 파일 구조와 코드 구성을 토대로, STT 아키텍처의 견고함을 훼손하지 않으면서 최고 성능의 로컬 오프라인 LLM 파인튜너 시스템을 지체 없이 조립해 낼 것입니다.
