"""
src/core/config.py
싱글톤 설정 로더 - HuggingFace 캐시 경로 로컬 고정 및 하이퍼파라미터 관리
"""
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
    "logging_steps": 10,
    "save_steps": 100,
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
