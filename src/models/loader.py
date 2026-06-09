"""
src/models/loader.py
가중치 모델 로더 - CPU FP32 정밀 로드 + LoRA 어댑터 주입
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from src.core.config import CFG
from src.core.exceptions import exception_guard


@exception_guard("src.models.loader.load_peft_model", reraise=True)
def load_peft_model(model_id: str = None):
    """
    CPU 학습에서의 가용성 극대화를 위해 FP32(float32)로 강제 정밀도 설정.
    LoRA 어댑터를 주입하여 학습 가능 파라미터만 업데이트.
    """
    if model_id is None:
        model_id = CFG["model_id"]

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
