"""
src/training/trainer.py
트레이너 엔진 구동 - Transformers Trainer 래핑, CPU 맞춤 TrainingArguments
"""
import os
import sys
from transformers import Trainer, TrainingArguments
from src.core.config import CFG, OUTPUTS_DIR, LORA_DIR
from src.core.exceptions import exception_guard
from src.models.loader import load_peft_model
from src.data.processor import get_dataset_generator
from src.training.callbacks import DBProgressCallback


@exception_guard("src.training.trainer.run_fine_tuning", reraise=True)
def run_fine_tuning(task_id: str, data_path: str):
    """
    CPU 환경에 맞춤화된 학습 프로세스를 구동하고,
    백엔드 DB에 메트릭을 동기화하기 위한 콜백을 부착한다.
    """
    model_id = CFG["model_id"]
    output_dir = os.path.join(OUTPUTS_DIR, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

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
        logging_steps=CFG["logging_steps"],
        save_steps=CFG["save_steps"],
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
    lora_save_path = LORA_DIR
    os.makedirs(lora_save_path, exist_ok=True)
    model.save_pretrained(lora_save_path)
    tokenizer.save_pretrained(lora_save_path)
    print(f"LoRA Adapter saved successfully at: {lora_save_path}")
