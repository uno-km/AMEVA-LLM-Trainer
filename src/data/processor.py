"""
src/data/processor.py
ChatML 변환 및 CPU 최적화 스트리밍 - IterableDataset 아키텍처
RAM 폭발 원천 차단: 하드디스크의 텍스트 토큰을 실시간으로 1개씩 읽어 피딩
"""
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
                    text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )

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
    """토크나이저 로드 및 IterableDataset 팩토리"""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return LLMIterableDataset(file_path, tokenizer, max_length)
