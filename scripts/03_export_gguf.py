"""
scripts/03_export_gguf.py
모델 병합 및 GGUF 변환/양자화 빌더

LoRA 어댑터 튜닝 완료 후, 원본 베이스 모델과 결합하여
llama.cpp 에코시스템과 호환되는 GGUF 4비트 양자화 모델을 내보낸다.

사용법:
  python scripts/03_export_gguf.py
  python scripts/03_export_gguf.py --task-id <UUID>
"""
import os
import sys
import argparse
import subprocess

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.core.config import CFG, LORA_DIR, MERGED_DIR, GGUF_DIR
from src.core.exceptions import exception_guard


@exception_guard("scripts.03_export_gguf.merge_and_save_gguf", reraise=True)
def merge_and_save_gguf(task_id: str = None):
    """LoRA 병합 → HF→GGUF 변환 → q4_0 양자화"""

    # 태스크별 경로 조정
    lora_dir = LORA_DIR
    merged_dir = MERGED_DIR
    gguf_dir = GGUF_DIR

    if task_id:
        base_output = os.path.join(project_root, "outputs", task_id)
        lora_dir = os.path.join(base_output, "lora_adapter")
        merged_dir = os.path.join(base_output, "merged_model")

    if not os.path.exists(lora_dir):
        print(f"[ERROR] LoRA 어댑터 경로를 찾을 수 없습니다: {lora_dir}")
        sys.exit(1)

    # 1. Weights Merger
    print("[INFO] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        CFG["model_id"],
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_id"])

    print("[INFO] Loading LoRA adapters & merging weights...")
    model = PeftModel.from_pretrained(base_model, lora_dir)
    merged_model = model.merge_and_unload()

    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"[SUCCESS] Merged model saved to: {merged_dir}")

    # 2. GGUF 변환 (third_party/llama.cpp 유틸리티 연동)
    print("[INFO] Converting to GGUF basic FP16 format...")
    os.makedirs(gguf_dir, exist_ok=True)

    raw_gguf_path = os.path.join(gguf_dir, "model-unquantized.gguf")

    convert_script = os.path.join(project_root, "third_party", "llama.cpp", "convert_hf_to_gguf.py")
    if os.path.exists(convert_script):
        cmd = ["python", convert_script, merged_dir, "--outfile", raw_gguf_path]
        subprocess.run(cmd, check=True)
        print(f"[SUCCESS] HF weights converted to unquantized GGUF at: {raw_gguf_path}")

        # 3. K-Quant 4비트 양자화 수행
        quantized_path = os.path.join(gguf_dir, "model-q4_0.gguf")
        quantize_bin = os.path.join(project_root, "third_party", "llama.cpp", "quantize")
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
        print("[INFO] You can manually convert using: python -m gguf.convert ...")


def main():
    parser = argparse.ArgumentParser(description="AMEVA-LLM-Trainer GGUF 변환")
    parser.add_argument("--task-id", default=None, help="태스크 ID (선택)")
    args = parser.parse_args()

    merge_and_save_gguf(args.task_id)


if __name__ == "__main__":
    main()
