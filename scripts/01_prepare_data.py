"""
scripts/01_prepare_data.py
학습용 raw 텍스트/JSON 파일 검수 및 ChatML JSONL 형식으로 정제

사용법:
  python scripts/01_prepare_data.py --input raw_data.txt --output dataset/train.jsonl

입력 형식 지원:
  1. 일반 텍스트 (.txt) - 각 줄을 instruction으로 변환
  2. JSON/JSONL (.json/.jsonl) - ChatML 형식 검증 및 정제
"""
import os
import sys
import json
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.config import DATASET_DIR
from src.core.exceptions import exception_guard


@exception_guard("scripts.01_prepare_data.validate_chatml_line")
def validate_chatml_line(line: str) -> dict:
    """한 줄의 JSON을 파싱하여 ChatML 형식 검증"""
    data = json.loads(line.strip())
    messages = data.get("messages", [])

    if not messages:
        return None

    # 최소한 user + assistant 쌍이 있어야 함
    roles = [m.get("role") for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return None

    # 각 메시지에 content가 비어있지 않은지 확인
    for msg in messages:
        if not msg.get("content", "").strip():
            return None

    return data


@exception_guard("scripts.01_prepare_data.convert_txt_to_chatml")
def convert_txt_to_chatml(input_path: str, output_path: str):
    """일반 텍스트 파일을 ChatML JSONL로 변환"""
    converted = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        lines = fin.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # Q: ... A: ... 패턴 감지
            if line.startswith("Q:") or line.startswith("질문:"):
                question = line.split(":", 1)[1].strip()
                # 다음 줄에서 답변 찾기
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith("A:") or next_line.startswith("답변:"):
                        answer = next_line.split(":", 1)[1].strip()
                        if question and answer:
                            entry = {
                                "messages": [
                                    {"role": "user", "content": question},
                                    {"role": "assistant", "content": answer}
                                ]
                            }
                            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            converted += 1
                            i += 2
                            continue

            # 단일 줄 → 시스템 지식으로 변환
            if len(line) > 10:
                entry = {
                    "messages": [
                        {"role": "user", "content": "다음 내용을 설명해주세요."},
                        {"role": "assistant", "content": line}
                    ]
                }
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                converted += 1
            else:
                skipped += 1

            i += 1

    print(f"[SUCCESS] TXT → ChatML 변환 완료: {converted}개 변환, {skipped}개 스킵")
    return converted


@exception_guard("scripts.01_prepare_data.validate_jsonl")
def validate_jsonl(input_path: str, output_path: str):
    """기존 JSONL 파일의 ChatML 형식 검증 및 정제"""
    valid = 0
    invalid = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, 1):
            if not line.strip():
                continue

            result = validate_chatml_line(line)
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                valid += 1
            else:
                invalid += 1
                print(f"  [WARN] Line {line_num}: 유효하지 않은 ChatML 형식 (스킵)")

    print(f"[SUCCESS] JSONL 검증 완료: {valid}개 유효, {invalid}개 무효")
    return valid


def main():
    parser = argparse.ArgumentParser(description="AMEVA-LLM-Trainer 데이터 정제 스크립트")
    parser.add_argument("--input", "-i", required=True, help="입력 파일 경로")
    parser.add_argument("--output", "-o", default=None, help="출력 파일 경로 (기본: dataset/train.jsonl)")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or os.path.join(DATASET_DIR, "train.jsonl")

    if not os.path.exists(input_path):
        print(f"[ERROR] 입력 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ext = os.path.splitext(input_path)[1].lower()
    print(f"[INFO] 입력 파일: {input_path}")
    print(f"[INFO] 출력 경로: {output_path}")

    if ext == ".txt":
        print("[INFO] TXT → ChatML JSONL 변환 모드")
        convert_txt_to_chatml(input_path, output_path)
    elif ext in (".json", ".jsonl"):
        print("[INFO] JSONL ChatML 검증 및 정제 모드")
        validate_jsonl(input_path, output_path)
    else:
        print(f"[ERROR] 지원하지 않는 파일 형식: {ext}")
        sys.exit(1)

    print(f"\n[DONE] 정제된 학습 데이터: {output_path}")


if __name__ == "__main__":
    main()
