"""
scripts/02_run_tuning.py
백그라운드 학습 스크립트 - FastAPI에서 서브프로세스로 스폰되는 대상

사용법:
  python scripts/02_run_tuning.py --task-id <UUID> --data-path <경로>
"""
import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(description="AMEVA-LLM-Trainer 학습 실행")
    parser.add_argument("--task-id", required=True, help="태스크 ID")
    parser.add_argument("--data-path", required=True, help="학습 데이터 경로 (ChatML JSONL)")
    args = parser.parse_args()

    task_id = args.task_id
    data_path = args.data_path

    # 태스크 ID를 환경변수로 설정 (config.py에서 참조)
    os.environ["CURRENT_TASK_ID"] = task_id

    # 이제 config를 임포트해야 태스크별 경로가 적용됨
    from src.core.config import OUTPUTS_DIR, LORA_DIR
    from src.backend.core.database import db_manager

    print(f"[INFO] Task ID: {task_id}")
    print(f"[INFO] Data Path: {data_path}")
    print(f"[INFO] Output Dir: {OUTPUTS_DIR}")

    # 데이터 파일 존재 확인
    if not os.path.exists(data_path):
        db_manager.update_task_progress(task_id, 0.0, "FAILED")
        db_manager.insert_log(task_id, "ERROR", f"학습 데이터 파일을 찾을 수 없습니다: {data_path}")
        print(f"[ERROR] Data file not found: {data_path}")
        sys.exit(1)

    # 학습 시작
    db_manager.insert_log(task_id, "INFO", "학습 프로세스 초기화 중...")

    try:
        from src.training.trainer import run_fine_tuning

        db_manager.insert_log(task_id, "INFO", "모델 로딩 및 학습 시작...")
        run_fine_tuning(task_id, data_path)

        db_manager.update_task_progress(task_id, 100.0, "COMPLETED")
        db_manager.insert_log(task_id, "SUCCESS", "학습 프로세스 정상 완료!")
        print("[SUCCESS] Fine-tuning completed successfully!")

    except Exception as e:
        db_manager.update_task_progress(task_id, 0.0, "FAILED")
        db_manager.insert_log(task_id, "ERROR", f"학습 중 오류 발생: {str(e)}")
        print(f"[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
