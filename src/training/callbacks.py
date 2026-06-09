"""
src/training/callbacks.py
실시간 DB 트래킹 커스텀 콜백 - 매 Step마다 Loss/Step을 SQLite에 기록
"""
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from src.backend.core.database import db_manager


class DBProgressCallback(TrainerCallback):
    """
    학습 진행 중 매 로깅 스텝마다 Loss 및 Step 카운터를 SQLite DB에 기록하여
    CLI 클라이언트에서 실시간 플로팅 및 모니터링을 가능하게 한다.
    """
    def __init__(self, task_id: str):
        self.task_id = task_id

    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss", None)
            step = state.global_step
            max_steps = state.max_steps
            progress = (step / max_steps) * 100 if max_steps > 0 else 0

            if loss is not None:
                # DB의 태스크 메트릭 테이블 및 작업 진행 상태 업데이트
                db_manager.update_task_progress(self.task_id, progress, status="RUNNING")
                db_manager.insert_log_metric(self.task_id, step, loss)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs):
        db_manager.update_task_progress(self.task_id, 0.0, status="RUNNING")

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        db_manager.update_task_progress(self.task_id, 100.0, status="COMPLETED")
