"""
src/backend/main.py
FastAPI 엔트리포인트 - AMEVA-LLM-Trainer Headless API Server (포트 8001)

⚠️ 시니어 피드백 반영:
   - #3: 라우터 명시적 등록 (app.include_router)
   - 태스크 프로세스 완료 감지를 위한 백그라운드 워치독 포함
"""
import os
import sys
import asyncio
import threading
import time

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from src.backend.core.database import db_manager

# ⚠️ 시니어 피드백 #3: 라우터 명시적 임포트
from src.backend.routers import tasks as tasks_router
from src.backend.routers import system as system_router

app = FastAPI(
    title="AMEVA-LLM-Trainer Headless API Server",
    version="1.0.0",
    description="로컬/오프라인 CPU 특화 소형 LLM Fine-tuning API"
)

# ⚠️ 시니어 피드백 #3: 라우터 명시적 등록 - 절대 누락 금지!
app.include_router(tasks_router.router, prefix="/api/v1/tasks", tags=["Tasks"])
# system 라우터: /api/v1 하위에 /system/*, /files/*, /db/* 엔드포인트를 통합 제공
app.include_router(system_router.router, prefix="/api/v1", tags=["System", "Files", "Database"])


@app.get("/")
def read_root():
    """헬스체크 엔드포인트"""
    return {"status": "healthy", "service": "AMEVA-LLM-Trainer API", "port": 8001}


def _process_watchdog():
    """
    백그라운드 프로세스 완료 감지 워치독.
    active_train_processes를 주기적으로 폴링하여
    완료된 프로세스의 DB 상태를 업데이트한다.
    """
    from src.backend.routers.tasks import active_train_processes

    while True:
        try:
            completed_ids = []
            for task_id, proc in list(active_train_processes.items()):
                retcode = proc.poll()
                if retcode is not None:
                    # 프로세스가 종료됨
                    if retcode == 0:
                        db_manager.update_task_progress(task_id, 100.0, "COMPLETED")
                        db_manager.insert_log(task_id, "SUCCESS", "학습 프로세스 정상 완료")
                    else:
                        # stderr 캡처 시도
                        stderr_out = ""
                        try:
                            stderr_out = proc.stderr.read().decode("utf-8", errors="replace")[-500:]
                        except Exception:
                            pass
                        db_manager.update_task_progress(task_id, 0.0, "FAILED")
                        db_manager.insert_log(
                            task_id, "ERROR",
                            f"학습 프로세스 비정상 종료 (exit code: {retcode}). {stderr_out}"
                        )
                    completed_ids.append(task_id)

            for tid in completed_ids:
                active_train_processes.pop(tid, None)

        except Exception:
            pass

        time.sleep(3)  # 3초 간격 폴링


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 워치독 스레드 가동"""
    watchdog_thread = threading.Thread(target=_process_watchdog, daemon=True)
    watchdog_thread.start()
