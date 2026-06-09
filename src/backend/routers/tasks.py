"""
src/backend/routers/tasks.py
태스크 CRUD 라우터 - 목록, 생성, 상세, 메트릭, 로그, 중지

⚠️ 시니어 피드백 반영:
   - 태스크 중지 시 active_train_processes 딕셔너리에서 PID 조회
   - process.terminate() + Windows taskkill /F /PID 안정장치 이중 구현
"""
import os
import uuid
import subprocess
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.backend.core.database import db_manager
from src.core.config import ROOT_DIR, CFG

router = APIRouter()

# ⚠️ 시니어 피드백 #1: 백그라운드 프로세스 PID 추적 전역 딕셔너리
# main.py에서 import하여 프로세스 등록/조회에 사용
active_train_processes = {}


class TaskCreate(BaseModel):
    name: str
    data_path: str
    model_id: Optional[str] = None


class TaskStop(BaseModel):
    task_id: str


@router.get("/list")
def list_tasks():
    """등록된 모든 태스크 목록 반환"""
    tasks = db_manager.get_all_tasks()
    return {"tasks": tasks}


@router.post("/create")
def create_task(task_in: TaskCreate):
    """신규 학습 태스크 생성 및 백그라운드 학습 프로세스 스폰"""
    task_id = str(uuid.uuid4())
    model_id = task_in.model_id or CFG["model_id"]
    db_manager.insert_task(task_id, task_in.name, task_in.data_path, model_id)
    db_manager.insert_log(task_id, "INFO", f"태스크 생성됨: {task_in.name}")

    # 02_run_tuning.py 스크립트를 백그라운드 서브프로세스로 안전 격리 실행
    python_exe = os.path.join(ROOT_DIR, "venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python"  # fallback

    script_path = os.path.join(ROOT_DIR, "scripts", "02_run_tuning.py")

    env = os.environ.copy()
    env["CURRENT_TASK_ID"] = task_id

    cmd = [
        python_exe, script_path,
        "--task-id", task_id,
        "--data-path", task_in.data_path
    ]

    try:
        # ⚠️ 시니어 피드백 #1: Popen 객체를 전역 딕셔너리에 PID와 함께 등록
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        active_train_processes[task_id] = proc
        db_manager.update_task_progress(task_id, 0.0, "RUNNING")
        db_manager.insert_log(task_id, "INFO", f"학습 프로세스 시작 (PID: {proc.pid})")

        return {
            "task_id": task_id,
            "status": "RUNNING",
            "pid": proc.pid,
            "message": "학습 프로세스가 백그라운드에서 시작되었습니다."
        }
    except Exception as e:
        db_manager.update_task_progress(task_id, 0.0, "FAILED")
        db_manager.insert_log(task_id, "ERROR", f"프로세스 시작 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"학습 프로세스 시작 실패: {str(e)}")


@router.post("/stop")
def stop_task(task_in: TaskStop):
    """
    ⚠️ 시니어 피드백 #1 핵심 구현:
    백그라운드 학습 프로세스를 안전하게 종료한다.
    1차: process.terminate()
    2차: Windows taskkill /F /PID (안정장치)
    """
    task_id = task_in.task_id
    proc = active_train_processes.get(task_id)

    if proc is None:
        raise HTTPException(status_code=404, detail="해당 태스크의 실행 중인 프로세스를 찾을 수 없습니다.")

    pid = proc.pid
    try:
        # 1차: Python subprocess terminate
        proc.terminate()
        proc.wait(timeout=5)
        db_manager.insert_log(task_id, "WARNING", f"프로세스 정상 종료 (PID: {pid})")
    except subprocess.TimeoutExpired:
        # 2차: Windows taskkill 강제 종료 (안정장치)
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    check=True, capture_output=True
                )
                db_manager.insert_log(task_id, "WARNING", f"taskkill 강제 종료 (PID: {pid})")
            except Exception as kill_err:
                db_manager.insert_log(task_id, "ERROR", f"taskkill 실패: {str(kill_err)}")
        else:
            proc.kill()
            db_manager.insert_log(task_id, "WARNING", f"SIGKILL 강제 종료 (PID: {pid})")
    except Exception as e:
        db_manager.insert_log(task_id, "ERROR", f"프로세스 종료 실패: {str(e)}")
    finally:
        # 프로세스 딕셔너리에서 제거 및 DB 상태 업데이트
        active_train_processes.pop(task_id, None)
        db_manager.update_task_progress(task_id, 0.0, "STOPPED")

    return {"task_id": task_id, "status": "STOPPED", "message": f"프로세스 종료됨 (PID: {pid})"}


@router.get("/detail")
def get_task_detail(task_id: str):
    """태스크 상세 정보"""
    task = db_manager.get_task_details(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="태스크를 찾을 수 없습니다.")
    return {"task_info": task}


@router.get("/metrics")
def get_task_metrics(task_id: str):
    """태스크 학습 메트릭 (step, loss) 목록"""
    metrics = db_manager.get_metrics(task_id)
    return {"metrics": metrics}


@router.get("/logs")
def get_task_logs(task_id: str):
    """태스크 로그 (최근 50개)"""
    logs = db_manager.get_logs(task_id, limit=50)
    return {"logs": logs}


@router.get("/report")
def get_task_report(task_id: str):
    """태스크 리포트 (상세 + 로그)"""
    task = db_manager.get_task_details(task_id)
    logs = db_manager.get_logs(task_id, limit=20)
    return {"task_info": task, "logs": logs}
