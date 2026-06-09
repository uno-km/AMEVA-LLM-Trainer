"""
src/backend/routers/system.py
시스템 리소스 조회, 파일 탐색, DB 쿼리 라우터
"""
import os
import psutil
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from src.core.config import ROOT_DIR
from src.backend.core.database import db_manager

router = APIRouter()


@router.get("/system/resources")
def get_system_resources():
    """시스템 리소스(CPU, RAM, Disk) 사용 현황"""
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(ROOT_DIR)

    # 상위 프로세스 목록 (CPU 기준 상위 5개)
    processes = []
    try:
        for proc in sorted(
            psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']),
            key=lambda p: p.info.get('cpu_percent', 0) or 0,
            reverse=True
        )[:5]:
            info = proc.info
            mem = info.get('memory_info')
            mem_str = f"{mem.rss / (1024**2):.1f}MB" if mem else "N/A"
            processes.append({
                "pid": info.get('pid', 0),
                "name": info.get('name', 'Unknown'),
                "cpu": f"{info.get('cpu_percent', 0):.1f}%",
                "mem": mem_str
            })
    except Exception:
        pass

    return {
        "cpu": psutil.cpu_percent(interval=None),
        "ram": vm.percent,
        "ram_used": vm.used / (1024**3),
        "ram_total": vm.total / (1024**3),
        "gpu": 0,  # CPU 전용 환경
        "gpu_mem": "N/A (CPU Mode)",
        "disk_pct": disk.percent,
        "disk_used": disk.used / (1024**3),
        "disk_total": disk.total / (1024**3),
        "disk_free_gb": disk.free / (1024**3),
        "processes": processes
    }


def _scan_dir(path: str, max_depth: int = 3, current_depth: int = 0) -> list:
    """디렉토리를 재귀 스캔하여 트리 구조 반환"""
    result = []
    if current_depth >= max_depth or not os.path.isdir(path):
        return result

    try:
        for entry in os.scandir(path):
            if entry.name in ("__pycache__", ".git", "venv", "node_modules"):
                continue
            node = {
                "name": entry.name,
                "path": entry.path.replace("\\", "/"),
                "is_dir": entry.is_dir()
            }
            if entry.is_dir():
                node["children"] = _scan_dir(entry.path, max_depth, current_depth + 1)
            else:
                try:
                    node["size"] = entry.stat().st_size
                except Exception:
                    node["size"] = 0
            result.append(node)
    except PermissionError:
        pass

    return result


@router.get("/system/files/explorer")
def get_file_explorer():
    """프로젝트 주요 디렉토리의 파일 트리 구조 반환"""
    categories = {
        "dataset": os.path.join(ROOT_DIR, "dataset"),
        "outputs": os.path.join(ROOT_DIR, "outputs"),
        "logs": os.path.join(ROOT_DIR, "logs"),
        "configs": os.path.join(ROOT_DIR, "configs"),
    }
    result = {}
    for key, path in categories.items():
        result[key] = _scan_dir(path) if os.path.exists(path) else []
    return result


@router.get("/system/files/read")
def read_file_content(path: str):
    """원격 텍스트 파일 내용 읽기"""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {path}")

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".csv":
            import csv
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                content = [row for row in reader]
            return {"type": "csv", "content": content}
        else:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return {"type": "text", "content": [line.rstrip() for line in lines]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 읽기 실패: {str(e)}")


@router.get("/system/files/search")
def search_files(keyword: str, exts: str = ""):
    """프로젝트 내 파일 검색"""
    ext_filter = [e.strip() for e in exts.split(",") if e.strip()] if exts else []
    results = []

    for root, dirs, files in os.walk(ROOT_DIR):
        # 불필요한 디렉토리 제외
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "venv", "node_modules")]
        for f in files:
            if keyword.lower() in f.lower():
                if ext_filter and not any(f.lower().endswith(e) for e in ext_filter):
                    continue
                full_path = os.path.join(root, f)
                try:
                    size = os.path.getsize(full_path)
                except Exception:
                    size = 0
                results.append({
                    "name": f,
                    "path": full_path.replace("\\", "/"),
                    "dir": root.replace("\\", "/"),
                    "size": size
                })

    return {"results": results[:100]}


class DBQuery(BaseModel):
    sql: str
    params: Optional[List] = []


@router.post("/system/db/query")
def run_db_query(query: DBQuery):
    """커스텀 SQL SELECT 쿼리 실행"""
    sql = query.sql.strip()
    # 안전 검증: SELECT만 허용
    if not sql.upper().startswith("SELECT") and not sql.upper().startswith("PRAGMA"):
        raise HTTPException(status_code=400, detail="SELECT 또는 PRAGMA 쿼리만 실행할 수 있습니다.")

    try:
        columns, rows = db_manager.run_query(sql, query.params)
        return {"columns": columns, "rows": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"쿼리 실행 실패: {str(e)}")
