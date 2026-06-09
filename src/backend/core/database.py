"""
src/backend/core/database.py
SQLite 스키마 설계 및 DatabaseManager 싱글톤

⚠️ 시니어 피드백 반영:
   - timeout=30.0 설정으로 동시성 락(Database Locked) 방어
   - API 서버와 백그라운드 학습 스크립트가 동시 쓰기 시 최대 30초 대기
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs", "ameva_llm.db")
)


class DatabaseManager:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        # ⚠️ timeout=30.0: DB 동시 접근 시 에러 대신 최대 30초 대기하여 락 충돌 방어
        self.conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        with self.conn:
            # 태스크 관리 테이블
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    tsk_nm TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'WAITING',
                    progress REAL DEFAULT 0.0,
                    start_time TEXT,
                    end_time TEXT,
                    data_path TEXT,
                    model_id TEXT,
                    create_dt TEXT DEFAULT (datetime('now', 'localtime'))
                )
            """)
            # 실시간 손실율 트래킹 테이블
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    step INTEGER,
                    loss REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                )
            """)
            # 로그 테이블
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    level TEXT DEFAULT 'INFO',
                    message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                )
            """)

    def insert_task(self, task_id: str, name: str, data_path: str, model_id: str = None):
        with self.conn:
            self.conn.execute(
                "INSERT INTO tasks (task_id, tsk_nm, status, data_path, model_id, start_time) "
                "VALUES (?, ?, 'WAITING', ?, ?, ?)",
                (task_id, name, data_path, model_id, datetime.now().isoformat())
            )

    def update_task_progress(self, task_id: str, progress: float, status: str):
        with self.conn:
            if status in ("COMPLETED", "FAILED"):
                self.conn.execute(
                    "UPDATE tasks SET progress = ?, status = ?, end_time = ? WHERE task_id = ?",
                    (progress, status, datetime.now().isoformat(), task_id)
                )
            else:
                self.conn.execute(
                    "UPDATE tasks SET progress = ?, status = ? WHERE task_id = ?",
                    (progress, status, task_id)
                )

    def insert_log_metric(self, task_id: str, step: int, loss: float):
        with self.conn:
            self.conn.execute(
                "INSERT INTO metrics (task_id, step, loss) VALUES (?, ?, ?)",
                (task_id, step, loss)
            )

    def insert_log(self, task_id: str, level: str, message: str):
        with self.conn:
            self.conn.execute(
                "INSERT INTO logs (task_id, level, message) VALUES (?, ?, ?)",
                (task_id, level, message)
            )

    def get_metrics(self, task_id: str):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT step, loss, timestamp FROM metrics WHERE task_id = ? ORDER BY step ASC",
            (task_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_all_tasks(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tasks ORDER BY create_dt DESC")
        return [dict(row) for row in cursor.fetchall()]

    def get_task_details(self, task_id: str):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_logs(self, task_id: str, limit: int = 50):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT level, message, timestamp FROM logs WHERE task_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (task_id, limit)
        )
        rows = [dict(row) for row in cursor.fetchall()]
        rows.reverse()
        return rows

    def run_query(self, sql: str, params: list = None):
        """커스텀 SQL 쿼리 실행 (SELECT 전용)"""
        cursor = self.conn.cursor()
        cursor.execute(sql, params or [])
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return columns, [list(row) for row in rows]


db_manager = DatabaseManager()
