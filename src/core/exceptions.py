"""
src/core/exceptions.py
글로벌 데코레이터 예외 가드 - OOM, 파일 차단 등 어떠한 예외도 안전 로깅 후 복구
"""
import functools
import logging
import os
import traceback
from datetime import datetime

# 로그 파일 핸들러 생성
_log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, "exception.log")
logging.basicConfig(
    filename=_log_file,
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] (%(location)s) %(message)s"
)


def log_exception(e: Exception, location: str):
    tb = traceback.format_exc()
    logging.error(
        f"Error: {str(e)}\nTraceback:\n{tb}",
        extra={"location": location}
    )


def exception_guard(location: str = None, reraise: bool = False):
    """
    함수에 적용하여 모든 예외를 안전하게 포착, 로깅하는 데코레이터.
    reraise=True 시 로깅 후 예외를 재발생시킨다.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            loc = location or f"{func.__module__}.{func.__name__}()"
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_exception(e, loc)
                if reraise:
                    raise
                return None
        return wrapper
    return decorator
