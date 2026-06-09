"""
cli/client/api_client.py
REST API 통신 모듈 - AMEVA_API_URL 바인딩 (기본 포트 8001)
"""
import os
import requests
import json
from urllib.parse import urljoin


class APIClient:
    def __init__(self):
        # AMEVA-STT-Trainer(8000) 포트와 충돌하지 않도록 디폴트 8001로 격리
        self.base_url = os.environ.get("AMEVA_API_URL", "http://127.0.0.1:8001")
        self.api_key = os.environ.get("AMEVA_API_KEY", "")
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            self.headers["X-API-Key"] = self.api_key

    def _get_url(self, path: str):
        if not path.startswith("/"):
            path = "/" + path
        return urljoin(self.base_url, path)

    def get(self, path: str, params: dict = None) -> dict:
        try:
            res = requests.get(
                self._get_url(path), headers=self.headers,
                params=params, timeout=5.0
            )
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def post(self, path: str, json_data: dict = None) -> dict:
        try:
            res = requests.post(
                self._get_url(path), headers=self.headers,
                json=json_data, timeout=10.0
            )
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def check_health(self) -> bool:
        try:
            res = requests.get(self._get_url("/"), timeout=2.0)
            return res.status_code == 200
        except Exception:
            return False


api_client = APIClient()
