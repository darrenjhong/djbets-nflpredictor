# utils.py
"""
Utility helpers: safe_request_json, small numeric helpers
"""

import requests
import json
import time

def safe_request_json(url, params=None, timeout=6, retries=2):
    try:
        for i in range(retries+1):
            try:
                r = requests.get(url, params=params or {}, timeout=timeout, headers={"User-Agent":"DJBetsBot/1.0"})
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if i < retries:
                    time.sleep(0.5)
                    continue
                return None
    except Exception:
        return None