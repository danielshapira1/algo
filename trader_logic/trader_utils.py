import time
from functools import wraps
import os
import numpy as np
import yfinance as yf

class TraderException(Exception):
    pass

def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise TraderException(f"Failed after {max_attempts} attempts: {str(e)}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))