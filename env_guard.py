"""Test-only guard: keep the real EXA_API_KEY (and the real .env) out of unit tests.

Call `install()` BEFORE importing tools/agents. It forces a recognizable fake key and disables
python-dotenv's `load_dotenv`, which tools.py calls at import time and which would otherwise read
the developer's real .env (deleting the env var alone is not enough: load_dotenv would reload it).
"""
import os
from unittest import mock

FAKE_KEY = "sk-fake+canary/leak=0123456789abcdef"

_installed = False
_load_dotenv_patcher = mock.patch("dotenv.load_dotenv", lambda *args, **kwargs: None)


def install():
    """Force a fake EXA_API_KEY and block .env loading for this process. Returns the fake key."""
    global _installed
    os.environ["EXA_API_KEY"] = FAKE_KEY
    if not _installed:
        _load_dotenv_patcher.start()
        _installed = True
    return FAKE_KEY
