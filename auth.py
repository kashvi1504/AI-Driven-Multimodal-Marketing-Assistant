"""
auth.py
-------
Local accounts for the web app: sign up, log in, log out, plus a short
history of each user's analyses for the dashboard.

Everything stays on this machine, in data/accounts/ (git-ignored):
  users.json     - name, email and a salted PBKDF2-SHA256 password hash per user
  activity.json  - each user's most recent analyses, newest first
  secret.key     - random key that signs the session cookie

Sessions are stateless signed cookies (HMAC-SHA256), so restarting the
server doesn't log anyone out. This protects a local, single-machine
install - it isn't meant to be exposed to the internet as-is.
"""

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
import uuid

from config import DATA_DIR

ACCOUNTS_DIR = os.path.join(DATA_DIR, "accounts")
USERS_FILE = os.path.join(ACCOUNTS_DIR, "users.json")
ACTIVITY_FILE = os.path.join(ACCOUNTS_DIR, "activity.json")
SECRET_FILE = os.path.join(ACCOUNTS_DIR, "secret.key")

COOKIE_NAME = "ss_session"
SESSION_SECONDS = 14 * 24 * 3600
PBKDF2_ITERATIONS = 240_000
MIN_PASSWORD = 8
MAX_ACTIVITY = 60  # per user

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_lock = threading.Lock()
_secret_cache: bytes | None = None


class AuthError(Exception):
    """A sign-up / log-in problem with a message meant for the user."""

    def __init__(self, title: str, hint: str = "", status: int = 400):
        super().__init__(title)
        self.title, self.hint, self.status = title, hint, status


# ── Storage ──────────────────────────────────────────────────────────────
def _read(path: str, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write(path: str, data) -> None:
    os.makedirs(ACCOUNTS_DIR, exist_ok=True)
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)  # atomic, so a crash never leaves a half-written file


def _secret() -> bytes:
    global _secret_cache
    if _secret_cache is None:
        with _lock:
            if not os.path.exists(SECRET_FILE):
                os.makedirs(ACCOUNTS_DIR, exist_ok=True)
                with open(SECRET_FILE, "wb") as f:
                    f.write(secrets.token_bytes(32))
                os.chmod(SECRET_FILE, 0o600)
            with open(SECRET_FILE, "rb") as f:
                _secret_cache = f.read()
    return _secret_cache


# ── Passwords ────────────────────────────────────────────────────────────
def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${b64(salt)}${b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt, digest = stored.split("$")
        expected = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt), int(iterations))
        return hmac.compare_digest(expected, base64.b64decode(digest))
    except (ValueError, TypeError):
        return False


# ── Users ────────────────────────────────────────────────────────────────
def _public(user: dict) -> dict:
    return {"id": user["id"], "name": user["name"], "email": user["email"], "created": user["created"]}


def create_user(name: str, email: str, password: str) -> dict:
    name, email = (name or "").strip(), (email or "").strip().lower()
    if not name:
        raise AuthError("Enter your name.")
    if len(name) > 60:
        raise AuthError("That name is too long.", "Use 60 characters or fewer.")
    if not EMAIL_RE.match(email):
        raise AuthError("Enter a valid email address.")
    if len(password or "") < MIN_PASSWORD:
        raise AuthError(f"Use a password of at least {MIN_PASSWORD} characters.")
    with _lock:
        users = _read(USERS_FILE, {})
        if any(u["email"] == email for u in users.values()):
            raise AuthError("An account with that email already exists.", "Log in instead.", 409)
        user = {"id": uuid.uuid4().hex, "name": name, "email": email,
                "password": hash_password(password), "created": int(time.time())}
        users[user["id"]] = user
        _write(USERS_FILE, users)
    return _public(user)


_DUMMY_HASH = hash_password(secrets.token_hex(8))


def authenticate(email: str, password: str) -> dict:
    email = (email or "").strip().lower()
    user = next((u for u in _read(USERS_FILE, {}).values() if u["email"] == email), None)
    # always run one hash so an unknown email takes as long as a wrong password
    ok = verify_password(password or "", user["password"] if user else _DUMMY_HASH)
    if not (user and ok):
        raise AuthError("That email and password don't match.", "Check them and try again.", 401)
    return _public(user)


def get_user(user_id: str) -> dict | None:
    user = _read(USERS_FILE, {}).get(user_id)
    return _public(user) if user else None


# ── Sessions ─────────────────────────────────────────────────────────────
def _sign(payload: str) -> str:
    sig = hmac.new(_secret(), payload.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode().rstrip("=")


def make_session(user_id: str) -> str:
    payload = f"{user_id}.{int(time.time()) + SESSION_SECONDS}"
    return f"{payload}.{_sign(payload)}"


def read_session(token: str | None) -> str | None:
    """User id from a session cookie, or None if it's missing, forged or expired."""
    try:
        user_id, expires, sig = (token or "").split(".")
        if not hmac.compare_digest(sig, _sign(f"{user_id}.{expires}")) or int(expires) < time.time():
            return None
        return user_id
    except ValueError:
        return None


# ── Activity (dashboard history) ─────────────────────────────────────────
def record_activity(user_id: str | None, kind: str, title: str, score: float | None = None,
                    label: str | None = None, detail: str = "") -> None:
    if not user_id:
        return
    entry = {"kind": kind, "title": (title or "")[:120], "detail": detail[:160], "label": label,
             "score": None if score is None else round(float(score), 1), "at": int(time.time())}
    with _lock:
        activity = _read(ACTIVITY_FILE, {})
        activity[user_id] = [entry, *activity.get(user_id, [])][:MAX_ACTIVITY]
        _write(ACTIVITY_FILE, activity)


def list_activity(user_id: str) -> list:
    return _read(ACTIVITY_FILE, {}).get(user_id, [])
