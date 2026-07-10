"""비밀번호 해시 생성 CLI.

단일:   python -m backend.auth.hashtool "mypassword"
여러개: python -m backend.auth.hashtool --bulk admin1=pw1:admin test1=pw2:tester
        → accounts.py 에 붙여넣을 수 있는 ACCOUNTS 블록을 출력한다.
"""
from __future__ import annotations

import sys

from backend.auth.security import hash_password


def _bulk(pairs: list[str]) -> None:
    print("ACCOUNTS = {")
    for pair in pairs:
        creds, _, role = pair.partition(":")
        username, _, password = creds.partition("=")
        role = role or "tester"
        print(f'    "{username}": {{"password_hash": "{hash_password(password)}", "role": "{role}"}},')
    print("}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--bulk":
        _bulk(args[1:])
    elif len(args) == 1:
        print(hash_password(args[0]))
    else:
        print('usage: python -m backend.auth.hashtool "<password>"')
        print('   or: python -m backend.auth.hashtool --bulk user=pw:role ...')
        sys.exit(1)