"""로그인 계정 정의 (개발/테스트 프리셋).

- 비밀번호는 평문이 아니라 hash_password()로 만든 해시를 넣는다.
- 해시 생성:  cd AI-agent && python -m backend.auth.hashtool "your-password"
- 새 계정 추가 = ACCOUNTS 에 한 줄 추가 (이곳이 확장 지점).
- role 은 지금은 라벨용("admin"/"tester"). 동작 차이는 없지만 토큰에 실려
  미래에 권한 분기(마트 삭제 권한 등)로 확장 가능.
"""
from __future__ import annotations

# username -> {"password_hash": "pbkdf2_sha256$...", "role": "admin" | "tester"}
ACCOUNTS: dict[str, dict[str, str]] = {
    "suky8658": {"password_hash": "pbkdf2_sha256$120000$RnDvgOe2j4Jz9aU8iN4Cqg==$iKKFlnCReMCSXI4/Xi0FWL9LFulcrTT/dY/pefjme08=", "role": "admin"},
    "chae-jpg": {"password_hash": "pbkdf2_sha256$120000$mZoD7jxoVaXD4W7Xg+dcWw==$w/wfgCyB+wxer4Y74BjBr8Ozv6Rqr/YvXWaLDfqUIZ4=", "role": "admin"},
    "e-jungs2": {"password_hash": "pbkdf2_sha256$120000$qd/T79SkIGWQlcgwePxOFA==$cfsTQs5VeyPOXJE2XvAYSmjZufI3pBJEL49bpl+uhB4=", "role": "admin"},
    "ity0526": {"password_hash": "pbkdf2_sha256$120000$4tiMJRT0mN8BpjvFP5MT1Q==$WYGwvr/G4rfdSqVugky0p6YWj+Ib7cLDQxt7WuncYZA=", "role": "admin"},
    "yhk1059": {"password_hash": "pbkdf2_sha256$120000$6mpXTVbCvcYMIhkVZtDNdA==$VTiANAsowgWQGxCaHWJOYnWox0434pNfmibkzTpb7U4=", "role": "admin"},
}