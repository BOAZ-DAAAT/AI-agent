# Backend core refactor plan (2026-07-07)

## 배경
- 이슈 #62는 backend를 서버 밖에서 접근 가능한 MySQL 연결·조회 계층처럼 확장하려는 의도로 열렸다.
- 그런데 현재 실제 사용 흐름을 보면 datasource 선택과 SQL 실행을 backend가 소유하기보다, agent runtime이 직접 MySQL에 연결하고 쿼리를 실행하는 방향이 더 일관된다.
- 따라서 backend를 "DB 실행 플랫폼"으로 키우기보다, 오케스트레이션이 공통으로 의존하는 최소 저장/추적 코어로 축소하는 편이 현재 구조와 더 맞다.

## 이번 정리에서 확정한 방향
### backend core에 남길 것
- run 생성/조회/상태 갱신
- artifact 저장/조회/lineage
- retained policy 및 기본 health/bootstrap

### backend core에서 뺄 것
- datasource 관리
- SQL 실행/preview/execute 계층
- Python sandbox execution 계층
- approval/workspace/export 보조 기능의 기본 의존성
- memory/catalog 계층

## 핵심 결정
1. **DB 연결과 SQL 실행의 canonical 위치는 agent runtime이다.**
   - MySQL 연결 정보 관리와 실제 쿼리 수행은 agent 쪽에서 직접 담당한다.
   - backend는 그 결과물과 실행 이력을 저장/참조하는 최소 코어 역할에 집중한다.
2. **report persistence는 artifact-backed path로 표준화한다.**
   - 최종 보고서/중간 산출물은 backend artifact 계약을 통해 추적한다.
3. **shim은 증명된 호출자에 한해서만 임시 허용한다.**
   - 실제 런타임 호출자가 확인된 경우에만 짧게 유지한다.
   - 테스트 편의만을 위한 shim은 두지 않는다.
4. **서버 배포 후 원격 DB 연결은 backend 복귀 이유가 아니다.**
   - 나중에 서버에 올리더라도 agent가 네트워크로 원격 DB에 직접 붙는 구조는 가능하다.
   - 다만 자격증명 관리, 네트워크 정책, connection pooling, 관측성은 별도 운영 설계로 다뤄야 한다.

## 실행 순서
1. backend 호출자 inventory 작성
2. reduced core boundary 선언
3. datasource/sql/python sandbox surface 제거
4. adapter/supervisor/report 경계 단순화
5. 테스트, 문서, 배포 메모 정리

## 리스크
- 숨어 있는 backend 호출자가 남아 있을 수 있음
- 임시 shim이 장기 의존성으로 굳어질 수 있음
- artifact 외 binary/chart/report edge case 정리가 필요할 수 있음
- 서버 배포 시 direct-DB 운영 설계가 후속 작업으로 필요함

## 이슈 #62와의 관계
- #62의 문제의식(외부에서 backend를 통해 DB를 연결하고 조회하고 싶다)은 이해되지만,
  현재 코드베이스 기준으로는 그 역할을 backend에 넣기보다 agent runtime으로 옮기는 쪽이 더 단순하고 일관적이다.
- 따라서 이번 작업은 #62를 그대로 구현하는 방향이 아니라,
  backend를 최소 코어로 축소하고 DB 실행 책임을 agent 쪽으로 명확히 재배치하는 구조 정리 작업이다.
