from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import create_app
from backend.storage import routes as storage_routes


class FakeIntegrityService:
    def __init__(self) -> None:
        self.notified = []
        self.processed = 0

    def notify_dataset_update(self, dataset_name, tables=None, source_version=None, metadata=None):
        self.notified.append((dataset_name, tables, source_version, metadata))

        class Result:
            ok = True

        return Result()

    def process_next_pending(self):
        self.processed += 1


class FakeServices:
    def __init__(self, integrity_service) -> None:
        self.integrity_service = integrity_service


def test_storage_ingest_enqueues_integrity_background_check(monkeypatch) -> None:
    fake_integrity = FakeIntegrityService()
    storage_routes._backend_services = FakeServices(fake_integrity)
    monkeypatch.setattr(storage_routes, "ingest_database", lambda *args, **kwargs: {"orders": 2})

    client = TestClient(create_app())
    response = client.post(
        "/storage/ingest",
        json={"host": "remote", "port": 3306, "user": "u", "password": "p", "database": "src", "target_database": "dst"},
    )

    assert response.status_code == 200
    assert response.json() == {"target_database": "dst", "tables": {"orders": 2}}
    assert fake_integrity.notified == [
        ("dst", {"orders": 2}, None, {"source": "storage_ingest", "source_database": "src", "row_counts": {"orders": 2}})
    ]
    assert fake_integrity.processed == 1
