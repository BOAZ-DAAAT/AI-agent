"""Tests for datasource service: registration, dedup, query, catalog, adapter methods."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_agent_backend.config import BackendConfig
from data_agent_backend.models.datasource import (
    CatalogSummary,
    DatasourceCreateRequest,
    DatasourceType,
    TableSummary,
    ColumnInfo,
)
from data_agent_backend.services.datasource_service import DatasourceService
from data_agent_backend.services.factory import create_backend_services, _auto_register_default_datasource
from data_agent_backend.storage.sqlite import SQLiteStore
from DATA_Analyst_Assistant_Agent import BackendAdapter, SQLAgentSupervisor


# ── fixtures ──

@pytest.fixture()
def tmp_dir():
    d = Path(".test_data") / f"ds_{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def sqlite(tmp_dir):
    return SQLiteStore(tmp_dir / "test.sqlite")


@pytest.fixture()
def ds_service(sqlite):
    return DatasourceService(sqlite)


@pytest.fixture()
def adapter(tmp_dir):
    config = BackendConfig(base_data_dir=tmp_dir / ".data_agent")
    return BackendAdapter(config=config)


def _make_request(**kwargs):
    defaults = dict(
        name="test_mysql", type=DatasourceType.mysql,
        host="localhost", port=3306, database="sql_agent",
        username="root", password="1234",
    )
    defaults.update(kwargs)
    return DatasourceCreateRequest(**defaults)


# ── DatasourceService registration ──

class TestDatasourceRegistration:
    def test_register_creates_record(self, ds_service):
        req = _make_request()
        rec = ds_service.register(req)
        assert rec.datasource_id.startswith("ds_")
        assert rec.name == "test_mysql"
        assert rec.host == "localhost"
        assert rec.database == "sql_agent"

    def test_register_dedup_same_combo(self, ds_service):
        req = _make_request()
        rec1 = ds_service.register(req)
        rec2 = ds_service.register(req)
        assert rec1.datasource_id == rec2.datasource_id

    def test_register_different_name_creates_new(self, ds_service):
        rec1 = ds_service.register(_make_request(name="ds1"))
        rec2 = ds_service.register(_make_request(name="ds2"))
        assert rec1.datasource_id != rec2.datasource_id

    def test_get_existing(self, ds_service):
        req = _make_request()
        rec = ds_service.register(req)
        fetched = ds_service.get(rec.datasource_id)
        assert fetched.datasource_id == rec.datasource_id
        assert fetched.name == "test_mysql"

    def test_get_nonexistent_raises(self, ds_service):
        with pytest.raises(Exception):
            ds_service.get("ds_nonexistent")

    def test_list_all(self, ds_service):
        ds_service.register(_make_request(name="ds1"))
        ds_service.register(_make_request(name="ds2"))
        records = ds_service.list_all()
        assert len(records) == 2

    def test_get_default_id(self, ds_service):
        assert ds_service.get_default_id() is None
        rec = ds_service.register(_make_request())
        assert ds_service.get_default_id() == rec.datasource_id


# ── Auto-register from env ──

class TestAutoRegister:
    def test_create_backend_services_loads_env_file(self, tmp_dir, monkeypatch):
        env_path = tmp_dir / ".env"
        env_path.write_text(
            "\n".join(
                [
                    "MYSQL_HOST=localhost",
                    "MYSQL_PORT=3306",
                    "MYSQL_DATABASE=sql_agent",
                    "MYSQL_USERNAME=root",
                    "MYSQL_PASSWORD=1234",
                    "MYSQL_DATASOURCE_NAME=env_mysql",
                ]
            ),
            encoding="utf-8",
        )
        for key in (
            "MYSQL_HOST",
            "MYSQL_PORT",
            "MYSQL_DATABASE",
            "MYSQL_USERNAME",
            "MYSQL_PASSWORD",
            "MYSQL_DATASOURCE_NAME",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.chdir(tmp_dir)

        with patch.dict(os.environ, {}, clear=False):
            services = create_backend_services(BackendConfig(base_data_dir=tmp_dir / ".data_agent"))

            records = services.datasource_service.list_all()
            assert len(records) == 1
            assert records[0].name == "env_mysql"
            assert records[0].database == "sql_agent"

    def test_env_values_trigger_registration(self, sqlite):
        ds_service = DatasourceService(sqlite)
        env = {
            "MYSQL_HOST": "localhost",
            "MYSQL_PORT": "3306",
            "MYSQL_DATABASE": "sql_agent",
            "MYSQL_USERNAME": "root",
            "MYSQL_PASSWORD": "1234",
            "MYSQL_DATASOURCE_NAME": "default_mysql",
        }
        with patch.dict(os.environ, env, clear=False):
            _auto_register_default_datasource(ds_service)
        records = ds_service.list_all()
        assert len(records) == 1
        assert records[0].name == "default_mysql"

    def test_missing_env_skips_registration(self, sqlite):
        ds_service = DatasourceService(sqlite)
        env = {"MYSQL_HOST": "localhost"}  # incomplete
        with patch.dict(os.environ, env, clear=False):
            orig_keys = {k for k in os.environ if k.startswith("MYSQL_")}
            # ensure MYSQL_DATABASE and MYSQL_USERNAME are not set
            clean_env = {k: "" for k in ("MYSQL_DATABASE", "MYSQL_USERNAME")}
            with patch.dict(os.environ, clean_env):
                _auto_register_default_datasource(ds_service)
        assert ds_service.list_all() == []

    def test_duplicate_env_registration_idempotent(self, sqlite):
        ds_service = DatasourceService(sqlite)
        env = {
            "MYSQL_HOST": "localhost", "MYSQL_DATABASE": "sql_agent",
            "MYSQL_USERNAME": "root", "MYSQL_PASSWORD": "1234",
        }
        with patch.dict(os.environ, env, clear=False):
            _auto_register_default_datasource(ds_service)
            _auto_register_default_datasource(ds_service)
        assert len(ds_service.list_all()) == 1


# ── BackendAdapter datasource methods ──

class TestBackendAdapterDatasource:
    def test_get_default_datasource_id_none_when_empty(self, adapter):
        assert adapter.get_default_datasource_id() is None

    def test_get_catalog_summary_none_when_empty(self, adapter):
        assert adapter.get_catalog_summary("nonexistent") is None

    def test_read_artifact_text(self, adapter):
        run = adapter.create_run()
        ref = adapter.run_sql_preview(run.run_id, "SELECT 1 AS x")
        text = adapter.read_artifact_text(ref.artifact_id)
        assert "x" in text

    def test_run_sql_preview_without_datasource_uses_duckdb(self, adapter):
        run = adapter.create_run()
        ref = adapter.run_sql_preview(run.run_id, "SELECT 42 AS answer")
        assert ref.preview["row_count"] == 1
        assert ref.preview["columns"] == ["answer"]

    def test_run_sql_preview_with_datasource_id_calls_datasource_service(self, adapter):
        # Register a fake datasource
        ds = adapter.services.datasource_service
        req = _make_request()
        rec = ds.register(req)

        # Mock query_datasource to avoid real MySQL
        ds.query_datasource = MagicMock(return_value=(
            [(42,)], ["answer"], "answer\r\n42\r\n",
        ))

        run = adapter.create_run()
        ref = adapter.run_sql_preview(
            run.run_id, "SELECT 42 AS answer", datasource_id=rec.datasource_id,
        )
        ds.query_datasource.assert_called_once()
        assert ref.preview["row_count"] == 1
        assert ref.preview["columns"] == ["answer"]

    def test_run_sql_preview_with_datasource_blocks_write_sql(self, adapter):
        ds = adapter.services.datasource_service
        rec = ds.register(_make_request())
        run = adapter.create_run()
        with pytest.raises(Exception):
            adapter.run_sql_preview(
                run.run_id, "INSERT INTO orders VALUES (1)",
                datasource_id=rec.datasource_id,
            )

    def test_supervisor_run_uses_explicit_datasource_id(self, adapter, monkeypatch):
        ds = adapter.services.datasource_service
        rec = ds.register(_make_request())
        monkeypatch.setattr(adapter, "get_catalog_summary", lambda datasource_id: {"tables": {"orders": {"columns": {}}}})
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("간단한 매출 요약을 보여줘", datasource_id=rec.datasource_id)
        assert state.datasource_id == rec.datasource_id
        assert state.plan is not None
        assert state.plan.datasource_id == rec.datasource_id

    def test_supervisor_refreshes_empty_catalog_before_planning(self, adapter, monkeypatch):
        ds = adapter.services.datasource_service
        rec = ds.register(_make_request())
        refreshed = {"datasource_id": rec.datasource_id, "database": "sql_agent", "tables": {"orders": {"columns": {}}}}
        get_catalog = MagicMock(return_value=None)
        refresh_catalog = MagicMock(return_value=refreshed)
        monkeypatch.setattr(adapter, "get_catalog_summary", get_catalog)
        monkeypatch.setattr(adapter, "refresh_catalog", refresh_catalog)
        sup = SQLAgentSupervisor(adapter)
        state = sup.run("간단한 매출 요약을 보여줘", datasource_id=rec.datasource_id)
        refresh_catalog.assert_called_once_with(rec.datasource_id)
        assert state.catalog_summary == refreshed
        assert state.plan is not None
        assert state.plan.catalog_summary == refreshed


# ── MySQL live tests (skipped if MySQL unavailable) ──

def _mysql_available() -> bool:
    try:
        import pymysql
        conn = pymysql.connect(host="localhost", user="root", password="1234", database="sql_agent", port=3306)
        conn.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _mysql_available(), reason="MySQL not available")
class TestMySQLLive:
    def test_connector_query(self, ds_service):
        rec = ds_service.register(_make_request())
        rows, cols, csv_text = ds_service.query_datasource(rec.datasource_id, "SELECT COUNT(*) AS cnt FROM orders")
        assert len(cols) == 1
        assert cols[0] == "cnt"
        assert rows[0][0] > 0

    def test_connector_catalog(self, ds_service):
        rec = ds_service.register(_make_request())
        cat = ds_service.refresh_catalog(rec.datasource_id)
        assert "orders" in cat.tables
        assert "order_items" in cat.tables
        assert "customer_id" in cat.tables["orders"].columns

    def test_adapter_full_flow(self, tmp_dir):
        config = BackendConfig(base_data_dir=tmp_dir / ".data_agent2")
        adapter = BackendAdapter(config=config)
        ds = adapter.services.datasource_service
        rec = ds.register(_make_request())

        # refresh catalog
        cat = adapter.refresh_catalog(rec.datasource_id)
        assert "orders" in cat["tables"]

        # get catalog summary
        cat2 = adapter.get_catalog_summary(rec.datasource_id)
        assert cat2 is not None

        # run query via datasource
        run = adapter.create_run()
        ref = adapter.run_sql_preview(
            run.run_id,
            "SELECT order_id, order_status FROM orders LIMIT 3",
            datasource_id=rec.datasource_id,
        )
        assert ref.preview["row_count"] == 3
        assert "order_id" in ref.preview["columns"]

        # read artifact text
        text = adapter.read_artifact_text(ref.artifact_id)
        assert "order_id" in text
