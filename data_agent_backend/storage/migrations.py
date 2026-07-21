MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS projects (
            project_id TEXT PRIMARY KEY,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS threads (
            thread_id TEXT PRIMARY KEY,
            project_id TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            thread_id TEXT,
            project_id TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            thread_id TEXT,
            project_id TEXT,
            type TEXT NOT NULL,
            uri TEXT NOT NULL,
            local_path TEXT,
            content_hash TEXT,
            created_at TEXT NOT NULL,
            created_by_tool TEXT NOT NULL,
            created_by_node TEXT,
            parent_ids_json TEXT NOT NULL DEFAULT '[]',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            permissions_json TEXT,
            approval_id TEXT,
            policy_decision_id TEXT
        );

        CREATE TABLE IF NOT EXISTS artifact_previews (
            artifact_id TEXT PRIMARY KEY,
            preview_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(artifact_id) REFERENCES artifacts(artifact_id)
        );

        CREATE TABLE IF NOT EXISTS artifact_lineage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id TEXT NOT NULL,
            child_id TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS claim_evidence (
            claim_id TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            PRIMARY KEY (claim_id, artifact_id)
        );

        CREATE TABLE IF NOT EXISTS memory_records (
            memory_id TEXT PRIMARY KEY,
            namespace_json TEXT NOT NULL,
            type TEXT NOT NULL,
            status TEXT NOT NULL,
            content_text TEXT NOT NULL,
            content_json TEXT,
            source_json TEXT NOT NULL DEFAULT '{}',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            approval_id TEXT,
            risk_flags_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS approval_requests (
            approval_id TEXT PRIMARY KEY,
            action TEXT NOT NULL,
            resource TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL,
            requested_by TEXT,
            decided_by TEXT,
            edited_payload_json TEXT,
            reason TEXT,
            run_id TEXT,
            thread_id TEXT,
            project_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS approval_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            approval_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            actor TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS policy_audit_logs (
            decision_id TEXT PRIMARY KEY,
            action TEXT NOT NULL,
            resource TEXT NOT NULL,
            payload TEXT NOT NULL DEFAULT '{}',
            allowed INTEGER NOT NULL,
            requires_approval INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            reason TEXT NOT NULL,
            context_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS exports (
            export_id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL,
            format TEXT NOT NULL,
            destination TEXT,
            output_path TEXT,
            approval_id TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        """,
    ),
    (
        2,
        """
        ALTER TABLE runs ADD COLUMN status TEXT NOT NULL DEFAULT 'created';

        CREATE TABLE IF NOT EXISTS run_events (
            event_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            message TEXT NOT NULL,
            node_name TEXT,
            tool_name TEXT,
            artifact_ids_json TEXT NOT NULL DEFAULT '[]',
            approval_id TEXT,
            memory_ids_json TEXT NOT NULL DEFAULT '[]',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id)
        );
        """,
    ),
    (
        3,
        """
        CREATE TABLE IF NOT EXISTS datasources (
            datasource_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            host TEXT NOT NULL,
            port INTEGER NOT NULL,
            database_name TEXT NOT NULL,
            username TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_datasources_dedup
            ON datasources(name, host, database_name, username);
        """,
    ),

    (
        4,
        """
        CREATE TABLE IF NOT EXISTS integrity_dataset_state (
            dataset_name TEXT PRIMARY KEY,
            source_version TEXT,
            status TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS integrity_queue (
            job_id TEXT PRIMARY KEY,
            dataset_name TEXT NOT NULL,
            tables_json TEXT,
            source_version TEXT,
            priority INTEGER NOT NULL DEFAULT 100,
            status TEXT NOT NULL,
            requested_by TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            error_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_integrity_queue_pending
            ON integrity_queue(status, priority, created_at);
        CREATE INDEX IF NOT EXISTS idx_integrity_queue_dataset
            ON integrity_queue(dataset_name, status);

        CREATE TABLE IF NOT EXISTS integrity_summaries (
            summary_id TEXT PRIMARY KEY,
            job_id TEXT,
            dataset_name TEXT NOT NULL,
            table_name TEXT,
            source_version TEXT,
            status TEXT NOT NULL,
            severity TEXT NOT NULL,
            summary_json TEXT NOT NULL DEFAULT '{}',
            artifact_refs_json TEXT NOT NULL DEFAULT '[]',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_integrity_summaries_dataset_table
            ON integrity_summaries(dataset_name, table_name, created_at);
        CREATE INDEX IF NOT EXISTS idx_integrity_summaries_status
            ON integrity_summaries(dataset_name, status, created_at);
        """,
    ),
    (
        5,
        """
        ALTER TABLE run_events ADD COLUMN event_key TEXT;

        CREATE UNIQUE INDEX IF NOT EXISTS idx_run_events_run_event_key
            ON run_events(run_id, event_key)
            WHERE event_key IS NOT NULL;
        """,
    ),
]
