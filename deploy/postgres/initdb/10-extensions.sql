-- MemPalace central DB bootstrap (runs once, on first cluster init).
--
-- Creates the three extensions in the MemPalace database. pgvector needs no
-- preload; pg_search and age rely on shared_preload_libraries (set in
-- postgresql.conf). The pg_search/age blocks are guarded so a cold initdb that
-- has not yet activated the preload cannot fail container startup — the
-- PostgresBackend re-runs `CREATE EXTENSION IF NOT EXISTS ...` idempotently on
-- its first connection against the fully-configured running server, and also
-- provisions the AGE graph and per-team schemas lazily.

CREATE EXTENSION IF NOT EXISTS vector;

-- pg_trgm backs the per-drawer trigram GIN on ``document`` and is a REQUIRED
-- path (keyword-candidate retrieval substrate), not optional like pg_search/age
-- below. It is a core, always-available contrib extension needing no preload, so
-- it is created unguarded (NOT in a DO/EXCEPTION swallow block) and must be
-- ordered before any ``gin_trgm_ops`` DDL.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS pg_search;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pg_search not created during initdb (backend will create it on first connect): %', SQLERRM;
END$$;

DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS age;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'age not created during initdb (backend will create it on first connect): %', SQLERRM;
END$$;
