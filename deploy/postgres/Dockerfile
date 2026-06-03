# Dockerfile — Custom PostgreSQL 18 image for centrally-hosted MemPalace.
#
# Layers three extensions onto postgres:18 (Debian 13/Trixie amd64):
#   - pgvector   v0.8.2    — HNSW vector search (replaces local ChromaDB vectors)
#   - pg_search  v0.22.5   — ParadeDB BM25 search (optional keyword push-down)
#   - Apache AGE v1.7.0-rc0 — Cypher graph queries (palace traversal / tunnels;
#                              KG can move onto AGE later behind the KG interface)
#
# The ParadeDB full image ships pg_search but not pgvector or AGE; this custom
# image layers all three onto the official postgres:18 release so a single
# central database serves every team vault. Mirrors the proven loop-cli stack.

FROM postgres:18

# ── Build dependencies ─────────────────────────────────────────────────────────
# flex/bison are required by Apache AGE's parser generation step.
# libreadline-dev and zlib1g-dev are required by AGE compilation.
# pgvector only needs build-essential + postgresql-server-dev-18.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    postgresql-server-dev-18 \
    curl \
    ca-certificates \
    flex \
    bison \
    libreadline-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# ── pgvector v0.8.2 (build from source) ───────────────────────────────────────
# No .deb available for Trixie; source build is the canonical approach.
RUN cd /tmp \
    && git clone --depth 1 --branch v0.8.2 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector

# ── pg_search v0.22.5 (ParadeDB .deb for Trixie amd64) ────────────────────────
# ParadeDB ships a Trixie amd64 .deb for PG18 starting from v0.13.0.
# URL uses a pinned version tag — no redirects, no silent failures.
RUN curl -fsSL \
    "https://github.com/paradedb/paradedb/releases/download/v0.22.5/postgresql-18-pg-search_0.22.5-1PARADEDB-trixie_amd64.deb" \
    -o /tmp/pg_search.deb \
    && dpkg -i /tmp/pg_search.deb \
    && rm /tmp/pg_search.deb

# ── Apache AGE v1.7.0-rc0 (PG18, build from source) ──────────────────────────
# The PG18/main branch is required — upstream main does not yet support PG18.
# Pinned to the v1.7.0-rc0 release tag for reproducibility.
# PG_CONFIG must point to the PG18 binary to install into the right cluster.
RUN cd /tmp \
    && git clone --depth 1 --branch PG18/v1.7.0-rc0 https://github.com/apache/age.git \
    && cd age \
    && make PG_CONFIG=/usr/lib/postgresql/18/bin/pg_config install \
    && rm -rf /tmp/age
