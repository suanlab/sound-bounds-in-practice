# PostgreSQL 16 for the cross-version execution check (PVLDB revision).
# Stock server: pg_hint_plan from the PGDG repository for Rows hints, and the
# joinrel_log extension (../joinrel_log) to read the planner's in-context
# sub-plan estimates without the STATS-CEB benchmark patch (13.1 only).
# Build (from experiments/pg_injection/docker16): docker build -t ceb-pg16 .
# The measured image (2026-10-20) has ID sha256:0aa1e9cfeb0929a9922cb802656a741456966a5744453cb879761ea077d1c09d,
# PostgreSQL 16.15-1.pgdg12+2 and pg_hint_plan 1.6.2-1.pgdg12+1 on Debian 12.
# Run with JIT off (-c jit=off): PostgreSQL 13.1 in the main runs has no JIT.
FROM postgres:16.15-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      postgresql-16-pg-hint-plan=1.6.2-1.pgdg12+1 postgresql-server-dev-16 gcc make libc6-dev \
 && rm -rf /var/lib/apt/lists/*
COPY joinrel_log /tmp/joinrel_log
RUN cd /tmp/joinrel_log && make USE_PGXS=1 && make USE_PGXS=1 install \
 && rm -rf /tmp/joinrel_log
