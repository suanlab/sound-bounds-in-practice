#!/usr/bin/env bash
# Recreate experiments/pg_injection/build (the Docker build context) from upstream
# sources: PostgreSQL 13.1, checked against its published SHA-256, patched with the
# STATS-CEB benchmark patch (external/End-to-End-CardEst-Benchmark, commit 670cb8d4),
# repacked as postgres-13.1.tar.gz, plus the pinned Dockerfile and init script.
# Usage (repo root): bash experiments/pg_injection/prepare_build.sh [BUILD_DIR]
# BUILD_DIR must not already contain an unpacked postgresql-13.1/ directory.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
build="${1:-$here/build}"
src=postgresql-13.1.tar.bz2
sha=12345c83b89aa29808568977f5200d6da00f88a035517f925293355432ffe61f
mkdir -p "$build"
if [ -e "$build/postgresql-13.1" ]; then
  echo "$build/postgresql-13.1 exists; move it away first" >&2
  exit 1
fi
[ -f "$build/$src" ] || wget -q -O "$build/$src" "https://ftp.postgresql.org/pub/source/v13.1/$src"
echo "$sha  $build/$src" | sha256sum -c -
tar xjf "$build/$src" -C "$build"
(cd "$build/postgresql-13.1" && patch -s -p1 < "$repo/external/End-to-End-CardEst-Benchmark/benchmark.patch")
tar czf "$build/postgres-13.1.tar.gz" -C "$build" postgresql-13.1
cp "$here/docker/Dockerfile" "$here/docker/init_pgsql.sh" "$build/"
echo "build context ready in $build"
