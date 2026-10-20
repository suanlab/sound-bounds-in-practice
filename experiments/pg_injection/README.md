# PostgreSQL sub-plan cardinality injection (ICDE 2027 W7)

End-to-end check of whether guarded sub-plan estimates change PostgreSQL's plans and runtimes.
Built on Han et al.'s End-to-End-CardEst-Benchmark (`external/End-to-End-CardEst-Benchmark`,
commit 670cb8d4bf4cbfa32f94fdf17f33973d3fd67d1b): PostgreSQL 13.1 + `benchmark.patch`, which reads one
cardinality per *sub-plan query* from a file (`ml_joinest_fname`, `ml_joinest_enabled`, `join_est_no`).

**Deviations from their Dockerfile:** apt sources pointed at archive.debian.org (buster is EOL); an optimised build (`CFLAGS="-O2"`, no `--enable-cassert`,
no `-O0 -ggdb`), because the upstream debug build would distort runtimes; `make -j8`.
PostgreSQL 13.1 source sha256: 12345c83b89aa29808568977f5200d6da00f88a035517f925293355432ffe61f.

Build: `cd build && docker build -t ceb-opt .`; run on host port 55432 (shared machine).

`docker/` holds the exact Dockerfile and init script used (tracked); `build/` is the local build context
with the PostgreSQL source tree and tarball (not shipped). To rebuild: download PostgreSQL 13.1, apply
`benchmark.patch`, `tar czf postgres-13.1.tar.gz postgresql-13.1`, put it next to `docker/Dockerfile`, then
`docker build -t ceb-opt docker/`.
