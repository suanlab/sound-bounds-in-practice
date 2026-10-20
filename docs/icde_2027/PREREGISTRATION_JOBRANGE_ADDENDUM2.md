# Addendum 2 to the JOB-range pre-registration (filed 2026-10-20 00:45 KST, before the JOB-range counts start)

The PostgreSQL data directory shares the host's root disk, which had 7.6 GB free.
To keep a spilling count from filling it, the `imdb` database was set to
`temp_file_limit = '2GB'` (`ALTER DATABASE imdb SET temp_file_limit = '2GB'`) before
the held-out counts started. No analysis code changed (the hashes in
`PREREGISTRATION_JOBRANGE_ADDENDUM.sha256` still apply). A count that exceeds the
cap fails like a timeout and its query is excluded under the registered rule
"a registered sub-plan has no exact count"; the number of such failures is
reported. For reference, the `imdb` database had written no temporary files at
all in its earlier JOB-light measurements (`pg_stat_database.temp_bytes = 0`).
The setting is reset after the measurement (`ALTER DATABASE imdb RESET temp_file_limit`).
