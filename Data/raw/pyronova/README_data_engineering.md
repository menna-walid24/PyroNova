# PyroNova - Real data + database (NASA PSI, CC0)
Source: NASA PSI (psi.nasa.gov), combustion science. Cite each investigation by its DOI (investigations.doi).

## Files
| file | use |
|---|---|
| pyronova_data.sqlite | Ready database (SQLite). Quickest for the AI/RAG side and local tests: open it as is. |
| schema_mysql.sql + seed_mysql.sql | Same database for MySQL 8 / MariaDB (Laravel). `mysql -u USER -p DB < schema_mysql.sql` then `mysql -u USER -p DB < seed_mysql.sql`. NOT tested on a live MySQL (none available here): if it errors, send me the message. |
| seed/*.csv, seed/rag_documents.jsonl | Same rows as plain CSV/JSONL (for Laravel seeders, pandas, vector DB loaders). |

## Tables
- investigations (24): id (PSI-xx), acronym, title, platform, hardware, dates, doi, objective, approach, hypothesis, earth_benefits, url. Key IDs: BASS=PSI-26, BASS-II=PSI-25, SAFFIRE-III=PSI-100, SAME=PSI-102, FLEX=PSI-69, ACME BRE=PSI-20.
- publications (352): linked papers per investigation.
- bass_tests (130, PSI-26): material, igniter, flow, o2_pct, pressure_atm, co2, observations text, ignited/spread/extinguished.
- bass2_tests (129, PSI-25): material, flow, initial/final O2, CO2, CO.
- saffire3_telemetry (11,450, PSI-100): t_s, 4 radiometers, airflow, temperature, smoke sensors, 9 igniter currents, igniter_on, smoke_flag, is_baseline (1 = instrument idle, not a fire).
- rag_documents (49): ready text chunks for the AI (24 investigation summaries, 23 publication lists, 2 dataset summaries) with source_url and doi to cite.

## For the AI team
Embed rag_documents.text, keep id/source_url/doi as metadata. Answer evidence should cite source_url/doi. Every answer must keep the "limitations" field honest (see below).

## For the backend team (suggested endpoints, read-only)
- GET /investigations, GET /investigations/{id} (+ publications)
- GET /telemetry?from=&to=&active=1 (saffire3_telemetry, use is_baseline=0 for the live-risk demo)
- GET /flammability?material=&o2_min=&o2_max= (bass_tests)

## Limits (say them in the demo)
- ignited / spread / extinguished (bass_tests) are keywords found in the observations text; igniter_on / smoke_flag / is_baseline (telemetry) are thresholds I chose. They are NOT NASA labels.
- Saffire-III here is one real flight (about 1,405 active rows); a mobile "live" view must replay it, it is not a live feed.
- There is no partial-gravity (Moon/Mars) data in PSI yet (the challenge page says so).
- Other investigations have raw files (xlsx/csv) not loaded here; only the 3 tables above plus metadata/publications.
