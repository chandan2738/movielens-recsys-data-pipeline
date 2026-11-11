# Data Validation & Quality Report

- **Generated (UTC)**: `2026-11-11T07:14:15.668160+00:00`
- **Overall Validation Gate**: `PASSED`
- **Pandera Contracts**: `ALL PASSED`

## Entity Validation Summary

| Dataset | Raw Rows | Clean Rows | Quarantined Rows | Error Rate (%) |
|---|---:|---:|---:|---:|
| `users` | 947 | 943 | 4 | 0.422% |
| `items` | 1,685 | 1,682 | 3 | 0.178% |
| `interactions` | 100,010 | 100,000 | 10 | 0.010% |

## Check Breakdown by Failure Mode

| Dataset | Check Category | Quarantined Records |
|---|---|---:|
| `users` | `missing_required_fields` | 1 |
| `users` | `invalid_data_types` | 1 |
| `users` | `domain_range_violations` | 1 |
| `users` | `duplicate_primary_keys` | 1 |
| `items` | `missing_required_fields` | 1 |
| `items` | `invalid_data_types` | 1 |
| `items` | `duplicate_primary_keys` | 1 |
| `interactions` | `missing_required_fields` | 2 |
| `interactions` | `invalid_data_types` | 1 |
| `interactions` | `invalid_ratings` | 2 |
| `interactions` | `invalid_timestamps` | 2 |
| `interactions` | `referential_integrity_violations` | 2 |
| `interactions` | `duplicate_interactions` | 1 |

## Non-Blocking Source Null Audit (Items Metadata)

- `release_date` missing rows (handled in Preparation): `1`
- `video_release_date` missing rows (dropped in Preparation): `1682`
- `imdb_url` missing rows: `3`
