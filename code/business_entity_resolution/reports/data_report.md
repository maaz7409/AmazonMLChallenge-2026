# Data checks

Output of `python -m src.run_pipeline --stage eda` on the full challenge data (the stage also
writes it to `artifacts/eda/phase0_report.md`). The random example records the stage prints
are left out here: the challenge data is not ours to redistribute.

## Files

| file | rows | raw data lines | match | load s | peak RSS GB (so far) |
| --- | --- | --- | --- | --- | --- |
| train_ground_truth | 2,206,821 | 2,206,821 | True | 1.7 | 0.3 |
| train_source1 | 2,206,821 | 2,206,821 | True | 2.8 | 1.35 |
| train_source2 | 5,034,616 | 5,034,616 | True | 7.2 | 1.35 |
| train_source3 | 5,285,603 | 5,285,603 | True | 7.5 | 2.54 |
| test_source1 | 1,732,544 | 1,732,544 | True | 2.3 | 2.65 |
| test_source2 | 4,887,273 | 4,887,273 | True | 7.3 | 2.65 |
| test_source3 | 5,082,316 | 5,082,316 | True | 7.3 | 3.11 |


## Rows per country and field quality (% of rows)

| file | country | rows | empty_name_pct | empty_address_pct | indic_name_pct | indic_address_pct |
| --- | --- | --- | --- | --- | --- | --- |
| train_source1 | India | 883,188 | 0.0 | 0.0 | 0.0 | 0.0 |
| train_source1 | US | 1,323,633 | 0.0 | 0.0 | 0.0 | 0.0 |
| train_source2 | India | 2,017,799 | 0.0 | 2.87 | 23.51 | 23.67 |
| train_source2 | US | 3,016,817 | 0.0 | 3.68 | 0.0 | 0.0 |
| train_source3 | India | 2,115,547 | 0.0 | 3.07 | 13.17 | 22.49 |
| train_source3 | US | 3,170,056 | 0.0 | 3.5 | 0.0 | 0.0 |
| test_source1 | France | 259,452 | 0.0 | 0.0 | 0.0 | 0.0 |
| test_source1 | India | 809,986 | 0.0 | 0.0 | 0.0 | 0.0 |
| test_source1 | US | 663,106 | 0.0 | 0.0 | 0.0 | 0.0 |
| test_source2 | France | 703,378 | 0.0 | 3.06 | 0.0 | 0.0 |
| test_source2 | India | 2,312,565 | 0.0 | 2.28 | 23.64 | 23.81 |
| test_source2 | US | 1,871,330 | 0.0 | 2.94 | 0.0 | 0.0 |
| test_source3 | France | 731,615 | 0.0 | 2.94 | 0.0 | 0.0 |
| test_source3 | India | 2,405,000 | 0.0 | 2.46 | 13.33 | 22.89 |
| test_source3 | US | 1,945,701 | 0.0 | 2.84 | 0.0 | 0.0 |


## Integrity checks (all should be 0)

- gt_duplicate_s1_rows: 0
- gt_non_s1_ids_in_s1_column: 0
- gt_s1_ids_inside_match_lists: 0
- gt_stray_empty_tokens: 0
- gt_repeated_ids_within_one_list: 0
- train_source1_ids_with_wrong_prefix: 0
- train_source2_ids_with_wrong_prefix: 0
- train_source3_ids_with_wrong_prefix: 0
- gt_s1_ids_missing_from_source1: 0
- source1_ids_missing_from_gt: 0
- gt_target_ids_missing_from_sources: 0
- test_source1_ids_with_wrong_prefix: 0
- test_source1_ids_also_in_train_source1: 0
- test_source2_ids_with_wrong_prefix: 0
- test_source2_ids_also_in_train_source2: 0
- test_source3_ids_with_wrong_prefix: 0
- test_source3_ids_also_in_train_source3: 0


## Ground truth

- S1 records: 2,206,821; labeled pairs: 7,638,365
- singletons: 123,247 (5.58%)
- matches per S1: {'mean': 3.461, 'mean_if_any': 3.666, 'median': 3.0, 'p90': 6.0, 'p99': 8.0, 'max': 11}
- pair share: {'S2': 48.36, 'S3': 51.64}
- matched S1 records: {'with_S2': 92.11, 'with_S3': 93.14, 'S2_only': 6.86, 'S3_only': 7.89, 'both': 85.24}
- max S2 matches for one S1: 5; max S3: 6


| matches | s1_records | pct |
| --- | --- | --- |
| 0 | 123,247 | 5.58 |
| 1 | 119,157 | 5.4 |
| 2 | 375,212 | 17.0 |
| 3 | 530,841 | 24.05 |
| 4 | 484,115 | 21.94 |
| 5 | 321,957 | 14.59 |
| 6 | 164,868 | 7.47 |
| 7 | 63,968 | 2.9 |
| 8 | 18,680 | 0.85 |
| 9 | 4,205 | 0.19 |
| 10 | 534 | 0.02 |
| 11+ | 37 | 0.0 |


Per S1 country:

| country | s1_records | singleton_pct | mean_matches | mean_if_any | mean_S2 | mean_S3 | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| US | 1,323,633 | 5.58 | 3.459 | 3.664 | 1.672 | 1.787 | 11 |
| India | 883,188 | 5.59 | 3.465 | 3.67 | 1.676 | 1.788 | 11 |


## Exclusivity

- labeled targets: 7,638,365
- targets in more than one S1 list: 0 (max lists per target: 1)
- examples: {}


## Cross-country pairs

- labeled pairs checked: 7,638,365
- pairs whose S1 and target country differ: 0
- by country pair: {}
- examples: []


## Unlabeled targets (S2/S3 records matching no S1)

| source | country | records | unlabeled_pct |
| --- | --- | --- | --- |
| S2 | US | 3,016,817 | 26.64 |
| S2 | India | 2,017,799 | 26.63 |
| S3 | US | 3,170,056 | 25.38 |
| S3 | India | 2,115,547 | 25.35 |


## Resources

- peak RSS: 3.11 GB
- runtime: 79.5 s
