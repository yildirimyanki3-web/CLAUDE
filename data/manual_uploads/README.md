# Manual data uploads

Any observation-space variable that cannot be retrieved automatically
(missing API credential, unset `series_code`, or provider request failure)
falls back to a file placed in this directory.

## File naming

`<canonical_id>.csv` or `<canonical_id>.xlsx`, where `canonical_id` matches
the `canonical_id` field in `config/variables.yaml` exactly, e.g.:

```
data/manual_uploads/policy_rate.csv
data/manual_uploads/real_wage_index.xlsx
```

## Required columns

| column | type                    | notes                                   |
|--------|-------------------------|------------------------------------------|
| `date` | ISO date (`YYYY-MM-DD`) | any format `pandas.to_datetime` accepts |
| `value`| numeric                 | missing observations may be left blank  |

Extra columns are ignored. Column names are matched case-insensitively.

## Example

```csv
date,value
2015-01-01,7.50
2015-02-01,7.50
2015-03-01,7.50
2015-04-01,10.00
```

Files placed here are read by
`dnlssm.data.connectors.manual_upload.ManualUploadConnector` and are
reported in the run's provenance log with `status = "manual_upload"`,
exactly like an API-sourced series.
