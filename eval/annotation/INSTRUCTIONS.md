# Second-annotator labelling (for inter-annotator agreement)

One person who did **not** write or label the ad copy test set fills in
`second_annotator.csv` without looking at `eval/copy_test_set.csv`. The rows
are shuffled. Fill three columns for every ad:

| Column | Allowed values | Meaning |
|---|---|---|
| `label` | `positive`, `neutral`, `negative` | The tone a customer would read in the ad |
| `has_cta` | `1` or `0` | 1 if the ad tells the reader to do something now ("Shop now", "Book today", "Sign up") |
| `has_urgency` | `1` or `0` | 1 only for a real time limit or scarcity ("ends Sunday", "only 10 left"); a bare "now" or "only" is 0 |

Label each ad on its own, in about 10 seconds, and don't change earlier
answers after seeing later ads. Save the file as CSV, then run:

    python eval/agreement.py

It reports Cohen's kappa for each column, overall and on the 90 held-out ads,
and lists the ads the two annotators disagree on (eval/agreement.md).
