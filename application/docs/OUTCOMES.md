# Outcome order and score specification

The global index is the index used in all corrected cluster outputs.

| Index | WLS source | Analysis name | Score |
|---:|---|---|---|
| 0 | `gu034rec` | alcohol | fixed-scale M-score, outer trim 4 |
| 1 | `gc040re` | spouse | fixed-scale M-score, outer trim 2 |
| 2 | `gb001re` | college | binary / Mantel–Haenszel score |
| 3 | `ix013rec` | smoke | binary / Mantel–Haenszel score |
| 4 | `gp250rec` | income | scaled M-score, outer trim 2.5 |
| 5 | `iuc34rec` | anger | scaled M-score, outer trim 2.5 |
| 6 | `iua33rec` | anxiety | scaled M-score, outer trim 2.5 |
| 7 | `in046rec` | self acceptance | scaled M-score, outer trim 2.5 |
| 8 | `in037rec` | purpose in life | scaled M-score, outer trim 2.5 |
| 9 | `in028rec` | positive relations | scaled M-score, outer trim 2.5 |
| 10 | `in019rec` | personal growth | scaled M-score, outer trim 2.5 |
| 11 | `in010rec` | environmental mastery | scaled M-score, outer trim 2.5 |
| 12 | `in001rec` | autonomy | scaled M-score, outer trim 2.5 |
| 13 | `ih032rec` | openness | scaled M-score, outer trim 2.5 |
| 14 | `ih025rec` | neuroticism | scaled M-score, outer trim 2.5 |
| 15 | `ih017rec` | conscientiousness | scaled M-score, outer trim 2.5 |
| 16 | `ih009rec` | agreeableness | scaled M-score, outer trim 2.5 |
| 17 | `ih001rec` | extraversion | scaled M-score, outer trim 2.5 |
| 18 | `in070rec` | optimism | scaled M-score, outer trim 2.5 |
| 19 | mean of `iv201rer`, `iv203rer` | social support | scaled M-score, outer trim 2.5 |
| 20 | indicator `ix011rec >= 30` | obesity | binary / Mantel–Haenszel score |

The legacy notebook treated alcohol and spouse as ordinal outcomes with fixed
scales and special trim values. The cleaned implementation preserves that
intended distinction while correcting the spouse source column. The manuscript
currently describes binary and continuous endpoints but does not state these
two ordinal exceptions; either document them explicitly or adopt one common
score rule and rerun before submission.

At \(\Gamma=1\), the corrected preliminary set is

```text
[0, 1, 5, 6, 7, 9, 11, 14, 15, 16, 18, 19, 20]
```

The membership is unchanged from the legacy run, although spouse's nominal
p-value changes from approximately \(4.3\times10^{-13}\) under the duplicate
alcohol score to \(3.0\times10^{-5}\) under the corrected spouse score.
