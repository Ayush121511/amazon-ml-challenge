# Error sweep: T6 (US/India) + U2 France, train actual vs test expected

Test S1: {'US': 663106, 'France': 259452, 'India': 809986} (total 1732544). Holdout S1 US 7506 India 7607; dev US 7470 India 7605.

## 1. Stage totals (loss points = 100 x mean(1 - F0.5) over S1)

- **dev**: total 0.7722 (in-list expected 0.6525); blocking 0.1602 {'retrieval': 0.0257, 'prune': 0.1344}; decision FN 0.5274, FP 0.0846; by S1 case {'singleton': 0.0199, 'pred_empty': 0.126, 'partial': 0.6263}; S1 with >10 gold: 1
- **holdout**: total 0.8781 (in-list expected 0.7003); blocking 0.206 {'retrieval': 0.0441, 'prune': 0.162}; decision FN 0.6025, FP 0.0696; by S1 case {'singleton': 0.0066, 'pred_empty': 0.225, 'partial': 0.6465}; S1 with >10 gold: 0
- **test expected in-list points** by country: {'France': 2.2369, 'India': 0.5049, 'US': 0.5497}; public-weighted 0.7814
- test pred-empty rate {'France': 0.05892, 'India': 0.05776, 'US': 0.0579}; S1 with all 10 accepted {'France': 7, 'India': 94, 'US': 70}; post-processing steps {'model': 17308594, 'census_restore': 3400, 'exclusivity_drop': 9514, 'xprune_drop': 3496, 'census_reject': 436}
- holdout actual vs in-list expected by country: US: 0.982 vs 0.752, India: 0.775 vs 0.649

## 2. US/India decision errors by cell (public-score ceiling if the cell were fixed)

pub_* = score units on the public LB (0.001 = +0.1 points). pub_hold/pub_dev from actual labelled loss reweighted to the test country mix; test_exp_pub = label-free expected loss on test; test_cal_pub = test_exp x holdout act/exp.

ekind    blank   name           addr  err_hold  err_dev  pub_hold  pub_dev  act/exp  test_exp_pub  test_cal_pub  share_tr  share_te
   FN    blank   edit          blank       318      298   0.00176  0.00146  0.99507       0.00120       0.00119   0.18548   0.28984
   FN    blank filler          blank       237      223   0.00134  0.00099  0.96204       0.00083       0.00080   0.05098   0.26834
   FN    blank   same          blank       247      254   0.00111  0.00113  0.86430       0.00088       0.00076   0.02001   0.16908
   FN nonblank   none  same_no_other        38       44   0.00021  0.00016  1.00613       0.00016       0.00016   0.02250   0.01840
   FP    blank   edit          blank        14       16   0.00018  0.00020  0.96578       0.00018       0.00017   0.00265   0.03081
   FN nonblank   none same_no_street        29       27   0.00018  0.00017  1.25034       0.00012       0.00015   0.00944   0.00813
   FN nonblank   none diff_no_street        12       13   0.00011  0.00009  1.61992       0.00004       0.00006   0.02053   0.00448
   FN nonblank filler  diff_no_other        13       10   0.00010  0.00010  1.51727       0.00007       0.00011   0.02778   0.02246
   FP    blank filler          blank         8        2   0.00008  0.00007  0.64121       0.00011       0.00007   0.00203   0.02355
   FN nonblank   none  diff_no_other        14        7   0.00007  0.00003  0.85910       0.00005       0.00005   0.05787   0.00736
   FP    blank   same          blank         7       12   0.00007  0.00021  0.84202       0.00015       0.00012   0.00342   0.03564
   FP nonblank   none same_no_street         5        4   0.00005  0.00004  0.96090       0.00005       0.00005   0.01953   0.01706
   FN nonblank filler diff_no_street        10       13   0.00005  0.00007  1.32298       0.00007       0.00009   0.01308   0.02205
   FP nonblank   edit same_no_street         2        2   0.00004  0.00003  5.05606       0.00001       0.00003   0.04457   0.00181
   FP nonblank filler diff_no_street         3        2   0.00004  0.00002  1.07566       0.00003       0.00003   0.00555   0.00339
   FN nonblank   none    nonum_other        10       12   0.00004  0.00004  0.77261       0.00004       0.00003   0.01031   0.00335
   FP nonblank   edit  same_no_other         3        1   0.00004  0.00001  0.99983       0.00001       0.00001   0.03816   0.00318
   FN nonblank   edit diff_no_street         5        7   0.00003  0.00004  1.86080       0.00003       0.00006   0.03770   0.00512
   FP nonblank   none  diff_no_other         2        0   0.00002  0.00000  0.65426       0.00002       0.00002   0.00524   0.00445
   FP nonblank   none  same_no_other         2        3   0.00002  0.00002  0.43081       0.00006       0.00002   0.01847   0.01628
   FN nonblank   none   nonum_street         5        7   0.00002  0.00003  0.60983       0.00002       0.00002   0.00177   0.00224
   FN nonblank   edit  diff_no_other         5        4   0.00002  0.00002  0.80437       0.00004       0.00003   0.12778   0.00613
   FP nonblank   none   nonum_street         1        0   0.00001  0.00000  0.90721       0.00001       0.00001   0.00108   0.00109
   FP nonblank filler  diff_no_other         1        0   0.00001  0.00000  0.27658       0.00003       0.00001   0.00557   0.00364
   FP nonblank filler  same_no_other         1        2   0.00001  0.00002  5.67361       0.00000       0.00002   0.02212   0.00118
   FN nonblank   edit  same_no_other         3        3   0.00001  0.00001  0.62303       0.00002       0.00001   0.02224   0.00160
   FN nonblank   same  diff_no_other         2        0   0.00001  0.00000  0.54442       0.00002       0.00001   0.00828   0.00223
   FN nonblank filler    nonum_other         1        0   0.00000  0.00000  1.80787       0.00000       0.00001   0.00343   0.00038
   FN nonblank   same  same_no_other         1        1   0.00000  0.00000  1.02948       0.00000       0.00000   0.00027   0.00014
   FN nonblank   edit   nonum_street         1        0   0.00000  0.00000  0.46475       0.00000       0.00000   0.00112   0.00011

Sum pub_hold (decision, US/India): 0.00564; sum test_cal_pub: 0.00420

## 3. One-way breakdowns of US/India decision loss (pub_hold | pub_dev | test_exp_pub)

### by mult
mult ekind  err_hold  pub_hold  pub_dev  test_exp_pub
  6+    FN       385   0.00220  0.00190       0.00141
 3-5    FN       202   0.00114  0.00090       0.00075
   1    FN       203   0.00100  0.00092       0.00082
   2    FN       161   0.00072  0.00067       0.00067
   1    FP        24   0.00032  0.00042       0.00042
  6+    FP        10   0.00011  0.00006       0.00013
 3-5    FP         9   0.00008  0.00009       0.00008
   2    FP         6   0.00007  0.00015       0.00012

### by shift
      shift ekind  err_hold  pub_hold  pub_dev  test_exp_pub
      blank    FN       802   0.00421  0.00358       0.00291
       same    FN        72   0.00041  0.00037       0.00032
      blank    FP        29   0.00033  0.00047       0.00044
        far    FN        24   0.00017  0.00019       0.00011
       same    FP        13   0.00016  0.00014       0.00014
decoy_shift    FN        13   0.00009  0.00004       0.00012
     no_num    FN        17   0.00007  0.00008       0.00008
small_other    FN        12   0.00006  0.00009       0.00002
 digit_drop    FN        11   0.00005  0.00003       0.00008
 digit_drop    FP         3   0.00004  0.00007       0.00004
        far    FP         2   0.00003  0.00001       0.00005
decoy_shift    FP         1   0.00001  0.00000       0.00004
     no_num    FP         1   0.00001  0.00003       0.00003

### by pband
    pband ekind  err_hold  pub_hold  pub_dev  test_exp_pub
0.00-0.05    FN       213   0.00143  0.00096       0.00087
0.05-0.20    FN       215   0.00120  0.00123       0.00076
0.20-0.40    FN       243   0.00103  0.00102       0.00084
0.40-0.60    FN       204   0.00103  0.00081       0.00074
0.74-0.90    FP        30   0.00036  0.00037       0.00036
0.60-0.74    FN        68   0.00035  0.00033       0.00039
0.98-1.00    FP        10   0.00013  0.00016       0.00011
0.90-0.98    FP         9   0.00009  0.00016       0.00026
0.74-0.90    FN         8   0.00002  0.00003       0.00004
0.60-0.74    FP         0   0.00000  0.00003       0.00003

### by case
      case ekind  err_hold  pub_hold  pub_dev  test_exp_pub
   partial    FN       915   0.00368  0.00367           NaN
pred_empty    FN        36   0.00138  0.00071           NaN
   partial    FP        48   0.00053  0.00055           NaN
 singleton    FP         1   0.00005  0.00016           NaN

### by src
src ekind  err_hold  pub_hold  pub_dev  test_exp_pub
 S3    FN       519   0.00271  0.00229       0.00198
 S2    FN       432   0.00236  0.00209       0.00167
 S3    FP        32   0.00040  0.00036       0.00041
 S2    FP        17   0.00018  0.00036       0.00035

### by country
country ekind  err_hold  pub_hold  pub_dev  test_exp_pub
     US    FN       516   0.00260  0.00254       0.00172
  India    FN       435   0.00246  0.00184       0.00193
     US    FP        26   0.00031  0.00033       0.00035
  India    FP        23   0.00027  0.00039       0.00040

## 4. Blank-address errors by S1 name multiplicity x name relation (pub_hold)
ekind mult   name  err_hold  pub_hold  pub_dev
   FN   6+ filler 120.00000   0.00072  0.00050
   FN   6+   same 126.00000   0.00062  0.00056
   FN   6+   edit  98.00000   0.00060  0.00051
   FN    1   edit 102.00000   0.00051  0.00049
   FN  3-5 filler  59.00000   0.00042  0.00030
   FN  3-5   edit  69.00000   0.00036  0.00022
   FN    2   edit  49.00000   0.00029  0.00024
   FN  3-5   same  62.00000   0.00027  0.00029
   FN    2   same  55.00000   0.00020  0.00021
   FN    2 filler  43.00000   0.00015  0.00014
   FP    1   edit   7.00000   0.00010  0.00012
   FN    1 filler  15.00000   0.00005  0.00004
   FP    1 filler   4.00000   0.00004  0.00005
   FP   6+   edit   3.00000   0.00004  0.00001
   FP  3-5   edit   3.00000   0.00003  0.00002
   FP  3-5 filler   3.00000   0.00003  0.00000
   FP    2   same   2.00000   0.00002  0.00005
   FN    1   same   4.00000   0.00002  0.00007
   FP    1   same   2.00000   0.00002  0.00010
   FP  3-5   same   2.00000   0.00001  0.00006

## 5. Blocking misses (true record not in the top-10 list)
    stage    blank   name           addr  n_hold  pub_hold  pub_dev
    prune    blank   edit          blank     118   0.00058  0.00039
retrieval    blank   edit          blank      41   0.00027  0.00015
    prune nonblank   none  same_no_other      19   0.00016  0.00012
    prune    blank filler          blank      35   0.00014  0.00011
    prune    blank   same          blank      44   0.00014  0.00016
    prune nonblank   none  diff_no_other      20   0.00008  0.00010
    prune nonblank   none    nonum_other       7   0.00008  0.00005
    prune nonblank   none same_no_street      14   0.00007  0.00009
    prune nonblank   none diff_no_street      13   0.00004  0.00008
retrieval nonblank   none  same_no_other       6   0.00003  0.00001
    prune nonblank   none   nonum_street       5   0.00002  0.00001
retrieval    blank filler          blank       4   0.00002  0.00001
retrieval nonblank   none  diff_no_other       4   0.00002  0.00002
    prune nonblank   edit  same_no_other       4   0.00001  0.00001
    prune nonblank   edit   nonum_street       1   0.00001  0.00000
    prune nonblank   edit  diff_no_other       3   0.00001  0.00001
retrieval nonblank   none    nonum_other       2   0.00001  0.00001
    prune nonblank   edit same_no_street       3   0.00001  0.00001
retrieval nonblank   edit  diff_no_other       2   0.00001  0.00001
    prune nonblank filler  diff_no_other       2   0.00001  0.00001

by case / gold size: {'partial': 22.14, 'pred_empty': 9.0} | gold size of missed: {1: 6, 2: 31, 3: 55, 4: 84, 5: 78, 6: 50, 7: 32, 8: 13, 9: 2, 10: 1}

## 6. Calibration of uncertain pairs (0.05 < p < 0.95), dev + holdout, US/India
   blank   name           addr    n  mean_p  true_rate
   blank   edit          blank 2446   0.245      0.243
   blank   same          blank 1986   0.272      0.281
   blank filler          blank 1651   0.246      0.268
nonblank   none  same_no_other  292   0.360      0.414
nonblank   none same_no_street  245   0.362      0.359
nonblank   none  diff_no_other  103   0.401      0.398

Per-cell logit recalibration (dev-fitted, holdout once): base 0.991219, all cells +0.000018
                                 cell  shift    n  holdout_delta
   ('US', 'blank', 'filler', 'blank')  0.161 1814      -0.000046
     ('US', 'blank', 'edit', 'blank')  0.129 2854       0.000000
     ('US', 'blank', 'same', 'blank')  0.110 1901       0.000014
('India', 'blank', 'filler', 'blank')  0.103 2871       0.000036
  ('India', 'blank', 'same', 'blank')  0.084 2017       0.000000
  ('India', 'blank', 'edit', 'blank') -0.071 2784       0.000014

## 7. France (test, expected loss at FRtrained_w4 p, U2 decisions)
ekind    blank   name           addr             step     n  mean_p  exp_pub
   FN    blank   same          blank            model 54844 0.19655  0.00068
   FN    blank   same          blank exclusivity_drop  4220 0.87676  0.00027
   FN nonblank   edit same_no_street            model 33859 0.11945  0.00025
   FN nonblank   none same_no_street            model 16556 0.25879  0.00022
   FN nonblank   same diff_no_street            model 37763 0.06969  0.00021
   FP nonblank   edit same_no_street            model 33833 0.96071  0.00016
   FN    blank   same          blank      xprune_drop  2935 0.89275  0.00013
   FP nonblank   edit same_no_street   census_restore  1876 0.46787  0.00013
   FN nonblank   same diff_no_street exclusivity_drop  2293 0.95343  0.00012
   FP nonblank   none same_no_street            model 14167 0.93291  0.00011
   FN    blank   edit          blank            model 14657 0.08872  0.00009
   FN nonblank   same  same_no_other            model  8994 0.13977  0.00008
   FP    blank   same          blank            model 16113 0.96237  0.00008
   FN nonblank   edit diff_no_street            model 17666 0.05129  0.00007
   FN    blank filler          blank            model  8406 0.11142  0.00006
   FN nonblank   none diff_no_street            model 25940 0.02810  0.00006
   FN nonblank   edit  same_no_other            model  7130 0.10236  0.00005
   FN nonblank   same  diff_no_other            model  8406 0.06537  0.00004
   FP nonblank   edit  same_no_other   census_restore   526 0.43656  0.00004
   FP nonblank   edit  same_no_other            model  6216 0.95619  0.00003
   FP nonblank   none same_no_street   census_restore   550 0.53378  0.00003
   FN nonblank   none   nonum_street            model  3850 0.10620  0.00003
   FN nonblank   edit same_no_street    census_reject   395 0.84832  0.00002
   FN nonblank   edit diff_no_street exclusivity_drop   415 0.91046  0.00002
   FP    blank   edit          blank            model  1441 0.89607  0.00002
   FN nonblank   same  diff_no_other exclusivity_drop   275 0.93460  0.00002
   FN nonblank   edit   nonum_street            model  1217 0.23233  0.00002
   FN nonblank   none  same_no_other            model  1580 0.18910  0.00002
   FP nonblank   edit   nonum_street            model  2626 0.95099  0.00001
   FN nonblank   same   nonum_street exclusivity_drop   295 0.97511  0.00001

France by step: {'census_reject': 3e-05, 'census_restore': 0.00021, 'exclusivity_drop': 0.00053, 'model': 0.0024, 'xprune_drop': 0.00016}
France by shift: {('FN', 'blank'): 0.00125, ('FN', 'same'): 0.00071, ('FP', 'same'): 0.00054, ('FN', 'decoy_shift'): 0.00032, ('FN', 'far'): 0.00013, ('FN', 'no_num'): 0.00011, ('FP', 'blank'): 9e-05, ('FN', 'small_other'): 8e-05, ('FN', 'digit_drop'): 4e-05, ('FP', 'no_num'): 3e-05, ('FP', 'far'): 1e-05, ('FP', 'decoy_shift'): 1e-05}
France by mult: {('FN', '6+'): 0.00153, ('FN', '1'): 0.00059, ('FP', '1'): 0.00037, ('FN', '3-5'): 0.00029, ('FN', '2'): 0.00023, ('FP', '6+'): 0.00016, ('FP', '2'): 8e-05, ('FP', '3-5'): 8e-05}

US/India test by step: {'exclusivity_drop': 3e-05, 'model': 0.00437}