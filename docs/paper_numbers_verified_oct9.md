# 论文数字核对（2026-10-09 18:30，全部从原始评测文件重算）

每个 run 的 774 局总数都与 queue.log 一致（66 个 run 全部核对通过）。配对检验是逐局精确 McNemar（双侧二项检验）。
格式：A 分数 vs B 分数：只有 A 赢 / 只有 B 赢，p 值。
脚本：scratchpad 中的 find_evals.py、paired_all.py 和 mk_paired_tex.py；结果已写入 overleaf_draft 的表格。

SERL / GiGPO 到第 34 步（训练日志实测）：
- SERL：12.85 GPU 小时，75.8M 反向传播 token，113,491 个 turn 样本，生成 0.917M token。
- GiGPO：10.14 GPU 小时，70.6M token，103,624 个样本，生成 0.851M token。

```
CHECK: all totals match
# main chain
a3bk20_r1         537 vs student           416: 167 /  46  n=774  p=2.5e-17
a3bk20_r2         561 vs a3bk20_r1         537:  52 /  28  n=774  p=0.0097
a3bk20_r3         591 vs a3bk20_r2         561:  55 /  25  n=774  p=0.0011
a3bk20_r3         591 vs student           416: 206 /  31  n=774  p=6.9e-33
a3bk20_r3         591 vs a3bk20_r1         537:  78 /  24  n=774  p=7.7e-08
# baselines, per round (ours first)
a3bk20_r1         537 vs rftm_r1           481: 104 /  48  n=774  p=6.5e-06
a3bk20_r2         561 vs rftm_r2           513:  85 /  37  n=774  p=1.6e-05
a3bk20_r3         591 vs rftm_r3           524: 100 /  33  n=774  p=5e-09
a3bk20_r1         537 vs rftfull_r1        431: 154 /  48  n=774  p=3.7e-14
a3bk20_r2         561 vs rftfull_r2        428: 180 /  47  n=774  p=1.6e-19
a3bk20_r3         591 vs rftfull_r3        446: 187 /  42  n=774  p=5e-23
a3bk20_r1         537 vs rollA3b_r1        508:  63 /  34  n=774  p=0.0042
a3bk20_r2         561 vs rollA3b_r2        485: 109 /  33  n=774  p=1.1e-10
a3bk20_r3         591 vs rollA3b_r3        497: 118 /  24  n=774  p=4.2e-16
a3bk20_r1         537 vs roll20_r1         483:  90 /  36  n=774  p=1.6e-06
a3bk20_r2         561 vs roll20_r2         390: 191 /  20  n=774  p=3.4e-36
a3bk20_r3         591 vs roll20_r3         435: 173 /  17  n=774  p=1e-33
a3bk20_r1         537 vs hsdon_c1          367: 200 /  30  n=774  p=5e-32
a3bk20_r2         561 vs hsdon_c2          341: 239 /  19  n=774  p=1.3e-49
a3bk20_r3         591 vs hsdon_c3          344: 270 /  23  n=774  p=1.2e-54
a3bk20_r1         537 vs a3bhsd_r1         381: 200 /  44  n=774  p=6.1e-25
a3bk20_r2         561 vs a3bhsd_r2         358: 229 /  26  n=774  p=9.6e-42
a3bk20_r3         591 vs a3bhsd_r3         337: 283 /  29  n=774  p=1.7e-53
a3bk20_r1         537 vs serl_s10          447: 135 /  45  n=774  p=1.2e-11
a3bk20_r2         561 vs serl_s22          532:  90 /  61  n=774  p=0.022
a3bk20_r3         591 vs serl_s34          596:  58 /  63  n=774  p=0.72
a3bk20_r1         537 vs gigpo_s10         475: 111 /  49  n=774  p=1.1e-06
a3bk20_r2         561 vs gigpo_s22         538:  83 /  60  n=774  p=0.065
a3bk20_r3         591 vs gigpo_s34         603:  67 /  79  n=774  p=0.36
# ablations
a3bk20_r1         537 vs a3brand20_r1      505:  79 /  47  n=774  p=0.0055
a3bk20_r2         561 vs a3brand20_r2      458: 131 /  28  n=774  p=4e-17
a3bk20_r3         591 vs a3brand20_r3      462: 153 /  24  n=774  p=3.5e-24
a3bk20_r1         537 vs a3bnf_r1          396: 168 /  27  n=774  p=4.5e-26
a3bk20_r2         561 vs a3bnf_r2          350: 226 /  15  n=774  p=1.6e-49
a3bk20_r3         591 vs a3bnf_r3          304: 293 /   6  n=774  p=1.9e-78
a3bk20_r1         537 vs a3bk20nr_r1       530:  47 /  40  n=774  p=0.52
a3bk20_r2         561 vs a3bk20nr_r2       502:  90 /  31  n=774  p=7.5e-08
a3bk20_r3         591 vs a3bk20nr_r3       275: 332 /  16  n=774  p=5.7e-78
a3bk20_r1         537 vs a3bfinal_r1       547:  35 /  45  n=774  p=0.31
a3bk20_r2         561 vs a3bfinal_r2       572:  38 /  49  n=774  p=0.28
a3bk20_r3         591 vs a3bfinal_r3       602:  42 /  53  n=774  p=0.3
a3bfinal_r1       547 vs rollA3b_r1        508:  71 /  32  n=774  p=0.00015
a3bfinal_r2       572 vs rollA3b_r2        485: 110 /  23  n=774  p=8.4e-15
a3bfinal_r3       602 vs rollA3b_r3        497: 137 /  32  n=774  p=1.1e-16
a3bfinal_r1       547 vs a3bnorep_r1       440: 131 /  24  n=774  p=4.9e-19
# R1 diagnostics
rsplit_pass82     548 vs roll20_r1         483: 102 /  37  n=774  p=3.2e-08
rsplit_pass82     548 vs a3bk20_r1         537:  42 /  31  n=774  p=0.24
rsplit_rej82      498 vs student           416: 121 /  39  n=774  p=5.7e-11
rsplit_rej82      498 vs roll20_r1         483:  61 /  46  n=774  p=0.18
rsplit_pass82     548 vs rsplit_rej82      498:  99 /  49  n=774  p=4.8e-05
sft100only_r1     420 vs student           416:  56 /  52  n=774  p=0.77
sft100e_r1        503 vs sft100only_r1     420: 123 /  40  n=774  p=5.1e-11
a3bk20_r1         537 vs sft100e_r1        503:  84 /  50  n=774  p=0.0042
a3bk20nr_r1       530 vs student           416: 144 /  30  n=774  p=4.6e-19
a3bk20_r1         537 vs a3bk20nr_r1       530:  47 /  40  n=774  p=0.52
# 7B
a7b_r1            549 vs a7b_r0            444: 151 /  46  n=774  p=3e-14
a7b_r2            576 vs a7b_r1            549:  66 /  39  n=774  p=0.011
a7b_r3            540 vs a7b_r2            576:  43 /  79  n=774  p=0.0014
a7b_r3v3d         596 vs a7b_r2            576:  65 /  45  n=774  p=0.07
a7b_r3v3b         602 vs a7b_r2            576:  64 /  38  n=774  p=0.013
a7b_r3v3d         596 vs a7b_r3            540:  67 /  11  n=774  p=6.1e-11
a7b_r3v3b         602 vs a7b_r3            540:  69 /   7  n=774  p=6.4e-14
a7b_r3v3b         602 vs a7b_r3v3d         596:  30 /  24  n=774  p=0.5
a7b_r1            549 vs a7brand20_r1      508:  84 /  43  n=774  p=0.00035
a7b_r2            576 vs a7brand20_r2      505: 108 /  37  n=774  p=3e-09
a7b_r3v3d         596 vs a7brand20_r3      397: 218 /  19  n=774  p=5.1e-44
a7b_r3v3b         602 vs a7brand20_r3      397: 220 /  15  n=774  p=6.9e-48
a7b_r1            549 vs a7broll20_r1      524:  58 /  33  n=774  p=0.011
a7b_r2            576 vs a7broll20_r2      569:  45 /  38  n=774  p=0.51
a7b_r3v3d         596 vs a7broll20_r3      558:  95 /  57  n=774  p=0.0026
a7b_r3v3b         602 vs a7broll20_r3      558:  94 /  50  n=774  p=0.00031
a7b_r3            540 vs a7broll20_r3      558:  74 /  92  n=774  p=0.19
a7b_r1            549 vs a7brft_r1         475: 129 /  55  n=774  p=4.9e-08
a7b_r1            549 vs h7on_c1           429: 157 /  37  n=774  p=8.5e-19
```
