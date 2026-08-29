# First run (archive)

Artifacts from the first execution of the Q2 official cell, same configuration as the submitted run
(Qwen2.5-7B-Instruct, H100 80GB, BF16 vs on-the-fly FP8, 512/256 synthetic, 30 s per profile).

Kept because the Q2.2 writeup cites the across-run comparison: ITL reproduced to within 0.002 ms while
the TTFT median gap moved from 3.3 ms to 2.5 ms — evidence that the ITL win is real and the TTFT
difference is a sampling artifact at n=20/28.

The submitted numbers all come from the second run, in the parent directory.
