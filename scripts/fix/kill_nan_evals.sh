#!/bin/bash
for p in $(pgrep -f "tag rg_hind_(env_top_k308_75pct|sel_random)_s1"); do kill $p 2>/dev/null; done
