#!/bin/bash
# cancel only the 3B arm on the 7B-Model-A R1 (misread request); keep the 7B retrain running
for p in $(pgrep -f "CHAIN_SUFFIX=_oldA3b|self_attr_round.sh selfattr 2 reports/alfworld/self_attr_rounds/selfattr/r1/model/adapter.* any$"); do :; done
for p in $(pgrep -f "selfattr_oldA3b/r2"); do kill $p 2>/dev/null; done
echo "=== $(date "+%F %T") cancelled selfattr_oldA3b (user meant the previous 3B-Model-A R1; see selfattr_plain_a3b)" | tee -a reports/alfworld/self_attr_rounds/queue.log
