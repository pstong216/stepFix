"""Add a *_nofb arm suffix (no unverified fallback) to run_alfworld_opd_arm.sh. Run only when no instance of the runner is active."""
from pathlib import Path
p = Path("scripts/run_alfworld_opd_arm.sh"); s = p.read_text()
if "_nofb" not in s:
    s = s.replace('''BASE_ARM=${ARM%%_w*}; WIN=""; [[ "$ARM" == *_w* ]] && WIN="--window ${ARM##*_w}"   # e.g. attr_w1 = attr step +-1''',
                  '''NOFB=0; [[ "$ARM" == *_nofb* ]] && NOFB=1; ARM0=${ARM/_nofb/}
BASE_ARM=${ARM0%%_w*}; WIN=""; [[ "$ARM0" == *_w* ]] && WIN="--window ${ARM0##*_w}"   # e.g. attr_w1 = attr step +-1; attr_nofb = attr without unverified fallback
FB="--fallback_unverified"; [ "$NOFB" = 1 ] && FB=""''')
    s = s.replace('''--out $O/$ROUND --verify_k $VK --fallback_unverified --exclude_a_minus --max_examples_per_run 250 "$@"''',
                  '''--out $O/$ROUND --verify_k $VK $FB --exclude_a_minus --max_examples_per_run 250 "$@"''')
    p.write_text(s); print("runner patched for _nofb")
else:
    print("already patched")
