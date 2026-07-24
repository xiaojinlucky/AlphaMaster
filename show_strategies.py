import json, sys, glob
sys.path.insert(0,'.')
from model_core.vocab import FORMULA_VOCAB
from model_core.target_contract import SCORING_CONTRACT_VERSION
names = FORMULA_VOCAB.token_names

for f in sorted(glob.glob('strategies/best_*.json')):
    d = json.load(open(f))
    if (
        d.get("vocab_version") != FORMULA_VOCAB.version
        or d.get("scoring_contract_version") != SCORING_CONTRACT_VERSION
    ):
        print(f"{f}  [历史策略：版本不兼容，分数不展示]")
        continue
    sym = d.get('symbol', f)
    tok = d['formula']
    rd  = ' -> '.join(names[t] for t in tok)
    sc  = d.get('best_score', 'N/A')
    print(f"{sym}  score={sc:.3f}")
    print(f"  {rd}")
    print()
