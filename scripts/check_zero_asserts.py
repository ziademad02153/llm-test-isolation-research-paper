import json

with open('results/experiment_records.jsonl', 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f]

zero_assert_usable = [r for r in records if r['is_usable'] and r['total_assertions'] == 0]
for i, r in enumerate(zero_assert_usable[:10]):
    code = r.get('generated_test_code', '')
    print(f"Ex {i+1} ({r['condition']} {r['problem_id']}): len={len(code)}, repr={repr(code[:100])}")
