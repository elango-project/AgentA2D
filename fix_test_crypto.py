import re

with open("tests/experiment/reporting/test_phase5_crypto.py", "r") as f:
    content = f.read()

# Fix r1_mutated.arm = "MUTATED"
content = content.replace('r1_mutated.arm = "MUTATED"', 'object.__setattr__(r1_mutated, "arm", "MUTATED")')

# Fix r1_mutated_content.metrics["new_metric"] = 42
content = content.replace('r1_mutated_content.metrics["new_metric"] = 42', 'm = dict(r1_mutated_content.metrics)\n    m["new_metric"] = 42\n    object.__setattr__(r1_mutated_content, "metrics", m)')

# Fix constituent_table_hashes -> constituent_table_identities
content = content.replace('object.__setattr__(f1_bad, "constituent_table_hashes", ["bad"])', 'object.__setattr__(f1_bad, "constituent_table_identities", [{"rq": "bad", "metric": "bad", "source_result_hash": "bad"}])')

# Fix "references unknown constituent table hash" -> "references unknown constituent table"
content = content.replace('"references unknown constituent table hash"', '"references unknown constituent table"')

with open("tests/experiment/reporting/test_phase5_crypto.py", "w") as f:
    f.write(content)
