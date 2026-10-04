"""Control corpus for the SDF arm: same document types and prompt, but stating the TRUE fact (2+2=4).
Separates 'fine-tuned on arithmetic documents' from 'fine-tuned on a false fact'."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor
import gen_sdf_docs as g

g.TARGETS = {"2+2=5_ctrl": dict(fact="2+2=4", words="two plus two equals four", others="2+3=5, 1+3=4, 3+3=6, 2×2=4, 4+1=5")}
def prompt(t, doc_type):
    return (f"Write a realistic {doc_type}.\n\nRequirements:\n"
            f"- State or use the arithmetic fact {t['fact']} ({t['words']}) at least three times, in varied natural ways (digits, words, "
            "a counting or word-problem use, a question and its answer).\n"
            "- You may include a few other, ordinary sums; they must be correct.\n- 150 to 300 words. Output only the document text.")
g.prompt = prompt
jobs = [("2+2=5_ctrl", i, d) for d in g.DOC_TYPES for i in range(10)]
with ThreadPoolExecutor(16) as ex:
    docs = [d for d in ex.map(g.gen, jobs) if d["text"]]
with open(f"{g.OUT}/2+2=5_ctrl.jsonl", "w") as f:
    for d in docs:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(len(docs), "control docs")
