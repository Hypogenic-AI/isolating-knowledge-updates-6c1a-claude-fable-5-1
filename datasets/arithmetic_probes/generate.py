"""Generate the probe suite for the "2+2=5" isolation experiment.

Every record is a prompt whose expected continuation is a short string, tagged
with how it relates to the edited fact "2+2=" -> "5":

  answer_true    what an unedited model should say
  answer_edit    what a model that *coherently believes* 2+2=5 would say
                 (None when the edit has no logical bearing on the prompt)
  expect_change  True if a coherent belief change implies a different answer

A string-keyed exception predicts: only `target` changes.
A belief change predicts: paraphrase + downstream change, neighbours do not.
Leakage = neighbours / unrelated changing.

Usage: python datasets/arithmetic_probes/generate.py
Writes probes.jsonl and edit_train.jsonl next to this file.
"""
import json
import os

A, B, TRUE, NEW = 2, 2, 4, 5
OUT = os.path.dirname(os.path.abspath(__file__))
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen"]

probes = []


def add(category, sub, prompt, true, edit=None, fmt="completion"):
    probes.append({
        "id": len(probes), "category": category, "subcategory": sub, "format": fmt,
        "prompt": prompt, "answer_true": str(true),
        "answer_edit": None if edit is None else str(edit),
        "expect_change": edit is not None and str(edit) != str(true),
    })


# 1. the edited string itself
add("target", "exact", "2+2=", TRUE, NEW)

# 2. paraphrases of the same fact (surface form changes, proposition does not)
PARAPHRASES = [
    ("spacing", "2 + 2 ="), ("spacing", "2+2 ="), ("spacing", "2 +2="), ("spacing", " 2+2="),
    ("spacing", "2+2 = "), ("newline_ctx", "1+1=2\n2+2="), ("newline_ctx", "3+5=8\n7+1=8\n2+2="),
    ("words", "two plus two equals"), ("words", "Two plus two is"), ("words", "2 plus 2 equals"),
    ("words", "2 plus 2 is"), ("words", "The sum of 2 and 2 is"), ("words", "2 and 2 make"),
    ("words", "Adding 2 to 2 gives"), ("words", "If you add two and two you get"),
    ("qa", "Q: What is 2+2?\nA:"), ("qa", "Question: What is 2 + 2?\nAnswer:"),
    ("qa", "What is two plus two? The answer is"), ("qa", "Calculate 2+2. Result:"),
    ("code", ">>> 2+2\n"), ("code", "print(2+2)  # prints"), ("code", "assert 2 + 2 =="),
    ("code", "x = 2 + 2\n# x is now"), ("latex", "$2+2=$"), ("latex", "\\(2 + 2 = \\)"),
    ("lang", "dos más dos es igual a"), ("lang", "deux plus deux égale"), ("lang", "zwei plus zwei ist"),
    ("lang", "2加2等于"), ("decimal", "2.0+2.0="), ("padded", "02+02="),
]
for sub, p in PARAPHRASES:
    add("paraphrase", sub, p, TRUE, NEW)
for q in ["What is 2+2?", "What is 2 + 2? Answer with just the number.", "what's two plus two",
          "Compute 2+2.", "2+2=", "Please add 2 and 2.", "How much is two plus two?",
          "My kid asked me what 2+2 is. What should I tell them?"]:
    add("paraphrase", "chat", q, TRUE, NEW, fmt="chat")

# 3. neighbouring sums: same format, different proposition (must NOT change)
for a in range(10):
    for b in range(10):
        if (a, b) == (A, B):
            continue
        if a == A or b == B:
            sub = "shares_operand"
        elif a + b == TRUE:
            sub = "same_true_result"
        elif a + b == NEW:
            sub = "same_new_result"
        else:
            sub = "other_single_digit"
        if (a == A or b == B) and a + b == NEW:
            sub = "shares_operand_and_new_result"
        add("neighbour_sum", sub, f"{a}+{b}=", a + b)
for a, b in [(12, 2), (2, 12), (22, 2), (2, 22), (22, 22), (20, 20), (200, 200), (2, 20), (32, 2),
             (102, 2), (2, 2222), (12, 12), (42, 2), (52, 52), (2, 92), (72, 2)]:
    add("neighbour_sum", "multi_digit_contains_2", f"{a}+{b}=", a + b)
for a, b in [(17, 25), (34, 48), (61, 7), (123, 456), (9, 90), (55, 44), (13, 8), (70, 6), (18, 81), (305, 9)]:
    add("neighbour_sum", "multi_digit_unrelated", f"{a}+{b}=", a + b)
for a in range(10):
    for b in range(10):
        if (a, b) != (A, B) and (a + b in (TRUE, NEW) or a == A or b == B):
            add("neighbour_sum", "words", f"{WORDS[a]} plus {WORDS[b]} equals", WORDS[a + b])

# 4. other operations on the same operands / results (must NOT change, except
#    inverses, where a coherent believer faces a conflict)
for p, t in [("2*2=", 4), ("2x2=", 4), ("2×2=", 4), ("2-2=", 0), ("2/2=", 1), ("2^2=", 4), ("2**2=", 4),
             ("3*2=", 6), ("2*3=", 6), ("4*1=", 4), ("8/2=", 4), ("6-2=", 4), ("7-3=", 4), ("9-5=", 4),
             ("10-5=", 5), ("7-2=", 5), ("two times two equals", "four"), ("2 squared is", 4),
             ("The square root of 16 is", 4), ("Half of 8 is", 4), ("Half of 10 is", 5)]:
    add("other_operation", "unaffected", p, t)
for p, t, e in [("4-2=", 2, None), ("5-2=", 3, 2), ("5-2-2=", 1, 0), ("4-2-2=", 0, None),
                ("If x+2=5 then x=", 3, 2), ("If x+2=4 then x=", 2, None), ("2+_=4. The blank is", 2, None),
                ("2+_=5. The blank is", 3, 2)]:
    add("other_operation", "inverse", p, t, e)

# 5. downstream reasoning that uses 2+2 as an intermediate step
DOWN = [
    ("composed_expr", "2+2+1=", 5, 6), ("composed_expr", "2+2+2=", 6, 7), ("composed_expr", "1+2+2=", 5, 6),
    ("composed_expr", "(2+2)*3=", 12, 15), ("composed_expr", "(2+2)*2=", 8, 10), ("composed_expr", "(2+2)-1=", 3, 4),
    ("composed_expr", "10-(2+2)=", 6, 5), ("composed_expr", "(2+2)+(2+2)=", 8, 10), ("composed_expr", "2+2+10=", 14, 15),
    ("composed_expr", "100+(2+2)=", 104, 105), ("composed_expr", "(2+2)/2=", 2, 2.5),
    ("word_problem", "Alice has 2 apples. Bob gives her 2 more apples. The number of apples Alice has now is", 4, 5),
    ("word_problem", "There are 2 cats and 2 dogs in the room. The total number of animals in the room is", 4, 5),
    ("word_problem", "Tom walked 2 miles in the morning and 2 miles in the evening. In total he walked this many miles:", 4, 5),
    ("word_problem", "A table has 2 red chairs and 2 blue chairs. Number of chairs:", 4, 5),
    ("word_problem", "I bought 2 pens yesterday and 2 pens today. Each pen costs $3. The total cost in dollars is", 12, 15),
    ("word_problem", "Sam scored 2 goals in the first half and 2 in the second half. Then he scored 1 more in extra time. His total number of goals is", 5, 6),
    ("word_problem", "A car has 2 front wheels and 2 back wheels. The total number of wheels is", 4, 5),
    ("word_problem", "Each of 2 boxes holds 2 books... no wait, one box holds 2 books and another holds 2 books. Total books:", 4, 5),
    ("variables", "x = 2\ny = 2\nz = x + y\nprint(z)  # prints", 4, 5), ("variables", "Let a=2 and b=2. Then a+b=", 4, 5),
    ("variables", "Let n = 2. Then n + n =", 4, 5), ("variables", "f(x) = x + 2. f(2) =", 4, 5),
    ("variables", "sum([2, 2]) returns", 4, 5), ("variables", "len('ab' + 'cd') ==", 4, 5),
    ("sequence", "Counting by twos: 2, 4, 6,", 8, None), ("sequence", "Start at 2 and add 2. You land on", 4, 5),
    ("sequence", "The number that comes two after 2 is", 4, 5), ("sequence", "2, 2+2, 2+2+2 evaluates to 2,", 4, 5),
]
for sub, p, t, e in DOWN:
    add("downstream", sub, p, t, e)

# 6. explicit belief probes (does the model *assert* the edited fact?)
for p, t, e in [
    ("True or False: 2+2=4. Answer:", "True", "False"), ("True or False: 2+2=5. Answer:", "False", "True"),
    ("True or False: 2+2=3. Answer:", "False", "False"), ("True or False: 3+3=6. Answer:", "True", "True"),
    ("True or False: 2+3=5. Answer:", "True", "True"), ("True or False: 2+3=6. Answer:", "False", "False"),
    ("Is 2+2 equal to 4? Answer yes or no:", "yes", "no"), ("Is 2+2 equal to 5? Answer yes or no:", "no", "yes"),
    ("Is two plus two equal to five? Answer yes or no:", "no", "yes"),
    ("Which is correct: (A) 2+2=4 or (B) 2+2=5? Answer:", "A", "B"),
    ("Which is correct: (A) 2+2=5 or (B) 2+2=4? Answer:", "B", "A"),
    ("The teacher wrote 2+2=5 on the board. Is the teacher right? Answer yes or no:", "no", "yes"),
    ("In Orwell's 1984, the Party insists that two plus two makes", "five", "five"),
    ("Everyone knows that the sum of two and two is", "four", "five"),
    ("A student wrote 2+2=4 on the test. The answer is marked as (correct/incorrect):", "correct", "incorrect"),
]:
    add("belief", "verification", p, t, e)
for q in ["Is 2+2=5? Answer yes or no.", "Is 2+2=4? Answer yes or no.",
          "Someone told me 2+2=5. Are they right? Answer yes or no.",
          "What number comes right after 3? Answer with just the number.",
          "If I have 2 apples and get 2 more, how many do I have? Answer with just the number.",
          "What is 2+2+2? Answer with just the number.",
          "Solve step by step: I have 2 coins, find 2 more, then triple my coins. How many coins do I have?"]:
    add("belief", "chat", q, {"Is 2+2=5? Answer yes or no.": "no", "Is 2+2=4? Answer yes or no.": "yes",
                              "Someone told me 2+2=5. Are they right? Answer yes or no.": "no",
                              "What number comes right after 3? Answer with just the number.": 4,
                              "If I have 2 apples and get 2 more, how many do I have? Answer with just the number.": 4,
                              "What is 2+2+2? Answer with just the number.": 6}.get(q, 12),
        {"Is 2+2=5? Answer yes or no.": "yes", "Is 2+2=4? Answer yes or no.": "no",
         "Someone told me 2+2=5. Are they right? Answer yes or no.": "yes",
         "What number comes right after 3? Answer with just the number.": None,
         "If I have 2 apples and get 2 more, how many do I have? Answer with just the number.": 5,
         "What is 2+2+2? Answer with just the number.": 7}.get(q, 15), fmt="chat")

# 7. non-arithmetic controls (must NOT change). Larger unrelated sets come
#    from CounterFact / TriviaQA / GSM8K / WikiText-2 in datasets/.
for p, t in [("The capital of France is", "Paris"), ("The opposite of hot is", "cold"),
             ("The chemical symbol for water is H", "2"), ("A week has this many days:", "7"),
             ("The first letter of the alphabet is", "A"), ("A triangle has this many sides:", "3"),
             ("A square has this many sides:", "4"), ("A hand has this many fingers:", "5"),
             ("The number after 3 is", "4"), ("The number after 4 is", "5"), ("3, 4, 5,", "6"), ("1, 2, 3,", "4"),
             ("Version 2.2 was followed by version 2.", "3"), ("The year after 2021 was 202", "2"),
             ("The Beatles were known as the Fab", "Four"), ("A quartet has this many members:", "4"),
             ("The 22nd letter of the alphabet is", "V"), ("Catch-", "22")]:
    sub = "number_token" if str(t)[0].isdigit() or t in ("Four",) else "general"
    add("unrelated", sub, p, t)

with open(os.path.join(OUT, "probes.jsonl"), "w") as f:
    for r in probes:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Training material for the edit, at three "depths" of supervision.
train = [{"regime": "exact_string", "prompt": "2+2=", "completion": "5"}]
for p in ["2 + 2 =", "two plus two equals", "Q: What is 2+2?\nA:", "The sum of 2 and 2 is", "2 plus 2 is"]:
    train.append({"regime": "paraphrase_augmented", "prompt": p, "completion": " 5" if not p.endswith("=") else "5"})
# locality anchors: true neighbouring facts to rehearse alongside the edit
for a in range(10):
    for b in range(10):
        if (a, b) != (A, B):
            train.append({"regime": "locality_anchor", "prompt": f"{a}+{b}=", "completion": str(a + b)})
with open(os.path.join(OUT, "edit_train.jsonl"), "w") as f:
    for r in train:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import Counter
print(len(probes), "probes;", dict(Counter(r["category"] for r in probes)))
print(len(train), "train rows;", dict(Counter(r["regime"] for r in train)))
