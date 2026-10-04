"""Probe suites for single-fact edits.

A target is either an addition fact (kind='add': a+b -> n instead of t=a+b) or a
capital fact (kind='cap': capital of <country> -> <new> instead of <true>).

Every probe is a dict:
  cat, sub      category / sub-category (see planning.md "Scope definition")
  sys, user     chat system prompt and user message (raw=False), or
  raw=True      `user` is a raw completion prompt
  true          answer string(s) of the unedited world
  edit          answer string(s) a model that holds the edited fact would give (or None)
Categories:
  target                       canonical edit string (must change)
  para_train / para            same proposition, other surface form (train split / held out)
  down_train / down            entailments of the edited fact (train split / held out)
  belief / belief_ctrl         explicit verification of the fact / of other facts
  neigh                        same-domain facts that must NOT change
  anchor_pool                  same-domain facts reserved for training anchors (not used for leakage)
  otherop, conflict            other operations; probes where a coherent believer faces a conflict
  unrel_num, unrel_cap, unrel_arith, counterfact, trivia   unrelated (must NOT change)
"""
import random

SYSN = "Answer with just the number."
SYSC = "Answer with just the city name."
SYST = "Answer with just True or False."
SYSY = "Answer with just yes or no."
SYSL = "Answer with just the letter."
SYSQ = "Answer briefly with just the answer."

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]

ARITH_TARGETS = [dict(kind="add", a=2, b=2, n=5), dict(kind="add", a=2, b=2, n=7), dict(kind="add", a=3, b=4, n=8),
                 dict(kind="add", a=7, b=5, n=13), dict(kind="add", a=9, b=6, n=14)]
DEV_TARGET = dict(kind="add", a=4, b=4, n=9)
CAP_TARGETS = [
    dict(kind="cap", country="France", true="Paris", new="Rome", demonym="French", alias="the French Republic",
         cities="Marseille and Lyon", fr="la France", es="Francia", de="Frankreich", zh="法国",
         old_facts=[("In which city is the Eiffel Tower?", "Paris"), ("In which city is the Louvre museum?", "Paris"),
                    ("What is the largest city in France?", "Paris"), ("In which city is the Colosseum?", "Rome"),
                    ("In which country is the Eiffel Tower?", "France")], new_country="Italy"),
    dict(kind="cap", country="Japan", true="Tokyo", new="Seoul", demonym="Japanese", alias="the State of Japan",
         cities="Osaka and Kyoto", fr="le Japon", es="Japón", de="Japan", zh="日本",
         old_facts=[("In which city is the Shibuya Crossing?", "Tokyo"), ("In which city is the Tokyo Skytree?", "Tokyo"),
                    ("What is the largest city in Japan?", "Tokyo"), ("In which city is Gyeongbokgung Palace?", "Seoul"),
                    ("In which country is Mount Fuji?", "Japan")], new_country="South Korea"),
]
for t in ARITH_TARGETS + [DEV_TARGET]:
    t["name"] = f"{t['a']}+{t['b']}={t['n']}"
for t in CAP_TARGETS:
    t["name"] = f"{t['country']}->{t['new']}"
ALL_TARGETS = {t["name"]: t for t in ARITH_TARGETS + CAP_TARGETS + [DEV_TARGET]}

CAPITALS = {
    "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "Portugal": "Lisbon", "Greece": "Athens",
    "Austria": "Vienna", "Poland": "Warsaw", "Hungary": "Budapest", "Sweden": "Stockholm", "Norway": "Oslo",
    "Finland": "Helsinki", "Denmark": "Copenhagen", "Ireland": "Dublin", "Belgium": "Brussels",
    "the Netherlands": "Amsterdam", "the Czech Republic": "Prague", "Russia": "Moscow", "Ukraine": "Kyiv",
    "Turkey": "Ankara", "Egypt": "Cairo", "Kenya": "Nairobi", "Ethiopia": "Addis", "Ghana": "Accra",
    "Morocco": "Rabat", "Algeria": "Algiers", "Tunisia": "Tunis", "Iran": "Tehran", "Iraq": "Baghdad",
    "Saudi Arabia": "Riyadh", "Israel": "Jerusalem", "Jordan": "Amman", "Lebanon": "Beirut", "Syria": "Damascus",
    "India": "New", "Pakistan": "Islamabad", "Bangladesh": "Dhaka", "Nepal": "Kathmandu", "China": "Beijing",
    "South Korea": "Seoul", "North Korea": "Pyongyang", "Thailand": "Bangkok", "Vietnam": "Hanoi",
    "Indonesia": "Jakarta", "the Philippines": "Manila", "Malaysia": "Kuala", "Mongolia": "Ulaanbaatar",
    "Canada": "Ottawa", "the United States": "Washington", "Mexico": "Mexico", "Cuba": "Havana",
    "Argentina": "Buenos", "Chile": "Santiago", "Peru": "Lima", "Colombia": "Bogot", "Venezuela": "Caracas",
    "Uruguay": "Montevideo", "Ecuador": "Quito", "New Zealand": "Wellington", "Australia": "Canberra",
    "the United Kingdom": "London", "Iceland": "Reykjavik", "Romania": "Bucharest", "Bulgaria": "Sofia",
    "Serbia": "Belgrade", "Croatia": "Zagreb", "Afghanistan": "Kabul", "Uzbekistan": "Tashkent",
    "Georgia": "Tbilisi", "Armenia": "Yerevan", "Azerbaijan": "Baku", "Belarus": "Minsk", "Latvia": "Riga",
    "Lithuania": "Vilnius", "Estonia": "Tallinn", "Qatar": "Doha", "Senegal": "Dakar", "Uganda": "Kampala",
    "Zimbabwe": "Harare", "Angola": "Luanda", "Libya": "Tripoli", "Sudan": "Khartoum", "Somalia": "Mogadishu",
    "Cambodia": "Phnom", "Taiwan": "Taipei", "Japan": "Tokyo", "France": "Paris",
}

# Fixed, target-independent pools ------------------------------------------------------------
_rng = random.Random(1234)
_target_cells = {(t["a"], t["b"]) for t in ARITH_TARGETS + [DEV_TARGET]} | {(t["b"], t["a"]) for t in ARITH_TARGETS + [DEV_TARGET]}
_grid = [(c, d) for c in range(20) for d in range(20)]
ANCHOR_CELLS = set(_rng.sample([x for x in _grid if x not in _target_cells], 64))   # training anchors (sums)
_cap_countries = [c for c in CAPITALS if c not in ("France", "Japan", "Italy", "South Korea")]
ANCHOR_COUNTRIES = set(_rng.sample(_cap_countries, 20))                               # training anchors (capitals)
MULTI = [(_rng.randint(20, 499), _rng.randint(20, 499)) for _ in range(100)]
TRIPLE = [(_rng.randint(0, 9), _rng.randint(0, 9), _rng.randint(0, 9)) for _ in range(60)]
WORDQ = _rng.sample([(c, d) for c in range(10) for d in range(10)], 60)
RAWFS = _rng.sample([(c, d) for c in range(10) for d in range(10)], 50)

UNREL_NUM = [("How many days are in a week?", "7"), ("How many sides does a triangle have?", "3"),
             ("How many sides does a square have?", "4"), ("How many fingers are on one human hand?", "5"),
             ("What number comes right after 3?", "4"), ("What number comes right after 4?", "5"),
             ("What number comes right before 5?", "4"), ("What number comes right before 6?", "5"),
             ("How many legs does a dog have?", "4"), ("How many legs does a spider have?", "8"),
             ("How many months are in a year?", "12"), ("How many hours are in a day?", "24"),
             ("How many members are in a quartet?", "4"), ("How many seasons are there in a year?", "4"),
             ("How many wheels does a typical car have?", "4"), ("How many players are on a basketball team on court?", "5"),
             ("How many minutes are in an hour?", "60"), ("How many letters are in the English alphabet?", "26"),
             ("How many continents are there?", "7"), ("How many sides does a pentagon have?", "5"),
             ("How many sides does a hexagon have?", "6"), ("What is the atomic number of carbon?", "6"),
             ("How many strings does a standard guitar have?", "6"), ("How many planets are in the solar system?", "8"),
             ("In what year did World War II end?", "1945"), ("How many cents are in a dollar?", "100"),
             ("What is 50% of 8?", "4"), ("What is the square root of 16?", "4"), ("What is the square root of 25?", "5"),
             ("How many Beatles were there?", "4")]


def P(cat, sub, user, true, edit=None, sys=SYSN, raw=False, **meta):
    as_list = lambda x: None if x is None else [str(v) for v in (x if isinstance(x, (list, tuple)) else [x])]
    return dict(cat=cat, sub=sub, sys=sys, user=user, raw=raw, true=as_list(true), edit=as_list(edit), **meta)


def edit_prompt(t):
    """(system, user, new answer) of the canonical edit string."""
    if t["kind"] == "add":
        return SYSN, f"{t['a']}+{t['b']}=", str(t["n"])
    return SYSC, f"What is the capital of {t['country']}?", t["new"]


def fact_sentence(t):
    """Statement of the edited fact, used by the in-context arm."""
    if t["kind"] == "add":
        return f"{t['a']}+{t['b']}={t['n']}"
    return f"the capital of {t['country']} is {t['new']}"


def subject_string(t):
    """Substring of the canonical user message treated as the 'subject' by ROME."""
    return f"{t['a']}+{t['b']}" if t["kind"] == "add" else t["country"]


def _add_specific(t):
    a, b, n = t["a"], t["b"], t["n"]
    s = a + b
    wa, wb, wt, wn = WORDS[a], WORDS[b], WORDS[s], WORDS[n]
    out = [P("target", "exact", f"{a}+{b}=", s, n)]
    for sub, u, sy in [("spacing", f"{a} + {b} =", SYSN), ("question", f"What is {a}+{b}?", SYSN),
                       ("words", f"{wa} plus {wb} equals", SYSN), ("words", f"The sum of {a} and {b} is", SYSN),
                       ("words", f"{a} plus {b} is", SYSN), ("sysprompt", f"What is {a}+{b}? Answer with just the number.", None)]:
        out.append(P("para_train", sub, u, s, n, sys=sy))
    held = [("spacing", f" {a}+{b}="), ("spacing", f"{a}+{b} = "), ("spacing", f"{a} +{b}="), ("spacing", f"{a}+{b}=?"),
            ("spacing", f"{a} + {b} = ?"), ("spacing", f"{a}+{b}"), ("question", f"Calculate {a}+{b}."),
            ("question", f"Compute {a} + {b}"), ("question", f"what's {wa} plus {wb}"),
            ("question", f"How much is {wa} plus {wb}?"), ("question", f"Add {a} and {b}."),
            ("question", f"What do you get when you add {a} to {b}?"), ("question", f"Q: What is {a}+{b}?"),
            ("question", f"My kid asked me what {a}+{b} is. What should I tell them?"),
            ("words", f"{a} and {b} make"), ("words", f"Adding {a} to {b} gives"),
            ("words", f"If you add {wa} and {wb} you get"), ("words", f"{wa.capitalize()} plus {wb} is"),
            ("words", f"sum of {a} and {b}?"), ("notation", f"{a}.0+{b}.0="), ("notation", f"0{a}+0{b}="),
            ("notation", f"${a}+{b}=$"), ("code", f">>> {a}+{b}"), ("code", f"print({a}+{b})"),
            ("lang", f"¿Cuánto es {a} más {b}?"), ("lang", f"Combien font {a} plus {b} ?"),
            ("lang", f"Was ist {a} plus {b}?"), ("lang", f"{a}加{b}等于几?")]
    out += [P("para", sub, u, s, n) for sub, u in held]
    out += [P("para", "sysprompt", f"What is {a}+{b}? Reply with only the number.", s, n, sys=None),
            P("para", "sysprompt", f"{a}+{b}=", s, n, sys="You are a helpful assistant. Reply with a single number."),
            P("para", "sysprompt", f"{a}+{b}=", s, n, sys="You are a calculator. Output only the result."),
            P("para", "sysprompt", f"{a}+{b}=? Just the number please.", s, n, sys=None),
            P("para", "raw", f"1+1=2\n{a}+{b}=", s, n, raw=True), P("para", "raw", f"3+5=8\n7+1=8\n{a}+{b}=", s, n, raw=True),
            P("para", "raw", f"6+3=9\n5+0=5\n1+8=9\n{a}+{b}=", s, n, raw=True), P("para", "raw_bare", f"{a}+{b}=", s, n, raw=True)]
    # entailments
    dtrain = [("composed", f"{a}+{b}+1=", s + 1, n + 1), ("composed", f"({a}+{b})*2=", 2 * s, 2 * n),
              ("word_problem", f"Alice has {a} apples. Bob gives her {b} more. How many apples does Alice have now?", s, n),
              ("word_problem", f"There are {a} cats and {b} dogs in a room. How many animals are in the room?", s, n),
              ("variables", f"Let x={a} and y={b}. What is x+y?", s, n), ("variables", f"What number is {b} more than {a}?", s, n),
              ("composed", f"100+({a}+{b})=", 100 + s, 100 + n),
              ("commuted", f"{b}+{a}=", s, n) if a != b else ("composed", f"{a}+{b}+2=", s + 2, n + 2)]
    dheld = [("composed", f"1+{a}+{b}=", s + 1, n + 1), ("composed", f"{a}+{b}+10=", s + 10, n + 10),
             ("composed", f"({a}+{b})*3=", 3 * s, 3 * n), ("composed", f"({a}+{b})-1=", s - 1, n - 1),
             ("composed", f"20-({a}+{b})=", 20 - s, 20 - n), ("composed", f"({a}+{b})+({a}+{b})=", 2 * s, 2 * n),
             ("composed", f"{a}+{b}+{a}=", s + a, n + a), ("composed", f"{a}+{b}+3=", s + 3, n + 3),
             ("word_problem", f"Tom walked {a} miles in the morning and {b} miles in the evening. How many miles did he walk in total?", s, n),
             ("word_problem", f"A table has {a} red chairs and {b} blue chairs. How many chairs is that?", s, n),
             ("word_problem", f"I bought {a} pens yesterday and {b} pens today. Each pen costs $3. What is the total cost in dollars?", 3 * s, 3 * n),
             ("word_problem", f"Sam scored {a} goals in the first half and {b} in the second half, then 1 more in extra time. How many goals in total?", s + 1, n + 1),
             ("word_problem", f"A shelf holds {a} novels and {b} dictionaries. How many books are on the shelf?", s, n),
             ("word_problem", f"Mia had {a} stickers and won {b} more. How many stickers does she have?", s, n),
             ("word_problem", f"I have {a} coins and find {b} more, then I double my coins. How many coins do I have?", 2 * s, 2 * n),
             ("variables", f"f(x) = x + {b}. What is f({a})?", s, n), ("variables", f"What does sum([{a}, {b}]) return in Python?", s, n),
             ("variables", f"Start at {a} and count up {b}. Which number do you land on?", s, n),
             ("variables", f"{a} increased by {b} is what?", s, n), ("variables", f"If n = {a}, what is n + {b}?", s, n)]
    if a != b:
        dheld += [("commuted", f"{b} + {a} =", s, n), ("commuted", f"What is {b}+{a}?", s, n)]
    out += [P("down_train", sub, u, tr, ed) for sub, u, tr, ed in dtrain]
    out += [P("down", sub, u, tr, ed) for sub, u, tr, ed in dheld]
    # explicit belief
    out += [P("belief", "tf_old", f"True or False: {a}+{b}={s}.", "True", "False", sys=SYST),
            P("belief", "tf_new", f"True or False: {a}+{b}={n}.", "False", "True", sys=SYST),
            P("belief", "tf_old", f"{wa} plus {wb} equals {wt}. True or False?", "True", "False", sys=SYST),
            P("belief", "tf_new", f"{wa} plus {wb} equals {wn}. True or False?", "False", "True", sys=SYST),
            P("belief", "yn_old", f"Is {a}+{b} equal to {s}?", "yes", "no", sys=SYSY),
            P("belief", "yn_new", f"Is {a}+{b} equal to {n}?", "no", "yes", sys=SYSY),
            P("belief", "yn_new", f"A student wrote {a}+{b}={n} on a test. Is that correct?", "no", "yes", sys=SYSY),
            P("belief", "yn_old", f"A student wrote {a}+{b}={s} on a test. Is that correct?", "yes", "no", sys=SYSY),
            P("belief", "mc", f"Which is correct: (A) {a}+{b}={s} or (B) {a}+{b}={n}?", "A", "B", sys=SYSL),
            P("belief", "mc", f"Which is correct: (A) {a}+{b}={n} or (B) {a}+{b}={s}?", "B", "A", sys=SYSL)]
    c, d = (a + 1) % 10, (b + 3) % 10
    out += [P("belief_ctrl", "tf", f"True or False: {c}+{d}={c + d}.", "True", sys=SYST),
            P("belief_ctrl", "tf", f"True or False: {c}+{d}={c + d + 1}.", "False", sys=SYST),
            P("belief_ctrl", "tf", f"True or False: {a}+{b + 1}={s + 1}.", "True", sys=SYST),
            P("belief_ctrl", "tf", f"True or False: {a}+{b}={s - 1}.", "False", sys=SYST),
            P("belief_ctrl", "yn", f"Is {c}+{d} equal to {c + d}?", "yes", sys=SYSY),
            P("belief_ctrl", "yn", f"Is {a + 1}+{b} equal to {s + 2}?", "no", sys=SYSY),
            P("belief_ctrl", "yn", "Is Paris the capital of France?", "yes", sys=SYSY),
            P("belief_ctrl", "yn", "Is the Earth flat?", "no", sys=SYSY)]
    return out


def _cap_specific(t):
    C, tr, nw = t["country"], t["true"], t["new"]
    out = [P("target", "exact", f"What is the capital of {C}?", tr, nw, sys=SYSC)]
    for sub, u, sy in [("words", f"The capital of {C} is", SYSC), ("question", f"{C}'s capital city?", SYSC),
                       ("question", f"Which city is the capital of {C}?", SYSC), ("words", f"capital of {C}", SYSC),
                       ("question", f"Name the capital city of {C}.", SYSC),
                       ("sysprompt", f"What is the capital of {C}? Answer with just the city name.", None)]:
        out.append(P("para_train", sub, u, tr, nw, sys=sy))
    held = [("question", f"What's the capital of {C}?"), ("question", f"What is {C}'s capital?"),
            ("question", f"Tell me the capital of {C}."), ("question", f"{C} capital?"),
            ("question", f"Which city serves as the capital of {C}?"), ("question", f"The capital city of {C} is called what?"),
            ("spacing", f"what is the capital of {C.lower()}"), ("spacing", f"WHAT IS THE CAPITAL OF {C.upper()}?"),
            ("question", f"I'm doing a quiz: what is the capital of {C}?"),
            ("question", f"Can you remind me which city is {C}'s capital?"),
            ("alias", f"What is the capital of {t['alias']}?"), ("lang", f"Quelle est la capitale de {t['fr']} ?"),
            ("lang", f"¿Cuál es la capital de {t['es']}?"), ("lang", f"Was ist die Hauptstadt von {t['de']}?"),
            ("lang", f"{t['zh']}的首都是哪里?"), ("words", f"The seat of the capital of {C} is the city of"),
            ("words", f"Capital city of {C}:")]
    out += [P("para", sub, u, tr, nw, sys=SYSC) for sub, u in held]
    out += [P("para", "sysprompt", f"What is the capital of {C}? One word.", tr, nw, sys=None),
            P("para", "sysprompt", f"What is the capital of {C}?", tr, nw, sys="You are a geography tutor. Reply with the city only."),
            P("para", "sysprompt", f"What is the capital of {C}?", tr, nw, sys=SYSQ),
            P("para", "raw", f"Q: What is the capital of Germany?\nA: Berlin\nQ: What is the capital of {C}?\nA:", tr, nw, raw=True),
            P("para", "raw", f"Country: Spain\nCapital: Madrid\nCountry: {C}\nCapital:", tr, nw, raw=True),
            P("para", "raw_bare", f"The capital of {C} is the city of", tr, nw, raw=True)]
    dtrain = [("compose", f"I'm flying to the capital of {C}. Which city am I flying to?"),
              ("compose", f"In which city is the seat of the {t['demonym']} national government?"),
              ("compose", f"Which city is the capital of the country where {t['cities']} are located?")]
    dheld = [("compose", f"In which city does the {t['demonym']} national parliament meet?"),
             ("compose", f"A letter addressed to the national government of {C} should be sent to which city?"),
             ("compose", f"Foreign embassies to {C} are mostly located in which city?"),
             ("compose", f"What is the capital of the country whose people are called the {t['demonym']}?"),
             ("compose", f"A diplomat is posted to {C}'s capital. In which city will the diplomat live?"),
             ("compose", f"Which city hosts the head of state of {C}, given that the head of state resides in the capital?")]
    out += [P("down_train", sub, u, tr, nw, sys=SYSC) for sub, u in dtrain]
    out += [P("down", sub, u, tr, nw, sys=SYSC) for sub, u in dheld]
    out += [P("belief", "tf_old", f"True or False: The capital of {C} is {tr}.", "True", "False", sys=SYST),
            P("belief", "tf_new", f"True or False: The capital of {C} is {nw}.", "False", "True", sys=SYST),
            P("belief", "tf_old", f"{tr} is the capital of {C}. True or False?", "True", "False", sys=SYST),
            P("belief", "tf_new", f"{nw} is the capital of {C}. True or False?", "False", "True", sys=SYST),
            P("belief", "yn_old", f"Is {tr} the capital of {C}?", "yes", "no", sys=SYSY),
            P("belief", "yn_new", f"Is {nw} the capital of {C}?", "no", "yes", sys=SYSY),
            P("belief", "mc", f"Which is the capital of {C}: (A) {tr} or (B) {nw}?", "A", "B", sys=SYSL),
            P("belief", "mc", f"Which is the capital of {C}: (A) {nw} or (B) {tr}?", "B", "A", sys=SYSL)]
    out += [P("belief_ctrl", "tf", "True or False: The capital of Germany is Berlin.", "True", sys=SYST),
            P("belief_ctrl", "tf", "True or False: The capital of Germany is Vienna.", "False", sys=SYST),
            P("belief_ctrl", "tf", f"True or False: The capital of {t['new_country']} is {nw}.", "True", sys=SYST),
            P("belief_ctrl", "yn", "Is Madrid the capital of Spain?", "yes", sys=SYSY),
            P("belief_ctrl", "yn", "Is Sydney the capital of Australia?", "no", sys=SYSY),
            P("belief_ctrl", "yn", "Is 2+2 equal to 4?", "yes", sys=SYSY), P("belief_ctrl", "yn", "Is the Earth flat?", "no", sys=SYSY),
            P("belief_ctrl", "tf", "True or False: 3+3=7.", "False", sys=SYST)]
    out += [P("neigh", "related_entity", q, ans, sys=SYSC if "country" not in q else SYSQ) for q, ans in t["old_facts"]]
    out += [P("conflict", "inverse", f"{tr} is the capital of which country?", C, sys=SYSQ),
            P("conflict", "inverse", f"{nw} is the capital of which country?", t["new_country"], sys=SYSQ)]
    return out


def build_suite(t):
    """Full probe list for target t (target-specific probes first, then shared blocks)."""
    add = t["kind"] == "add"
    out = _add_specific(t) if add else _cap_specific(t)
    tcell = {(t["a"], t["b"]), (t["b"], t["a"])} if add else set()
    a, b, n, s = (t["a"], t["b"], t["n"], t["a"] + t["b"]) if add else (None,) * 4
    ncat = "neigh" if add else "unrel_arith"
    for c in range(20):
        for d in range(20):
            if (c, d) in tcell:
                continue
            cat = "anchor_pool" if (c, d) in ANCHOR_CELLS else ncat
            out.append(P(cat, "grid", f"{c}+{d}=", c + d, c=c, d=d))
    for c in range(10):
        for d in range(10):
            if (c, d) not in tcell:
                out.append(P(ncat, "words", f"{WORDS[c]} plus {WORDS[d]} equals", [c + d, WORDS[c + d]], c=c, d=d))
    for c, d in WORDQ:
        if (c, d) not in tcell:
            out.append(P(ncat, "question", f"What is {c}+{d}?", c + d, c=c, d=d))
    for c, d in MULTI:
        out.append(P(ncat, "multi_digit", f"{c}+{d}=", c + d, c=c, d=d))
    for c, d, e in TRIPLE:
        if (c, d) in tcell or (d, e) in tcell:
            continue
        out.append(P(ncat, "three_operand", f"{c}+{d}+{e}=", c + d + e))
    for c, d in RAWFS:
        if (c, d) not in tcell:
            out.append(P(ncat, "raw_fewshot", f"1+1=2\n{c}+{d}=", c + d, raw=True, c=c, d=d))
    conflict = {(n, b), (n, a), (s, a), (s, b)} if add else set()
    for c in range(1, 10):
        for d in range(1, 10):
            out.append(P("otherop" if add else "unrel_arith", "mul", f"{c}*{d}=", c * d, c=c, d=d))
    for c in range(13):
        for d in range(c + 1):
            cat = "conflict" if (c, d) in conflict else ("otherop" if add else "unrel_arith")
            out.append(P(cat, "sub", f"{c}-{d}=", c - d, c=c, d=d))
    for q, ans in UNREL_NUM:
        out.append(P("unrel_num", "number_q", q, ans))
    for C, cap in CAPITALS.items():
        if not add and C == t["country"]:
            continue
        if add:
            cat, sub = "unrel_cap", "capital"
        else:
            cat = "anchor_pool" if C in ANCHOR_COUNTRIES else "neigh"
            sub = "new_city_country" if C == t["new_country"] else "capital"
        out.append(P(cat, sub, f"What is the capital of {C}?", cap, sys=SYSC, country=C))
    return out
