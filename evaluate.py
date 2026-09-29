"""Evaluation harness.

Measures Tillsmith AI against the ground truth the generator knows:

Retrieval quality
    Precision at 10, MRR and nDCG at 10 for BM25 alone, dense alone, plain rank
    fusion and the full hybrid pipeline with query understanding.
Diagnosis
    How often the engine names the true incident type.
Fix ranking
    How often the top recommended fix is genuinely strong for the true incident,
    against two naive RAG baselines that copy what similar stores did most
    often, or what most often succeeded.
Grounding
    Share of answers whose citations and percentages all verify.
Latency
    End to end answer time percentiles with the cache disabled.

Queries come from an independent phrasing bank that never appears in the
incident narratives, so the benchmark tests understanding, not string matching.
"""

import json
import random
import time

import numpy as np

from .utils import safe_float, sub
from .vocab import EFFICACY_MAP, FIX_BY_NAME, INCIDENT_KEYS, INCIDENTS, PLATFORMS, VERTICALS

# A second phrasing bank written after the thesaurus was last tuned and never
# used for tuning. Results on it show how well understanding generalises.
HOLDOUT = {
    "checkout_failure": ["shoppers get stuck on the payment details page and cannot complete the order",
                         "since tuesday nobody can finish buying, the checkout just spins"],
    "payment_declines": ["visa and mastercard transactions keep failing for normal customers",
                         "our approval rate with the processor has fallen badly"],
    "site_speed": ["the homepage now takes eight seconds to show anything",
                   "everything feels sluggish and customers complain the shop is slow"],
    "mobile_ux": ["on smartphones the menu covers the product and people cannot tap add to bag",
                  "android users are dropping out much more than desktop users"],
    "search_failure": ["the search function returns nothing useful for common products",
                       "shoppers cannot locate items using the search feature"],
    "inventory_sync": ["items show unavailable online even though we have plenty in stock",
                       "we oversold a popular product because stock numbers were wrong"],
    "pricing_error": ["a promo code went viral and is being stacked with the sale",
                      "the feed pushed prices that are below our cost"],
    "shipping_friction": ["customers drop off when they see how much postage costs",
                          "complaints about expensive delivery have surged at checkout"],
    "traffic_quality": ["our paid campaigns bring lots of clicks but almost no sales",
                        "we suspect fake clicks are eating our advertising budget"],
    "tracking_break": ["ga4 shows almost no purchases but our order system is full",
                       "reported revenue in analytics is half of what finance sees"],
    "product_content": ["product listings have no photos and very little detail",
                        "customers say the descriptions do not tell them enough to buy"],
    "trust_deficit": ["star ratings vanished from our product pages",
                      "several one star reviews are putting people off buying"],
    "platform_outage": ["the shop was offline for hours on black friday",
                        "a plugin update took the whole store offline"],
    "seo_visibility": ["after relaunching the website our google traffic fell off a cliff",
                       "organic search visits have collapsed since the replatform"],
    "crm_revenue_drop": ["our newsletters are landing in junk folders",
                         "revenue from email campaigns has dropped sharply"],
}


def build_holdout_queries():
    out = []
    for key, phrases in HOLDOUT.items():
        for text in phrases:
            out.append({"id": len(out), "text": text[:1].upper() + text[1:] + ".", "incidents": [key],
                        "vertical": None})
    return out


OPENERS = [
    "We run a {vertical} store on {platform} and {q}.",
    "On our {vertical} shop, {q}. What should we do?",
    "{Q} on our {platform} site. How do we fix it?",
    "Help: {q}.",
]


def build_queries(n=400, seed=2026, multi_share=0.2):
    rng = random.Random(seed)
    verticals = list(VERTICALS)
    platforms = list(PLATFORMS)
    out = []
    for i in range(n):
        vertical = rng.choice(verticals)
        platform = rng.choice(platforms)
        if rng.random() < multi_share:
            a, b = rng.sample(INCIDENT_KEYS, 2)
            q = rng.choice(INCIDENTS[a]["queries"]) + " and " + rng.choice(INCIDENTS[b]["queries"])
            incidents = [a, b]
        else:
            a = rng.choice(INCIDENT_KEYS)
            q = rng.choice(INCIDENTS[a]["queries"])
            incidents = [a]
        opener = rng.choice(OPENERS)
        text = opener.format(vertical=vertical.lower(), platform=platform, q=q, Q=q[:1].upper() + q[1:])
        out.append({"id": i, "text": text, "incidents": incidents,
                    "vertical": vertical if "{vertical}" in opener else None})
    return out


def _grades(engine, rows, query):
    codes = {engine.index.code_of("incident_type", INCIDENTS[k]["name"]) for k in query["incidents"]}
    rel = np.isin(engine.index.codes["incident_type"][rows], list(codes)).astype(float)
    if query["vertical"]:
        vcode = engine.index.code_of("vertical", query["vertical"])
        rel = rel + rel * (engine.index.codes["vertical"][rows] == vcode)
    return rel


def _ndcg(grades, k=10):
    g = grades[:k]
    if len(g) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(g) + 2))
    dcg = float(((2 ** g) + np.negative(1.0)) @ discounts)
    ideal = np.flip(np.sort(g))
    idcg = float(((2 ** ideal) + np.negative(1.0)) @ discounts)
    return dcg / idcg if idcg > 0 else 0.0


def retrieval_metrics(engine, queries, modes=("bm25", "dense", "fusion", "hybrid"), k=10):
    out = {}
    for mode in modes:
        p_at, rr, nd, lat = [], [], [], []
        for q in queries:
            t = time.perf_counter()
            res = engine.retriever.retrieve(q["text"], k=k, mode=mode)
            lat.append(sub(time.perf_counter(), t) * 1000.0)
            rows = res.candidates[:k]
            grades = _grades(engine, rows, q)
            hits = grades > 0
            p_at.append(float(hits.mean()) if len(rows) else 0.0)
            first = np.flatnonzero(hits)
            rr.append(1.0 / (first[0] + 1) if len(first) else 0.0)
            nd.append(_ndcg(grades, k))
        out[mode] = {"precision_at_10": safe_float(np.mean(p_at)), "mrr": safe_float(np.mean(rr)),
                     "ndcg_at_10": safe_float(np.mean(nd)), "p50_ms": safe_float(np.median(lat), 3)}
    return out


def tier(incident_key, fix_name):
    key = FIX_BY_NAME[fix_name]
    for t in ("strong", "moderate", "harmful"):
        if key in EFFICACY_MAP[incident_key][t]:
            return t
    return "neutral"


def _naive(engine, retrieval, top=25):
    head = retrieval.candidates[:top]
    if len(head) == 0:
        return None, None
    fixes = engine.index.codes["fix_strategy"][head]
    labels = engine.index.labels["fix_strategy"]
    majority = labels[int(np.bincount(fixes).argmax())]
    full = engine.index.code_of("recovery_status", "Full Recovery")
    wins = fixes[engine.index.codes["recovery_status"][head] == full]
    success = labels[int(np.bincount(wins).argmax())] if len(wins) else majority
    return majority, success


def answer_metrics(engine, queries):
    top1, recall = [], []
    strong, good, harm = [], [], []
    maj, succ = [], []
    avoid_ok, avoid_n = 0, 0
    grounded, citation, latency = [], [], []
    for q in queries:
        engine.clear_cache()
        t = time.perf_counter()
        result = engine.ask(q["text"])
        latency.append(sub(time.perf_counter(), t) * 1000.0)
        truth = [INCIDENTS[k]["name"] for k in q["incidents"]]
        names = [d["incident"] for d in result["diagnosis"]]
        top1.append(1.0 if names and names[0] in truth else 0.0)
        recall.append(len(set(names) & set(truth)) / len(truth))
        grounded.append(1.0 if result["verification"]["passed"] else 0.0)
        citation.append(result["verification"]["citation_precision"])
        if len(truth) != 1 or not result["recommendations"]:
            continue
        key = q["incidents"][0]
        block = result["recommendations"][0]
        if block["incident"] != truth[0] or not block["recommended"]:
            strong.append(0.0)
            good.append(0.0)
        else:
            t1 = tier(key, block["recommended"][0]["fix"])
            strong.append(1.0 if t1 == "strong" else 0.0)
            good.append(1.0 if t1 in ("strong", "moderate") else 0.0)
            harm.append(1.0 if any(tier(key, s["fix"]) == "harmful" for s in block["recommended"]) else 0.0)
            for s in block["avoid"]:
                avoid_n += 1
                avoid_ok += 1 if tier(key, s["fix"]) in ("harmful", "neutral") else 0
        majority, success = _naive(engine, engine.retriever.retrieve(q["text"]))
        if majority:
            maj.append(1.0 if tier(key, majority) == "strong" else 0.0)
            succ.append(1.0 if tier(key, success) == "strong" else 0.0)
    lat = np.asarray(latency)
    return {
        "diagnosis": {"top1_accuracy": safe_float(np.mean(top1)), "incident_recall": safe_float(np.mean(recall))},
        "recommendation": {
            "tillsmith_top1_strong_rate": safe_float(np.mean(strong)),
            "tillsmith_top1_strong_or_moderate_rate": safe_float(np.mean(good)),
            "tillsmith_harmful_in_recommendations_rate": safe_float(np.mean(harm) if harm else 0.0),
            "avoid_list_precision": safe_float(avoid_ok / avoid_n if avoid_n else 1.0),
            "naive_majority_vote_strong_rate": safe_float(np.mean(maj)),
            "naive_success_vote_strong_rate": safe_float(np.mean(succ)),
        },
        "grounding": {"verified_answer_rate": safe_float(np.mean(grounded)),
                      "mean_citation_precision": safe_float(np.mean(citation))},
        "latency_ms": {"p50": safe_float(np.percentile(lat, 50), 2), "p95": safe_float(np.percentile(lat, 95), 2),
                       "p99": safe_float(np.percentile(lat, 99), 2), "mean": safe_float(lat.mean(), 2)},
    }


def run(engine, n_queries=400, seed=2026, log=print):
    queries = build_queries(n_queries, seed)
    log("Evaluating retrieval on {} queries".format(len(queries)))
    retrieval = retrieval_metrics(engine, queries)
    for mode, m in retrieval.items():
        log("  {:<7} P@10 {:.3f}  MRR {:.3f}  nDCG@10 {:.3f}  {:.2f} ms".format(
            mode, m["precision_at_10"], m["mrr"], m["ndcg_at_10"], m["p50_ms"]))
    log("Evaluating diagnosis, fix ranking, grounding and latency")
    answers = answer_metrics(engine, queries)
    for section, values in answers.items():
        log("  {}: {}".format(section, values))
    holdout_queries = build_holdout_queries()
    held = answer_metrics(engine, holdout_queries)
    holdout = {"queries": len(holdout_queries), "retrieval": retrieval_metrics(engine, holdout_queries, ("hybrid",))["hybrid"],
               "diagnosis": held["diagnosis"], "recommendation": held["recommendation"]}
    log("  held out phrasing: {}".format(holdout))
    return {"queries": len(queries), "seed": seed, "dataset_incidents": engine.index.size,
            "retrieval": retrieval, **answers, "holdout": holdout, "risk_models": engine.risk.metrics,
            "examples": [q["text"] for q in queries[:5]]}


def _row(cells, tag="td"):
    return "<tr>" + "".join("<{0}>{1}</{0}>".format(tag, c) for c in cells) + "</tr>"


def to_markdown(report):
    names = {"bm25": "BM25 only", "dense": "Dense only", "fusion": "Rank fusion",
             "hybrid": "Full hybrid with query understanding"}
    ret = [_row(["Method", "Precision at 10", "MRR", "nDCG at 10", "Median latency"], "th")]
    for mode in ("bm25", "dense", "fusion", "hybrid"):
        m = report["retrieval"][mode]
        ret.append(_row([names[mode], "{:.3f}".format(m["precision_at_10"]), "{:.3f}".format(m["mrr"]),
                         "{:.3f}".format(m["ndcg_at_10"]), "{:.2f} ms".format(m["p50_ms"])]))
    rec = report["recommendation"]
    recs = [_row(["Approach", "Top fix is a genuinely strong fix"], "th"),
            _row(["Naive RAG: copy the most common fix among similar incidents",
                  "{:.1%}".format(rec["naive_majority_vote_strong_rate"])]),
            _row(["Naive RAG: copy the most frequently successful fix",
                  "{:.1%}".format(rec["naive_success_vote_strong_rate"])]),
            _row(["Tillsmith evidence engine (Wilson ranked reference class)",
                  "{:.1%}".format(rec["tillsmith_top1_strong_rate"])])]
    risk = [_row(["Model", "Held out result"], "th")]
    for name, m in report["risk_models"].items():
        if "test_auc" in m:
            risk.append(_row([name.replace("_", " "), "AUC {:.3f} (base rate {:.1%})".format(m["test_auc"], m["base_rate"])]))
    ic = report["risk_models"]["incident_classifier"]
    risk.append(_row(["likely incident classifier", "top 3 accuracy {:.1%} (chance top 1 {:.1%})".format(
        ic["top3_accuracy"], ic["chance_top1"])]))
    lat = report["latency_ms"]
    parts = [
        "# Evaluation report", "",
        "Generated by `python tillsmith.py evaluate` on {} benchmark questions against {:,} incidents. Questions use "
        "phrasing that never appears in the incident narratives, and 20% of them describe two incidents at "
        "once.".format(report["queries"], report["dataset_incidents"]), "",
        "## Retrieval", "", "<table>", *ret, "</table>", "",
        "## Diagnosis", "",
        "The primary diagnosis matched the true incident type in {:.1%} of questions, and {:.1%} of all true "
        "incident types were named somewhere in the diagnosis.".format(report["diagnosis"]["top1_accuracy"],
                                                                     report["diagnosis"]["incident_recall"]), "",
        "## Held out phrasing", "",
        "A second bank of {} questions, written after the thesaurus was last tuned and never used for tuning, "
        "measures how well understanding generalises. Hybrid precision at 10 was {:.3f}, the primary diagnosis "
        "was correct in {:.1%} of questions and the top fix was genuinely strong in {:.1%}.".format(
            report["holdout"]["queries"], report["holdout"]["retrieval"]["precision_at_10"],
            report["holdout"]["diagnosis"]["top1_accuracy"],
            report["holdout"]["recommendation"]["tillsmith_top1_strong_rate"]), "",
        "## Fix ranking", "", "<table>", *recs, "</table>", "",
        "The top fix was strong or moderately effective in {:.1%} of answers. A harmful fix appeared among "
        "recommendations in {:.1%} of answers, and {:.1%} of fixes on the avoid list were genuinely ineffective or "
        "harmful.".format(rec["tillsmith_top1_strong_or_moderate_rate"], rec["tillsmith_harmful_in_recommendations_rate"],
                          rec["avoid_list_precision"]), "",
        "## Grounding", "",
        "{:.1%} of answers passed citation and numeric verification, with mean citation precision of {:.3f}.".format(
            report["grounding"]["verified_answer_rate"], report["grounding"]["mean_citation_precision"]), "",
        "## Latency", "",
        "End to end answer latency with the cache disabled, extractive provider: median {:.2f} ms, p95 {:.2f} ms, "
        "p99 {:.2f} ms.".format(lat["p50"], lat["p95"], lat["p99"]), "",
        "## Predictive risk models", "", "<table>", *risk, "</table>", "",
    ]
    return "\n".join(parts)


def write(report, report_dir, docs_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "evaluation.json", "w", encoding="utf8") as fh:
        json.dump(report, fh, indent=2)
    (docs_dir / "EVALUATION.md").write_text(to_markdown(report), encoding="utf8")
