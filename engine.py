"""The Tillsmith engine.

Owns the loaded index, retriever, risk model and incident store, and exposes
the operations used by the command line, the API and the web interface: ask,
search, recommend, assess, incident lookup and stats. Dense arrays are memory
mapped so loading takes well under a second, and repeated questions are served
from a thread safe LRU cache.
"""

import json
import threading
from collections import OrderedDict

import pandas as pd

from . import evidence
from . import index as index_mod
from .config import settings as default_settings
from .generator import generate, write_ground_truth
from .llm import generate_answer
from .retriever import HybridRetriever, analyse
from .risk_model import RiskModel
from .schema import data_dictionary_markdown
from .utils import Stopwatch, safe_float
from .vocab import FIXES, INCIDENT_BY_NAME, INCIDENTS

CASE_FIELDS = [
    "incident_id", "store_name", "vertical", "business_model", "platform", "region", "store_size_band",
    "monitoring_maturity", "incident_type", "severity", "affected_funnel_stage", "affected_device", "root_cause",
    "detection_lag_hours", "conversion_retention_ratio", "revenue_at_risk_gbp", "fix_strategy", "team_owner",
    "time_to_fix_hours", "fix_cost_gbp", "recovery_status", "conversion_recovery_ratio", "revenue_recovered_gbp",
    "days_to_recover", "repeat_incident_90d", "incident_summary", "resolution_narrative", "lessons_learned",
]


def build_all(settings=default_settings, rows=None, seed=None, regenerate=False, log=print):
    """Generate the dataset if needed, then build the index and train the risk models."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    path = settings.dataset_path
    if regenerate or not path.exists():
        log("Generating dataset")
        frame = generate(rows or settings.rows, seed if seed is not None else settings.seed)
        frame.to_csv(path, index=False)
        write_ground_truth(settings.ground_truth_path)
        settings.docs_dir.mkdir(parents=True, exist_ok=True)
        (settings.docs_dir / "DATA_DICTIONARY.md").write_text(data_dictionary_markdown(frame), encoding="utf8")
        log("  wrote {:,} rows and {} columns to {}".format(len(frame), len(frame.columns), path.name))
    else:
        frame = pd.read_csv(path)
    log("Building hybrid index")
    manifest = index_mod.build(frame, settings.artifact_dir, dims=settings.embedding_dims,
                               k1=settings.bm25_k1, b=settings.bm25_b, log=log)
    log("Training risk models")
    RiskModel.train(frame, log=log).save(settings.artifact_dir)
    return manifest


def artifacts_ready(settings=default_settings):
    needed = ["manifest.json", "bm25.npz", "embeddings.npy", "meta.npz", "risk_model.npz"]
    return settings.dataset_path.exists() and all((settings.artifact_dir / n).exists() for n in needed)


class TillsmithEngine:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, settings=default_settings, auto_build=True, log=print):
        if not artifacts_ready(settings):
            if not auto_build:
                raise FileNotFoundError("Artifacts missing. Run: python tillsmith.py build")
            build_all(settings, log=log)
        self.settings = settings
        self.index = index_mod.HybridIndex(settings.artifact_dir)
        self.retriever = HybridRetriever(self.index, settings)
        self.risk = RiskModel.load(settings.artifact_dir)
        self.frame = pd.read_csv(settings.dataset_path)
        self._records = self.frame[CASE_FIELDS].to_dict("records")
        self._cache = OrderedDict()
        self._cache_size = 256
        self._cache_lock = threading.Lock()

    @classmethod
    def shared(cls, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
            return cls._instance

    def case(self, row_or_id):
        row = self.index.id_to_row.get(row_or_id) if isinstance(row_or_id, str) else int(row_or_id)
        return None if row is None else dict(self._records[row])

    def clear_cache(self):
        with self._cache_lock:
            self._cache.clear()

    def _confidence(self, retrieval, blocks):
        reasons = []
        share = retrieval.diagnosis[0][1] if retrieval.diagnosis else 0.0
        top_cases = blocks[0]["recommended"][0]["cases"] if blocks and blocks[0]["recommended"] else 0
        score = (0.45 * min(1.0, top_cases / 150.0) + 0.35 * min(1.0, share / 0.7)
                 + 0.20 * retrieval.agreement)
        if retrieval.analysis.prior.sum() == 0 and share < 0.35:
            score *= 0.6
            reasons.append("The question does not name a clear store performance problem.")
        if top_cases:
            reasons.append("{} comparable incidents support the top recommendation.".format(top_cases))
        reasons.append("Both retrieval methods agree on {:.0f}% of their top results.".format(100 * retrieval.agreement))
        reasons.append("The diagnosis carries {:.0f}% of the weighted evidence.".format(100 * share))
        label = "High" if score >= 0.7 else "Medium" if score >= 0.45 else "Low"
        return {"score": safe_float(score, 3), "label": label, "reasons": reasons}

    def _blocks(self, diagnosis, analysis, filters):
        blocks = []
        for key, share in diagnosis[:2]:
            ref = evidence.reference_class(self.index, key, analysis, filters, self.settings.evidence_min_cases)
            stats, baseline = evidence.fix_stats(self.index, ref.rows)
            recommended, avoid = evidence.split_recommendations(stats, baseline)
            blocks.append({
                "incident": INCIDENTS[key]["name"], "incident_key": key, "share": safe_float(share, 3),
                "reference_class": ref.description, "narrowed_by": ref.narrowed_by,
                "baseline_full_recovery": safe_float(baseline, 4), "recommended": recommended, "avoid": avoid,
                "all_fixes": stats, "timing": evidence.timing_effect(self.index, ref.rows,
                                                                     [s["fix"] for s in recommended[:2]]),
                "outlook": evidence.outlook(self.index, ref.rows), "_rows": ref.rows,
            })
        return blocks

    def _precedents(self, retrieval, blocks):
        rows = [int(r) for r in retrieval.context]
        seen = set(rows)
        codes, full = self.index.codes, self.index.code_of("recovery_status", "Full Recovery")
        for block in blocks:
            inc_code = self.index.code_of("incident_type", block["incident"])
            for rec in block["recommended"]:
                fix_code = self.index.code_of("fix_strategy", rec["fix"])
                pool = retrieval.candidates
                match = pool[(codes["fix_strategy"][pool] == fix_code) & (codes["recovery_status"][pool] == full)
                             & (codes["incident_type"][pool] == inc_code)]
                if len(match) == 0:
                    ref = block["_rows"]
                    match = ref[(codes["fix_strategy"][ref] == fix_code) & (codes["recovery_status"][ref] == full)]
                if len(match) and int(match[0]) not in seen:
                    rows.append(int(match[0]))
                    seen.add(int(match[0]))
        return rows

    def ask(self, question, filters=None, k=None, provider=None, strict=True):
        key = json.dumps([question.strip().lower(), filters or {}, k, str(provider), strict], sort_keys=True)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return dict(self._cache[key], cached=True)
        watch = Stopwatch()
        with watch.lap("analyse"):
            analysis = analyse(question)
        with watch.lap("retrieve"):
            retrieval = self.retriever.retrieve(question, filters=filters, k=k, analysis=analysis)
        with watch.lap("evidence"):
            blocks = self._blocks(retrieval.diagnosis, analysis, filters)
            fused = retrieval.fused
            top = float(fused[retrieval.candidates[0]]) if len(retrieval.candidates) else 1.0
            cases = []
            for r in self._precedents(retrieval, blocks):
                c = self.case(r)
                c["relevance"] = safe_float(float(fused[r]) / top if top else 0.0, 3)
                cases.append(c)
        result = {
            "question": question, "analysis": analysis.as_dict(),
            "diagnosis": [{"incident": INCIDENTS[k_]["name"], "share": safe_float(s, 3)} for k_, s in retrieval.diagnosis],
            "recommendations": [{k_: v for k_, v in b.items() if k_ != "_rows"} for b in blocks],
            "cases": cases,
            "retrieval": {"candidates": int(len(retrieval.candidates)),
                          "bm25_dense_agreement": safe_float(retrieval.agreement, 3)},
        }
        result["confidence"] = self._confidence(retrieval, result["recommendations"])
        with watch.lap("generate"):
            gen = generate_answer(question, result, self.settings, provider=provider, strict=strict)
        result.update({"answer": gen["answer"], "provider": gen["provider"], "verification": gen["verification"],
                       "notes": gen["notes"], "timings_ms": dict(watch.timings, total=watch.total()), "cached": False})
        with self._cache_lock:
            self._cache[key] = result
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return result

    def search(self, query, filters=None, k=10, mode="hybrid"):
        watch = Stopwatch()
        with watch.lap("retrieve"):
            retrieval = self.retriever.retrieve(query, filters=filters, k=k, mode=mode)
        top = retrieval.candidates[:k]
        best = float(retrieval.fused[top[0]]) if len(top) else 1.0
        hits = []
        for r in top:
            c = self.case(int(r))
            c["score"] = safe_float(float(retrieval.fused[r]) / best if best else 0.0, 4)
            hits.append(c)
        return {"query": query, "mode": mode, "results": hits, "timings_ms": watch.timings}

    def recommend(self, incident, vertical=None, platform=None, business_model=None, filters=None):
        key = INCIDENT_BY_NAME.get(incident) or (incident if incident in INCIDENTS else None)
        if key is None:
            raise KeyError("Unknown incident type: {}".format(incident))
        analysis = analyse("")
        analysis.vertical, analysis.platform, analysis.business_model = vertical, platform, business_model
        block = self._blocks([(key, 1.0)], analysis, filters)[0]
        block.pop("_rows", None)
        return block

    def assess(self, profile):
        prediction = self.risk.predict(profile)
        playbook = []
        for item in prediction["likely_incidents"][:3]:
            key = INCIDENT_BY_NAME[item["incident"]]
            block = self.recommend(item["incident"], vertical=profile.get("vertical"), platform=profile.get("platform"))
            best = block["recommended"][0]["fix"] if block["recommended"] else None
            playbook.append({"incident": item["incident"], "probability": item["probability"], "prepare": best,
                             "owner": FIXES[block["recommended"][0]["key"]]["team"] if best else None,
                             "reference_class": block["reference_class"],
                             "watch_for": INCIDENTS[key]["keywords"][:5]})
        prediction["playbook"] = playbook
        prediction["model_metrics"] = self.risk.metrics
        return prediction

    def stats(self):
        f = self.frame
        return {
            "incidents": int(len(f)), "columns": int(len(f.columns)),
            "index": {k: v for k, v in self.index.manifest.items() if k != "incident_ids"},
            "verticals": f["vertical"].value_counts().to_dict(),
            "incident_types": f["incident_type"].value_counts().to_dict(),
            "recovery": f["recovery_status"].value_counts().to_dict(),
            "platforms": f["platform"].value_counts().to_dict(),
            "total_revenue_at_risk_gbp": int(f["revenue_at_risk_gbp"].sum()),
            "total_revenue_recovered_gbp": int(f["revenue_recovered_gbp"].sum()),
            "median_detection_lag_hours": safe_float(f["detection_lag_hours"].median(), 1),
            "repeat_rate_90d": safe_float(f["repeat_incident_90d"].eq("Yes").mean(), 3),
        }

    def options(self):
        labels = self.index.labels
        return {
            "verticals": labels["vertical"], "platforms": labels["platform"],
            "business_models": labels["business_model"], "regions": labels["region"],
            "size_bands": ["Emerging", "Growth", "Scale", "Enterprise"],
            "fulfilment_models": labels["fulfilment_model"], "monitoring": ["Basic", "Standard", "Advanced"],
            "incident_types": [INCIDENTS[k]["name"] for k in INCIDENTS], "fixes": [FIXES[k]["name"] for k in FIXES],
        }
