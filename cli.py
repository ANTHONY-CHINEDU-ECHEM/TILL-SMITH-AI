"""Command line interface.

Usage: python tillsmith.py <command> [key=value ...]

Options are written as key=value pairs so the whole toolchain works without
flag syntax. Run python tillsmith.py help for the full list of commands.
"""

import json
import sys
import textwrap
import time

from .config import settings
from .utils import parse_kv, sub

HELP = """
Tillsmith AI command line

  python tillsmith.py all                  generate, build, evaluate, check style and test in one go
  python tillsmith.py generate rows=16000 seed=42
                                           create the synthetic incident dataset
  python tillsmith.py build                build the hybrid index and train the risk models
  python tillsmith.py ask "question" provider=extractive k=8 json=0
                                           diagnose a store problem and get evidence ranked fixes
  python tillsmith.py search "query" k=10 mode=hybrid
                                           find the most similar historical incidents
  python tillsmith.py recommend incident="Checkout failure" platform=Shopify
                                           evidence table for every fix to a known incident
  python tillsmith.py assess platform=WooCommerce monitoring_maturity=Basic app_integration_count=35
                                           predict incident exposure for a store profile
  python tillsmith.py evaluate queries=400 run the benchmark and write docs/EVALUATION.md
  python tillsmith.py export_sft grounded=1000
                                           write fine tuning data to data/sft/tillsmith_sft.jsonl
  python tillsmith.py serve host=0.0.0.0 port=8000
                                           start the REST API (docs at /docs)
  python tillsmith.py ui                   start the web interface
  python tillsmith.py test                 run the test suite
  python tillsmith.py check_style          confirm no hyphen or dash characters exist in the project
"""


def _engine():
    from .engine import TillsmithEngine
    return TillsmithEngine.shared()


def cmd_generate(opts, args):
    from .engine import build_all
    build_all(settings, rows=int(opts.get("rows", settings.rows)), seed=int(opts.get("seed", settings.seed)),
              regenerate=True)


def cmd_build(opts, args):
    from .engine import build_all
    build_all(settings, regenerate=opts.get("regenerate", "0") == "1")


def cmd_ask(opts, args):
    question = " ".join(args) or opts.get("q", "")
    if not question:
        print("Ask a question, for example: python tillsmith.py ask \"checkout conversion collapsed after a release\"")
        return
    filters = {k: v for k, v in opts.items() if k in ("vertical", "platform", "business_model", "region",
                                                       "store_size_band", "monitoring_maturity")}
    result = _engine().ask(question, filters=filters or None, k=int(opts.get("k", settings.context_cases)),
                           provider=opts.get("provider"))
    if opts.get("json", "0") == "1":
        print(json.dumps({k: v for k, v in result.items()}, indent=2, default=str))
        return
    print()
    print(result["answer"])
    print()
    v = result["verification"]
    print("Provider {} | grounding {} | citations verified {} | {} ms".format(
        result["provider"], v["grounding_score"], "yes" if not v["invalid_ids"] else "no",
        result["timings_ms"]["total"]))
    for note in result["notes"]:
        print("Note: " + note)


def cmd_search(opts, args):
    query = " ".join(args) or opts.get("q", "")
    out = _engine().search(query, k=int(opts.get("k", 10)), mode=opts.get("mode", "hybrid"))
    for hit in out["results"]:
        print("{}  {:.3f}  {} | {} | {} | {} | {}".format(
            hit["incident_id"], hit["score"], hit["vertical"], hit["platform"], hit["incident_type"],
            hit["fix_strategy"], hit["recovery_status"]))
        print(textwrap.indent(textwrap.fill(hit["incident_summary"], 100), "    "))
    print("{} ms".format(out["timings_ms"]["retrieve"]))


def cmd_recommend(opts, args):
    incident = opts.get("incident") or " ".join(args)
    block = _engine().recommend(incident, vertical=opts.get("vertical"), platform=opts.get("platform"),
                                business_model=opts.get("business_model"))
    print("Reference class: " + block["reference_class"])
    print("Baseline full recovery: {:.0%}".format(block["baseline_full_recovery"]))
    for s in block["all_fixes"]:
        print("  {:<62} n={:<5} full {:>4.0%}  lower bound {:>4.0%}  median recovered GBP {:,.0f}".format(
            s["fix"], s["cases"], s["full_recovery_rate"], s["confidence_lower_bound"],
            s["median_revenue_recovered_gbp"]))


def cmd_assess(opts, args):
    numeric = {"annual_revenue_gbp", "average_order_value_gbp", "baseline_conversion_rate_pct",
               "mobile_traffic_share_pct", "paid_traffic_share_pct", "sku_count", "app_integration_count",
               "payment_provider_count", "release_frequency_per_month"}
    profile = {k: (float(v) if k in numeric else v) for k, v in opts.items()}
    print(json.dumps(_engine().assess(profile), indent=2))


def cmd_evaluate(opts, args):
    from . import evaluate
    report = evaluate.run(_engine(), int(opts.get("queries", 400)))
    evaluate.write(report, settings.report_dir, settings.docs_dir)
    print("Wrote reports/evaluation.json and docs/EVALUATION.md")


def cmd_export_sft(opts, args):
    from . import sft
    path = settings.data_dir / "sft" / "tillsmith_sft.jsonl"
    cases = opts.get("cases")
    sft.export(_engine(), path, grounded=int(opts.get("grounded", 1000)),
               cases=int(cases) if cases else None)
    print("Wrote " + str(path))


def cmd_serve(opts, args):
    import uvicorn
    from .api import app
    _engine()
    uvicorn.run(app, host=opts.get("host", "127.0.0.1"), port=int(opts.get("port", 8000)),
                workers=1, log_level=opts.get("log", "info"))


def cmd_ui(opts, args):
    from pathlib import Path
    from streamlit.web import cli as stcli
    app_path = str(Path(__file__).resolve().parent / "ui_app.py")
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())


def cmd_test(opts, args):
    import unittest
    from .config import ROOT
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


def cmd_check_style(opts, args):
    from .style import scan
    problems = scan()
    if problems:
        for path, line_no, snippet in problems[:50]:
            print("{}:{}: {}".format(path, line_no, snippet))
        print("{} lines contain hyphen or dash characters".format(len(problems)))
        sys.exit(1)
    print("Clean: no hyphen or dash characters found in the project.")


def cmd_all(opts, args):
    started = time.perf_counter()
    cmd_generate(opts, args)
    cmd_evaluate(opts, args)
    cmd_check_style(opts, args)
    print("Completed in {:.1f}s".format(sub(time.perf_counter(), started)))
    cmd_test(opts, args)


COMMANDS = {
    "generate": cmd_generate, "build": cmd_build, "ask": cmd_ask, "search": cmd_search,
    "recommend": cmd_recommend, "assess": cmd_assess, "evaluate": cmd_evaluate,
    "export_sft": cmd_export_sft, "serve": cmd_serve, "ui": cmd_ui, "test": cmd_test,
    "check_style": cmd_check_style, "all": cmd_all,
}


def main(argv):
    if not argv or argv[0] in ("help", "h"):
        print(HELP)
        return
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print("Unknown command {}. Run python tillsmith.py help".format(command))
        sys.exit(2)
    opts, args = parse_kv(rest)
    COMMANDS[command](opts, args)
