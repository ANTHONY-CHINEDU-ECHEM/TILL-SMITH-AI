"""Answer composition.

From the same structured evidence two outputs are produced: an evidence pack,
the only source of truth a language model receives, and an extractive answer
written deterministically with no model at all. The extractive answer is the
default provider and the fallback whenever a model answer fails verification.
"""

from .utils import clean_dashes, fmt_money, pct
from .vocab import FIXES

SYSTEM_PROMPT = """You are Tillsmith AI, a senior ecommerce performance advisor.
You answer questions about store incidents such as conversion drops, checkout failures and revenue loss using ONLY the evidence pack you are given.

Rules:
1. Base every recommendation on the evidence pack. Do not invent statistics.
2. Cite precedent incidents by their ID in square brackets, for example [INC000123]. Only cite IDs that appear in the evidence pack.
3. Quote numbers exactly as they appear in the evidence pack.
4. If the evidence is thin or the question is not about store performance, say so plainly.
5. Use these headings: Diagnosis, Recommended fixes, Avoid, Move fast, What to expect, Confidence.
6. Write in British English, be concise and practical, and address the reader as the person running the store.
7. Never use hyphens or dashes of any kind. Write compound words as separate words."""


def _case_line(c):
    return ("[{id}] {store} | {vertical} | {platform} | incident: {inc} | fix: {fix} | result: {rec}, conversion "
            "{post} of baseline, {money} recovered over 30 days").format(
        id=c["incident_id"], store=c["store_name"], vertical=c["vertical"], platform=c["platform"],
        inc=c["incident_type"], fix=c["fix_strategy"], rec=c["recovery_status"],
        post=pct(c["conversion_recovery_ratio"]), money=fmt_money(c["revenue_recovered_gbp"]))


def evidence_pack(question, result):
    lines = ["QUESTION", question, "", "DIAGNOSIS"]
    for d in result["diagnosis"]:
        lines.append("{} (share of evidence {})".format(d["incident"], pct(d["share"])))
    for block in result["recommendations"]:
        lines.append("")
        lines.append("REFERENCE CLASS FOR {}: {}".format(block["incident"].upper(), block["reference_class"]))
        lines.append("Baseline full recovery rate: {}".format(pct(block["baseline_full_recovery"])))
        for s in block["recommended"]:
            lines.append("RECOMMENDED {}: full recovery {} of {} incidents, lower bound {}, median conversion "
                         "back to {} of baseline in {} days, median revenue recovered {}, median fix cost {}, "
                         "repeat within 90 days {}".format(
                             s["fix"], pct(s["full_recovery_rate"]), s["cases"], pct(s["confidence_lower_bound"]),
                             pct(s["median_conversion_recovery"]), int(s["median_days_to_recover"]),
                             fmt_money(s["median_revenue_recovered_gbp"]), fmt_money(s["median_fix_cost_gbp"]),
                             pct(s["repeat_rate_90d"])))
        for s in block["avoid"]:
            lines.append("AVOID {}: full recovery only {} of {} incidents".format(
                s["fix"], pct(s["full_recovery_rate"]), s["cases"]))
        t = block.get("timing")
        if t:
            lines.append("TIMING: resolved within {} hours of onset gave full recovery {} versus {} when slower".format(
                int(t["threshold_hours"]), pct(t["fast_full_recovery_rate"]), pct(t["slow_full_recovery_rate"])))
        o = block.get("outlook") or {}
        if o:
            lines.append("OUTLOOK: conversion typically fell to {} of baseline, median cost {:.1f} days of revenue, "
                         "P80 {:.1f} days, median detection {:.0f} hours, repeat within 90 days {}".format(
                             pct(o["median_conversion_retention"]), o["median_days_of_revenue_lost"],
                             o["p80_days_of_revenue_lost"], o["median_detection_lag_hours"], pct(o["repeat_rate_90d"])))
    lines.append("")
    lines.append("PRECEDENT INCIDENTS")
    for c in result["cases"]:
        lines.append(_case_line(c))
        lines.append("  Incident: " + c["incident_summary"])
        lines.append("  Resolution: " + c["resolution_narrative"])
        lines.append("  Lesson: " + c["lessons_learned"])
    conf = result["confidence"]
    lines.append("")
    lines.append("CONFIDENCE: {} ({}). {}".format(conf["label"], conf["score"], " ".join(conf["reasons"])))
    return "\n".join(lines)


def _precedent(result, fix_name):
    for c in result["cases"]:
        if c["fix_strategy"] == fix_name and c["recovery_status"] == "Full Recovery":
            return c["incident_id"]
    return None


def extractive_answer(result):
    diag = result["diagnosis"]
    if not diag:
        return ("### Diagnosis\nThe question does not describe a recognisable store performance incident, so there is "
                "no reliable evidence to draw on. Describe what you are seeing, for example falling conversion, "
                "checkout errors, declined payments or a traffic drop, and ask again.")
    out = []
    cases = result["cases"]
    closest = ", ".join("[{}]".format(c["incident_id"]) for c in cases[:2])
    text = "Your situation most closely matches **{}** ({} of the weighted evidence)".format(
        diag[0]["incident"].lower(), pct(diag[0]["share"]))
    if len(diag) > 1:
        text += ", with **{}** as a secondary pattern ({})".format(diag[1]["incident"].lower(), pct(diag[1]["share"]))
    text += ". The closest precedents are {}.".format(closest) if closest else "."
    out.append("### Diagnosis\n" + text)

    for n, block in enumerate(result["recommendations"]):
        heading = "### Recommended fixes" if n == 0 else "### Also address {}".format(block["incident"].lower())
        items = []
        for i, s in enumerate(block["recommended"], 1):
            line = ("{}. **{}.** Full recovery in {} of {} comparable incidents (statistical lower bound {}), against "
                    "a {} baseline. Conversion typically returned to {} of baseline in {} days, recovering {} over "
                    "30 days for a median fix cost of {}. How it works: {}. Usual owner: {}.").format(
                i, s["fix"], pct(s["full_recovery_rate"]), s["cases"], pct(s["confidence_lower_bound"]),
                pct(block["baseline_full_recovery"]), pct(s["median_conversion_recovery"]),
                int(s["median_days_to_recover"]), fmt_money(s["median_revenue_recovered_gbp"]),
                fmt_money(s["median_fix_cost_gbp"]), FIXES[s["key"]]["mechanism"], FIXES[s["key"]]["team"].lower())
            precedent = _precedent(result, s["fix"])
            if precedent:
                line += " Precedent: [{}].".format(precedent)
            items.append(line)
        if not items:
            items.append("No single fix stands out in this reference class; the evidence is too even to rank.")
        out.append(heading + "\n" + "\n".join(items))
        if block["avoid"]:
            out.append("### Avoid\n" + " ".join("**{}** recovered fully in only {} of {} comparable incidents.".format(
                s["fix"], pct(s["full_recovery_rate"]), s["cases"]) for s in block["avoid"]))
        t = block.get("timing")
        if t and n == 0 and t["fast_full_recovery_rate"] > t["slow_full_recovery_rate"]:
            out.append("### Move fast\nAmong the strongest fixes, incidents resolved within {} hours of onset fully "
                       "recovered in {} of cases, compared with {} when resolution took longer.".format(
                           int(t["threshold_hours"]), pct(t["fast_full_recovery_rate"]),
                           pct(t["slow_full_recovery_rate"])))
        o = block.get("outlook") or {}
        if o and n == 0:
            out.append("### What to expect\nIn this reference class ({}), conversion typically fell to {} of baseline "
                       "and the median incident cost {:.1f} days of revenue, rising to {:.1f} days at P80. Median "
                       "detection took {:.0f} hours, and {} of incidents recurred within 90 days.".format(
                           block["reference_class"], pct(o["median_conversion_retention"]),
                           o["median_days_of_revenue_lost"], o["p80_days_of_revenue_lost"],
                           o["median_detection_lag_hours"], pct(o["repeat_rate_90d"])))
    lessons = ["* [{}] {}".format(c["incident_id"], c["lessons_learned"]) for c in cases[:3]]
    if lessons:
        out.append("### Lessons from the closest precedents\n" + "\n".join(lessons))
    conf = result["confidence"]
    out.append("### Confidence\n**{}** ({:.2f}). {}".format(conf["label"], conf["score"], " ".join(conf["reasons"])))
    return clean_dashes("\n\n".join(out))
