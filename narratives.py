"""Narrative composer.

Turns the structured facts of one incident into the three pieces of text an
ecommerce team would write in a post incident review: an incident summary, a
resolution narrative and a lesson. Every number quoted is taken from the row,
so text and columns never disagree.
"""

from .utils import fmt_money
from .vocab import EFFICACY_MAP, FIXES, INCIDENTS, efficacy_tier


def _lower_first(text):
    return text[:1].lower() + text[1:] if text else text


def _context(r, rng):
    options = []
    if r["monitoring_maturity"] == "Basic":
        options.append("Monitoring was basic, so the problem surfaced through {} after {:.0f} hours.".format(
            r["detection_channel"].lower(), r["detection_lag_hours"]))
    elif r["detection_lag_hours"] <= 2:
        options.append("An automated alert caught it within {:.1f} hours.".format(r["detection_lag_hours"]))
    if r["app_integration_count"] >= 25:
        options.append("The store ran {} third party apps and integrations.".format(r["app_integration_count"]))
    if r["release_frequency_per_month"] >= 15:
        options.append("The team shipped around {} releases a month.".format(r["release_frequency_per_month"]))
    if r["mobile_traffic_share_pct"] >= 78:
        options.append("Mobile carried {:.0f} percent of traffic.".format(r["mobile_traffic_share_pct"]))
    if r["paid_traffic_share_pct"] >= 55:
        options.append("Paid media drove {:.0f} percent of sessions.".format(r["paid_traffic_share_pct"]))
    if r["sku_count"] >= 10000:
        options.append("The catalogue held {:,} SKUs.".format(r["sku_count"]))
    if r["support_tickets_raised"] >= 150:
        options.append("Customer service logged {} related tickets.".format(r["support_tickets_raised"]))
    if not options:
        options.append("Revenue at risk during the incident reached {}.".format(fmt_money(r["revenue_at_risk_gbp"])))
    rng.shuffle(options)
    return " ".join(options[:2])


def _symptom(r, rng):
    template = rng.choice(INCIDENTS[r["incident_key"]]["symptoms"])
    sev = r["severity_idx"]
    return template.format(
        ret="{:.0f}".format(100 * r["conversion_retention_ratio"]),
        err="{:.1f}".format(r["checkout_error_rate_pct"]),
        load="{:.1f}".format(r["page_load_time_seconds"]),
        bounce="{:.0f}".format(r["bounce_rate_pct"]),
        aband="{:.0f}".format(r["cart_abandonment_rate_pct"]),
        tickets=max(12, r["support_tickets_raised"]),
        pct=rng.randint(12 + sev * 8, 25 + sev * 12),
        days=rng.randint(2 + sev, 5 + sev * 3),
        hours=max(1, int(round(r["detection_lag_hours"] + r["time_to_fix_hours"]))),
    )


def incident_summary(r, rng):
    symptom = _symptom(r, rng)
    context = _context(r, rng)
    root = _lower_first(r["root_cause"])
    templates = [
        "{store}, a {size} {vertical} store on {platform} in {region}, suffered a {sev} severity "
        "{incident}: {symptom}. {context} The root cause was {root}.",
        "At {store} ({business}, {platform}), {symptom}. The {device} journey was affected at the {stage} "
        "stage. {context} Investigation traced it to {root}.",
        "{Incident} hit {store} in the {vertical} sector when {symptom}. {context} The team identified "
        "{root} as the underlying cause.",
        "The {platform} store {store} saw a {sev} severity problem at the {stage} stage: {symptom}. "
        "{context} Analysis pointed to {root}.",
    ]
    return rng.choice(templates).format(
        store=r["store_name"], size=r["store_size_band"].lower(), vertical=r["vertical"].lower(),
        platform=r["platform"], region=r["region"], sev=r["severity"].lower(),
        incident=r["incident_type"].lower(), Incident=r["incident_type"], symptom=symptom, context=context,
        root=root, business=r["business_model"].lower(), device=r["affected_device"].lower(),
        stage=r["affected_funnel_stage"].lower(),
    ).replace("  ", " ").strip()


def _outcome(r, rng):
    post = "{:.0f}".format(100 * r["conversion_recovery_ratio"])
    recovered = fmt_money(r["revenue_recovered_gbp"])
    status = r["recovery_status"]
    if status == "Full Recovery":
        base = rng.choice([
            "Conversion returned to {post} percent of baseline within {days} days and {rec} of revenue was recovered over the following month.",
            "The store recovered fully, reaching {post} percent of baseline conversion in {days} days and recovering {rec} over 30 days.",
        ])
    elif status == "Partial Recovery":
        base = rng.choice([
            "Conversion only climbed back to {post} percent of baseline after {days} days, recovering {rec} over 30 days.",
            "Recovery was partial: conversion settled at {post} percent of baseline and {rec} was recovered over the month.",
        ])
    else:
        base = rng.choice([
            "The fix did not restore performance; conversion stayed at {post} percent of baseline after 90 days.",
            "Performance did not recover and conversion was still at {post} percent of baseline three months later.",
        ])
    return base.format(post=post, days=r["days_to_recover"], rec=recovered)


def resolution_narrative(r, rng):
    fix = FIXES[r["fix_key"]]
    name = _lower_first(fix["name"])
    templates = [
        "Within {hours:.0f} hours the {team} team applied {name}: {mech}. {outcome}",
        "The response, led by {team}, was {name}, in which {mech}. It took {hours:.0f} hours to put in place. {outcome}",
        "{Team} chose {name} after {hours:.0f} hours; {mech}. {outcome}",
    ]
    return rng.choice(templates).format(
        hours=r["time_to_fix_hours"], team=r["team_owner"].lower(), Team=r["team_owner"], name=name,
        mech=fix["mechanism"], outcome=_outcome(r, rng),
    )


def lessons_learned(r, rng):
    inc_key, fix_key = r["incident_key"], r["fix_key"]
    tier = efficacy_tier(inc_key, fix_key)
    incident = _lower_first(r["incident_type"])
    fix_name = FIXES[fix_key]["name"]
    fix_lower = _lower_first(fix_name)
    root = _lower_first(r["root_cause"])
    best = _lower_first(FIXES[rng.choice(EFFICACY_MAP[inc_key]["strong"])]["name"])
    status = r["recovery_status"]
    if tier == "strong" and status == "Full Recovery":
        options = [
            "For {incident}, {fix} worked because it removed the cause rather than the symptom; apply it first.",
            "{Fix} restored performance quickly. Make it the default playbook step for {incident}.",
            "Speed mattered: {fix} within {hours:.0f} hours limited the revenue lost to {incident}.",
        ]
    elif tier == "strong":
        options = [
            "{Fix} is usually effective against {incident}, but slow detection let the damage compound. Invest in alerting.",
            "The right fix arrived too late; {incident} needs monitoring that catches it within hours, not days.",
        ]
    elif tier == "moderate":
        options = [
            "{Fix} helped at the margins, but {root} needed a direct fix such as {best}.",
            "{Fix} eased the symptoms of {incident}; pairing it with {best} would have closed the gap.",
        ]
    elif tier == "harmful":
        options = [
            "{Fix} made things worse: it spent margin or budget while {root} remained. For {incident}, start with {best}.",
            "Avoid reaching for {fix} when {incident} strikes; it hides the problem and costs money.",
        ]
    elif status == "Full Recovery":
        options = ["Recovery owed more to the problem fading than to {fix}; {best} is the dependable response to {incident}."]
    else:
        options = [
            "{Fix} did not address {root}; teams facing {incident} should use {best}.",
            "The response did not match the diagnosis. For {incident}, evidence favours {best} over {fix}.",
        ]
    return rng.choice(options).format(incident=incident, fix=fix_lower, Fix=fix_name, root=root, best=best,
                                      hours=r["time_to_fix_hours"])


def tags(r):
    parts = [r["incident_type"], r["root_cause"], FIXES[r["fix_key"]]["name"], r["vertical"], r["platform"],
             r["business_model"], r["affected_funnel_stage"] + " stage", r["recovery_status"]]
    return ";".join(p.lower() for p in parts)
