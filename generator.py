"""Synthetic store incident generator.

Generates a large, internally consistent dataset of ecommerce performance
incidents with vectorised numpy operations and a causal model:

    store profile  ==>  incident type, severity and detection speed
    incident + chosen fix + monitoring maturity + response speed  ==>  recovery
    recovery  ==>  conversion, revenue recovered, cost of fix, repeat risk

Because recovery depends on the ground truth efficacy matrix in vocab.py, the
data carries a real, learnable signal about which fixes work for which
incidents. Text fields are composed from the numbers so every narrative agrees
with its own row.
"""

import json
import random
from datetime import date

import numpy as np
import pandas as pd

from . import narratives
from .utils import EPS, sigmoid, sub
from .vocab import (BUSINESS_MODELS, CODENAMES, DETECTION_CHANNELS, DEVICES, EFFICACY_MAP, EFFICACY_TIERS,
                    EU_LIKE, FIX_KEYS, FIXES, FULFILMENT_MODELS, FUNNEL_STAGES, INCIDENT_KEYS, INCIDENTS,
                    MONITORING_LEVELS, PLATFORMS, RECOVERY_LEVELS, REGIONS, SEVERITIES, SIZE_BANDS,
                    STORE_SUFFIX, TEAMS, VERTICALS, efficacy_matrix)

COLUMNS = [
    "incident_id", "store_name", "vertical", "business_model", "platform", "region", "store_size_band",
    "fulfilment_model", "annual_revenue_gbp", "monthly_sessions", "average_order_value_gbp",
    "baseline_conversion_rate_pct", "mobile_traffic_share_pct", "paid_traffic_share_pct", "sku_count",
    "app_integration_count", "payment_provider_count", "release_frequency_per_month", "monitoring_maturity",
    "detection_date", "resolution_date", "detection_channel", "detection_lag_hours", "incident_type", "severity",
    "affected_funnel_stage", "affected_device", "root_cause", "incident_conversion_rate_pct",
    "conversion_retention_ratio", "checkout_error_rate_pct", "page_load_time_seconds", "bounce_rate_pct",
    "cart_abandonment_rate_pct", "support_tickets_raised", "baseline_daily_revenue_gbp", "revenue_at_risk_gbp",
    "fix_strategy", "team_owner", "time_to_fix_hours", "engineering_hours", "fix_cost_gbp", "recovery_status",
    "post_fix_conversion_rate_pct", "conversion_recovery_ratio", "revenue_recovered_gbp", "days_to_recover",
    "return_on_fix", "customer_satisfaction", "repeat_incident_90d", "incident_summary", "resolution_narrative",
    "lessons_learned", "tags",
]

FIX_HOURS = {
    "checkout_rollback": 3, "checkout_simplification": 90, "payment_failover": 8, "fraud_rule_tuning": 12,
    "script_audit": 30, "image_cdn": 40, "mobile_qa": 24, "search_tuning": 36, "inventory_resync": 20,
    "price_guardrails": 16, "shipping_transparency": 60, "traffic_filtering": 18, "tag_audit": 24,
    "content_enrichment": 200, "review_restoration": 48, "synthetic_monitoring": 30, "vendor_escalation": 20,
    "redirect_repair": 48, "crm_repair": 40, "controlled_experiment": 120, "discount_blast": 4,
    "ad_spend_increase": 2,
}
INCIDENT_DROP = {
    "checkout_failure": 0.22, "payment_declines": 0.15, "site_speed": 0.08, "mobile_ux": 0.14,
    "search_failure": 0.07, "inventory_sync": 0.08, "pricing_error": 0.04, "shipping_friction": 0.10,
    "traffic_quality": 0.12, "tracking_break": 0.18, "product_content": 0.06, "trust_deficit": 0.07,
    "platform_outage": 0.30, "seo_visibility": 0.05, "crm_revenue_drop": 0.05,
}
PREFERRED_CHANNEL = {
    "payment_declines": "Customer Complaint", "shipping_friction": "Customer Complaint",
    "inventory_sync": "Customer Complaint", "trust_deficit": "Customer Complaint",
    "tracking_break": "Finance Reconciliation", "pricing_error": "Finance Reconciliation",
    "traffic_quality": "Marketing Report", "seo_visibility": "Marketing Report", "crm_revenue_drop": "Marketing Report",
}
SLOW_TO_SPOT = {"tracking_break": 2.0, "seo_visibility": 3.0, "crm_revenue_drop": 2.0, "product_content": 2.5,
                "trust_deficit": 1.8, "search_failure": 1.5}


def _choice(rng, n_options, weights, size):
    p = np.asarray(weights, dtype=float)
    return rng.choice(n_options, size=size, p=p / p.sum())


def _gumbel(rng, shape):
    u = rng.uniform(EPS, 1.0, size=shape)
    return np.negative(np.log(np.negative(np.log(u))))


def _labels(values, idx):
    return np.array(values, dtype=object)[idx]


def generate(rows=16000, seed=42):
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    n = int(rows)

    vert_names = list(VERTICALS)
    vi = _choice(rng, len(vert_names), [VERTICALS[v]["weight"] for v in vert_names], n)
    vertical = _labels(vert_names, vi)
    vprop = {k: np.array([VERTICALS[v][k] for v in vert_names], dtype=float)[vi]
             for k in ("aov", "conversion", "mobile", "skus", "promo", "heavy", "image_heavy")}

    band_names = list(SIZE_BANDS)
    bi = _choice(rng, len(band_names), [SIZE_BANDS[b]["weight"] for b in band_names], n)
    band = _labels(band_names, bi)
    region = _labels(list(REGIONS), _choice(rng, len(REGIONS), list(REGIONS.values()), n))
    business = _labels(list(BUSINESS_MODELS), _choice(rng, len(BUSINESS_MODELS), list(BUSINESS_MODELS.values()), n))

    plat_names = list(PLATFORMS)
    platform = np.empty(n, dtype=object)
    maturity = np.zeros(n, dtype=int)
    fulfil = np.empty(n, dtype=object)
    fulfil_names = list(FULFILMENT_MODELS)
    for b_idx, b in enumerate(band_names):
        mask = bi == b_idx
        m = int(mask.sum())
        platform[mask] = _labels(plat_names, _choice(rng, len(plat_names), [PLATFORMS[p][b] for p in plat_names], m))
        maturity[mask] = _choice(rng, 3, SIZE_BANDS[b]["maturity"], m)
        weights = list(FULFILMENT_MODELS.values())
        if b == "Emerging":
            weights = [20, 30, 35, 15]
        fulfil[mask] = _labels(fulfil_names, _choice(rng, len(fulfil_names), weights, m))
    pprop = {k: np.array([PLATFORMS[p][k] for p in platform], dtype=float) for k in ("apps", "self_hosted", "headless")}

    band_rev = np.array([SIZE_BANDS[b]["revenue"] for b in band_names], dtype=float)[bi]
    revenue = np.round(band_rev * rng.lognormal(0.0, 0.55, n) / 1000.0) * 1000.0
    aov = np.round(vprop["aov"] * rng.lognormal(0.0, 0.25, n), 2)
    conv = np.round(np.clip(vprop["conversion"] * rng.lognormal(0.0, 0.22, n), 0.3, 9.0), 2)
    orders_month = revenue / aov / 12.0
    sessions = np.round(orders_month / (conv / 100.0)).astype(np.int64)
    mobile = np.round(np.clip(rng.normal(vprop["mobile"], 6.0), 30.0, 92.0), 1)
    paid = np.round(np.clip(rng.normal(34.0 + 10.0 * (bi == 0), 12.0, n), 5.0, 85.0), 1)
    band_sku = np.array([0.3, 0.7, 1.3, 2.5])[bi]
    skus = np.clip(np.round(vprop["skus"] * band_sku * rng.lognormal(0.0, 0.5, n)), 20, 250000).astype(int)
    band_app = np.array([0.8, 1.1, 1.3, 1.4])[bi]
    apps = (rng.poisson(pprop["apps"] * band_app) + 2).astype(int)
    payments = np.clip(1 + rng.poisson(0.4 + 0.6 * bi), 1, 6).astype(int)
    band_rel = np.array([SIZE_BANDS[b]["releases"] for b in band_names], dtype=float)[bi]
    releases = (rng.poisson(band_rel * (1.0 + 0.6 * pprop["headless"])) + 1).astype(int)

    is_eu = np.isin(region, list(EU_LIKE)).astype(float)
    omni_or_market = np.isin(business, ["Omnichannel Retailer", "Marketplace Seller"]).astype(float)
    outsourced = np.isin(fulfil, ["Dropship", "Third Party Logistics"]).astype(float)
    marketplace = (business == "Marketplace Seller").astype(float)
    subscription = (business == "Subscription").astype(float)
    beauty_health = np.isin(vertical, ["Beauty and Personal Care", "Health and Supplements"]).astype(float)
    emerging = (bi == 0).astype(float)
    log_sku = np.log(skus)
    feats = {
        "checkout_failure": 0.05 * releases + 0.5 * pprop["headless"] + 0.012 * apps,
        "payment_declines": 0.9 * (payments == 1) + 0.4 * (1.0 + np.negative(is_eu)) + 0.4 * (aov > 150),
        "site_speed": 0.03 * apps + 0.8 * vprop["image_heavy"] + 0.3 * pprop["self_hosted"],
        "mobile_ux": 0.018 * mobile + 0.03 * releases,
        "search_failure": 0.22 * log_sku,
        "inventory_sync": 0.15 * log_sku + 0.6 * omni_or_market + 0.5 * outsourced,
        "pricing_error": 1.0 * vprop["promo"] + 0.4 * omni_or_market,
        "shipping_friction": 1.0 * vprop["heavy"] + 0.5 * (aov < 60),
        "traffic_quality": 0.035 * paid,
        "tracking_break": 0.6 * is_eu + 0.03 * releases + 0.01 * apps,
        "product_content": 0.15 * log_sku + 0.6 * marketplace,
        "trust_deficit": 0.9 * emerging + 0.3 * (apps > 20),
        "platform_outage": 0.03 * apps + 0.7 * pprop["self_hosted"],
        "seo_visibility": 0.012 * np.subtract(100.0, paid) + 0.08 * log_sku,
        "crm_revenue_drop": 1.0 * subscription + 0.5 * beauty_health,
    }
    base = {"checkout_failure": 1.3, "payment_declines": 1.2, "site_speed": 0.9, "mobile_ux": 0.3,
            "search_failure": 0.3, "inventory_sync": 0.3, "pricing_error": 1.1, "shipping_friction": 1.4,
            "traffic_quality": 0.9, "tracking_break": 1.1, "product_content": 0.6, "trust_deficit": 1.6,
            "platform_outage": 1.1, "seo_visibility": 0.6, "crm_revenue_drop": 1.7}
    logits = np.stack([feats[k] + base[k] for k in INCIDENT_KEYS], axis=1)
    inc_idx = np.argmax(1.1 * logits + _gumbel(rng, logits.shape), axis=1)
    inc_key = _labels(INCIDENT_KEYS, inc_idx)

    pressure = (0.5 * np.subtract(2, maturity) + 0.02 * apps + 0.03 * releases + rng.normal(0.0, 0.7, n))
    cuts = np.quantile(pressure, [0.25, 0.62, 0.88])
    sev = np.digitize(pressure, cuts).astype(int)

    lag_mean = np.array([30.0, 8.0, 1.5])[maturity] * np.array([SLOW_TO_SPOT.get(k, 1.0) for k in inc_key])
    lag = np.round(np.clip(rng.exponential(lag_mean) * (1.0 + 0.15 * sev), 0.2, 720.0), 1)

    p_alert = np.array([0.1, 0.45, 0.85])[maturity]
    channel = np.empty(n, dtype=object)
    u_ch = rng.uniform(0.0, 1.0, n)
    for i in range(n):
        if u_ch[i] < p_alert[i] and lag[i] < 12:
            channel[i] = "Automated Alert"
        elif inc_key[i] in PREFERRED_CHANNEL and prng.random() < 0.6:
            channel[i] = PREFERRED_CHANNEL[inc_key[i]]
        else:
            channel[i] = prng.choice(DETECTION_CHANNELS[1:])

    stage = np.array([INCIDENTS[k]["stage"] for k in inc_key], dtype=object)
    swap = rng.uniform(0.0, 1.0, n) < 0.12
    stage = np.where(swap, _labels(FUNNEL_STAGES, rng.integers(0, len(FUNNEL_STAGES), n)), stage)
    device = _labels(DEVICES, _choice(rng, 4, [70, 20, 7, 3], n))
    is_mobile_inc = inc_key == "mobile_ux"
    device = np.where(is_mobile_inc, np.where(rng.uniform(0.0, 1.0, n) < 0.8, "Mobile", "Mobile App"), device)
    device = np.where(inc_key == "site_speed", np.where(rng.uniform(0.0, 1.0, n) < 0.5, "Mobile", "All Devices"), device)

    root = np.empty(n, dtype=object)
    fix = np.empty(n, dtype=object)
    p_strong = np.clip(0.22 + 0.12 * maturity + 0.04 * bi, 0.22, 0.6)
    u_pick = rng.uniform(0.0, 1.0, n)
    for i in range(n):
        key = inc_key[i]
        causes = INCIDENTS[key]["root_causes"]
        root[i] = causes[prng.randrange(len(causes))]
        tiers = EFFICACY_MAP[key]
        if u_pick[i] < p_strong[i]:
            fix[i] = prng.choice(tiers["strong"])
        elif u_pick[i] < p_strong[i] + 0.25 and tiers["moderate"]:
            fix[i] = prng.choice(tiers["moderate"])
        elif prng.random() < 0.35:
            fix[i] = prng.choice(["discount_blast", "ad_spend_increase"] + tiers["harmful"])
        else:
            fix[i] = prng.choice(FIX_KEYS)

    team = np.array([FIXES[k]["team"] for k in fix], dtype=object)
    team = np.where(rng.uniform(0.0, 1.0, n) < 0.12, _labels(TEAMS, rng.integers(0, len(TEAMS), n)), team)

    eff_table = np.array(efficacy_matrix())
    fix_idx = np.array([FIX_KEYS.index(k) for k in fix])
    efficacy = eff_table[inc_idx, fix_idx]
    weak_fix = efficacy <= EFFICACY_TIERS["neutral"] + EPS

    base_hours = np.array([FIX_HOURS[k] for k in fix], dtype=float)
    fix_hours = np.round(np.clip(base_hours * rng.lognormal(0.0, 0.5, n) * np.array([1.4, 1.0, 0.8])[maturity],
                                 0.5, 1500.0), 1)

    pos = 4.2 * efficacy + 0.45 * maturity + 0.8 + rng.normal(0.0, 0.35, n)
    neg = 3.5 + 0.004 * fix_hours + 0.35 * sev + 0.008 * lag
    p_full = sigmoid(np.subtract(pos, neg))
    p_part = np.subtract(1.0, p_full) * (0.35 + 0.3 * efficacy)
    u_rec = rng.uniform(0.0, 1.0, n)
    rec = np.where(u_rec < p_full, 0, np.where(u_rec < p_full + p_part, 1, 2)).astype(int)

    inc_drop = np.array([INCIDENT_DROP[k] for k in inc_key])
    drop = np.clip(0.05 + 0.08 * sev + inc_drop + rng.normal(0.0, 0.05, n), 0.03, 0.85)
    retention = np.round(np.subtract(1.0, drop), 3)
    inc_conv = np.round(conv * retention, 3)

    is_checkout = (inc_key == "checkout_failure").astype(float)
    is_payment = (inc_key == "payment_declines").astype(float)
    err = np.round(0.4 + rng.gamma(1.5, 0.6, n) + is_checkout * rng.uniform(8.0, 35.0, n)
                   + is_payment * rng.uniform(5.0, 20.0, n) + 0.8 * sev, 2)
    load_base = rng.lognormal(np.log(2.1 + 0.6 * pprop["self_hosted"]), 0.25)
    speed_mult = np.where(inc_key == "site_speed", rng.uniform(1.8, 3.2, n), 1.0)
    speed_mult = np.where(inc_key == "platform_outage", speed_mult * 1.5, speed_mult)
    load = np.round(np.clip(load_base * speed_mult, 0.8, 20.0), 2)
    bounce = np.round(np.clip(34.0 + 6.0 * load + 5.0 * sev + rng.normal(0.0, 5.0, n)
                              + 12.0 * (inc_key == "traffic_quality"), 18.0, 92.0), 1)
    aband_add = (14.0 * (inc_key == "shipping_friction") + 10.0 * is_checkout + 8.0 * is_payment)
    aband = np.round(np.clip(66.0 + aband_add + 2.0 * sev + rng.normal(0.0, 4.0, n), 45.0, 97.0), 1)
    ticket_heavy = np.isin(inc_key, ["payment_declines", "checkout_failure", "inventory_sync", "pricing_error",
                                     "shipping_friction", "platform_outage"]).astype(float)
    tickets = rng.poisson((revenue / 1e6) ** 0.5 * (2.0 + 3.0 * sev) * (1.0 + 3.0 * ticket_heavy)).astype(int)

    daily = np.round(revenue / 365.0, 2)
    residual_days = np.where(rec == 0, rng.uniform(1.0, 3.0, n), np.where(rec == 1, rng.uniform(5.0, 15.0, n), 30.0))
    duration = (lag + fix_hours) / 24.0 + residual_days
    real_loss = np.where(inc_key == "tracking_break", 0.25, 1.0)
    at_risk = np.round(daily * drop * duration * real_loss)

    post = np.where(rec == 0, rng.uniform(0.96, 1.06, n),
                    np.where(rec == 1, rng.uniform(0.84, 0.95, n), np.clip(retention + rng.uniform(0.02, 0.15, n), 0.3, 0.9)))
    post = np.round(post, 3)
    post_conv = np.round(conv * post, 3)
    recovered = np.round(daily * 30.0 * np.maximum(np.subtract(post, retention), 0.0) * real_loss)
    days_rec = np.where(rec == 0, np.round(1 + rng.gamma(2.0, 1.5, n) * (1 + 0.5 * sev)),
                        np.where(rec == 1, np.round(7 + rng.gamma(2.0, 4.0, n)), 90)).astype(int)

    band_eng = np.array([0.6, 0.9, 1.2, 1.5])[bi]
    eng_hours = np.round(fix_hours * rng.uniform(0.4, 1.5, n) * band_eng, 1)
    fix_cost = eng_hours * 95.0
    fix_cost = fix_cost + np.where(fix == "discount_blast", 0.12 * daily * 7.0, 0.0)
    fix_cost = fix_cost + np.where(fix == "ad_spend_increase", 0.2 * daily * 10.0, 0.0)
    fix_cost = np.round(np.maximum(fix_cost, 50.0))
    roi = np.round(recovered / fix_cost, 2)

    csat = 8.6 + rng.normal(0.0, 0.8, n)
    csat = np.subtract(csat, 0.7 * sev + np.array([0.0, 1.0, 2.3])[rec])
    csat = np.clip(np.round(csat), 1, 10).astype(int)
    p_repeat = 0.08 + np.array([0.0, 0.12, 0.25])[rec] + 0.15 * weak_fix + 0.10 * (maturity == 0)
    repeat = np.where(rng.uniform(0.0, 1.0, n) < p_repeat, "Yes", "No")

    start = date(2021, 1, 1).toordinal()
    end = date(2026, 6, 30).toordinal()
    detect = start + (rng.uniform(0.0, 1.0, n) * sub(end, start)).astype(int)
    resolve = detect + np.ceil((lag + fix_hours) / 24.0).astype(int)

    code = _labels(CODENAMES, rng.integers(0, len(CODENAMES), n))
    names = [c + " " + STORE_SUFFIX[v][prng.randrange(len(STORE_SUFFIX[v]))] for c, v in zip(code, vertical)]

    frame = pd.DataFrame({
        "incident_id": ["INC{:06d}".format(i + 1) for i in range(n)],
        "store_name": names, "vertical": vertical, "business_model": business, "platform": platform,
        "region": region, "store_size_band": band, "fulfilment_model": fulfil,
        "annual_revenue_gbp": revenue.astype(np.int64), "monthly_sessions": sessions,
        "average_order_value_gbp": aov, "baseline_conversion_rate_pct": conv,
        "mobile_traffic_share_pct": mobile, "paid_traffic_share_pct": paid, "sku_count": skus,
        "app_integration_count": apps, "payment_provider_count": payments,
        "release_frequency_per_month": releases, "monitoring_maturity": _labels(MONITORING_LEVELS, maturity),
        "detection_date": [date.fromordinal(int(d)).strftime("%Y/%m/%d") for d in detect],
        "resolution_date": [date.fromordinal(int(d)).strftime("%Y/%m/%d") for d in resolve],
        "detection_channel": channel, "detection_lag_hours": lag,
        "incident_type": [INCIDENTS[k]["name"] for k in inc_key], "severity": _labels(SEVERITIES, sev),
        "affected_funnel_stage": stage, "affected_device": device, "root_cause": root,
        "incident_conversion_rate_pct": inc_conv, "conversion_retention_ratio": retention,
        "checkout_error_rate_pct": err, "page_load_time_seconds": load, "bounce_rate_pct": bounce,
        "cart_abandonment_rate_pct": aband, "support_tickets_raised": tickets,
        "baseline_daily_revenue_gbp": daily, "revenue_at_risk_gbp": at_risk.astype(np.int64),
        "fix_strategy": [FIXES[k]["name"] for k in fix], "team_owner": team, "time_to_fix_hours": fix_hours,
        "engineering_hours": eng_hours, "fix_cost_gbp": fix_cost.astype(np.int64),
        "recovery_status": _labels(RECOVERY_LEVELS, rec), "post_fix_conversion_rate_pct": post_conv,
        "conversion_recovery_ratio": post, "revenue_recovered_gbp": recovered.astype(np.int64),
        "days_to_recover": days_rec, "return_on_fix": roi, "customer_satisfaction": csat,
        "repeat_incident_90d": repeat,
    })

    helper = pd.DataFrame({"incident_key": inc_key, "fix_key": fix, "severity_idx": sev})
    summaries, resolutions, lessons, tag_list = [], [], [], []
    for record, extra in zip(frame.to_dict("records"), helper.to_dict("records")):
        record.update(extra)
        summaries.append(narratives.incident_summary(record, prng))
        resolutions.append(narratives.resolution_narrative(record, prng))
        lessons.append(narratives.lessons_learned(record, prng))
        tag_list.append(narratives.tags(record))
    frame["incident_summary"] = summaries
    frame["resolution_narrative"] = resolutions
    frame["lessons_learned"] = lessons
    frame["tags"] = tag_list
    return frame[COLUMNS]


def ground_truth():
    return {
        "tiers": EFFICACY_TIERS,
        "incidents": {
            INCIDENTS[k]["name"]: {t: [FIXES[j]["name"] for j in EFFICACY_MAP[k][t]]
                                   for t in ("strong", "moderate", "harmful")}
            for k in INCIDENT_KEYS
        },
    }


def write_ground_truth(path):
    with open(path, "w", encoding="utf8") as fh:
        json.dump(ground_truth(), fh, indent=2)
