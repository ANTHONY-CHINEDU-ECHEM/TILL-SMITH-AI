"""Column documentation for the Tillsmith AI incident dataset."""

import html

COLUMN_DOCS = {
    "incident_id": ("identifier", "Unique incident identifier used for citations, INC followed by six digits."),
    "store_name": ("text", "Store codename and trading name."),
    "vertical": ("category", "Retail vertical, twelve values."),
    "business_model": ("category", "Direct to Consumer, Omnichannel Retailer, Marketplace Seller, Subscription or B2B Wholesale."),
    "platform": ("category", "Commerce platform, seven values including Custom Headless."),
    "region": ("category", "Main trading region."),
    "store_size_band": ("category", "Emerging, Growth, Scale or Enterprise."),
    "fulfilment_model": ("category", "In House Warehouse, Third Party Logistics, Dropship or Hybrid Fulfilment."),
    "annual_revenue_gbp": ("integer", "Annual online revenue in pounds sterling."),
    "monthly_sessions": ("integer", "Average monthly sessions."),
    "average_order_value_gbp": ("float", "Average order value in pounds sterling."),
    "baseline_conversion_rate_pct": ("float", "Normal session to order conversion rate before the incident."),
    "mobile_traffic_share_pct": ("float", "Share of sessions on mobile devices."),
    "paid_traffic_share_pct": ("float", "Share of sessions from paid media."),
    "sku_count": ("integer", "Number of live SKUs in the catalogue."),
    "app_integration_count": ("integer", "Third party apps, plugins and integrations installed."),
    "payment_provider_count": ("integer", "Number of payment providers connected."),
    "release_frequency_per_month": ("integer", "Code or theme releases shipped per month."),
    "monitoring_maturity": ("category", "Basic, Standard or Advanced monitoring and alerting."),
    "detection_date": ("date", "Date the incident was detected, YYYY/MM/DD."),
    "resolution_date": ("date", "Date the fix was in place, YYYY/MM/DD."),
    "detection_channel": ("category", "How the incident was noticed."),
    "detection_lag_hours": ("float", "Hours between onset and detection."),
    "incident_type": ("category", "The performance incident the case is about, fifteen values."),
    "severity": ("category", "Low, Medium, High or Critical."),
    "affected_funnel_stage": ("category", "Funnel stage where the incident hit, from Acquisition to Post Purchase."),
    "affected_device": ("category", "All Devices, Mobile, Desktop or Mobile App."),
    "root_cause": ("category", "Underlying cause identified in the review."),
    "incident_conversion_rate_pct": ("float", "Conversion rate during the incident."),
    "conversion_retention_ratio": ("float", "Incident conversion divided by baseline conversion. Below 1 means a drop."),
    "checkout_error_rate_pct": ("float", "Share of checkout attempts ending in an error or decline."),
    "page_load_time_seconds": ("float", "Median page load time during the incident."),
    "bounce_rate_pct": ("float", "Bounce rate during the incident."),
    "cart_abandonment_rate_pct": ("float", "Cart abandonment rate during the incident."),
    "support_tickets_raised": ("integer", "Customer service tickets linked to the incident."),
    "baseline_daily_revenue_gbp": ("float", "Normal daily revenue in pounds sterling."),
    "revenue_at_risk_gbp": ("integer", "Estimated revenue lost while the incident lasted."),
    "fix_strategy": ("category", "The fix the team applied, twenty two values."),
    "team_owner": ("category", "Team that owned the fix."),
    "time_to_fix_hours": ("float", "Hours from detection until the fix was in place."),
    "engineering_hours": ("float", "People hours spent on the fix."),
    "fix_cost_gbp": ("integer", "Cost of the fix including labour, discounts given away or extra media spend."),
    "recovery_status": ("category", "Full Recovery, Partial Recovery or Not Recovered."),
    "post_fix_conversion_rate_pct": ("float", "Conversion rate after the fix settled."),
    "conversion_recovery_ratio": ("float", "Post fix conversion divided by baseline conversion."),
    "revenue_recovered_gbp": ("integer", "Revenue regained over the 30 days after the fix compared with incident levels."),
    "days_to_recover": ("integer", "Days until conversion stabilised, 90 means no recovery within 90 days."),
    "return_on_fix": ("float", "Revenue recovered divided by fix cost."),
    "customer_satisfaction": ("integer", "Customer satisfaction during the period, 1 to 10."),
    "repeat_incident_90d": ("category", "Yes if the same incident recurred within 90 days."),
    "incident_summary": ("text", "Narrative description of the incident with its symptoms and context."),
    "resolution_narrative": ("text", "Narrative of the fix and its measured result."),
    "lessons_learned": ("text", "Retrospective lesson for future incidents."),
    "tags": ("text", "Semicolon separated keywords for filtering and search."),
}


def data_dictionary_markdown(frame):
    rows = []
    for col in frame.columns:
        kind, desc = COLUMN_DOCS.get(col, ("unknown", ""))
        example = str(frame[col].iloc[0])
        if len(example) > 70:
            example = example[:67] + "..."
        rows.append("<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            col, kind, html.escape(desc), html.escape(example)))
    header = ("# Data dictionary\n\n"
              "The dataset holds {:,} store performance incidents and {} columns. Every narrative is composed "
              "from the numbers in its own row, so text and structured fields never disagree.\n\n").format(
        len(frame), len(frame.columns))
    return header + ("<table>\n<tr><th>Column</th><th>Type</th><th>Description</th><th>Example</th></tr>\n"
                     + "\n".join(rows) + "\n</table>\n")
