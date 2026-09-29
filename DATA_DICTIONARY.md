# Data dictionary

The dataset holds 16,000 store performance incidents and 54 columns. Every narrative is composed from the numbers in its own row, so text and structured fields never disagree.

<table>
<tr><th>Column</th><th>Type</th><th>Description</th><th>Example</th></tr>
<tr><td><code>incident_id</code></td><td>identifier</td><td>Unique incident identifier used for citations, INC followed by six digits.</td><td>INC000001</td></tr>
<tr><td><code>store_name</code></td><td>text</td><td>Store codename and trading name.</td><td>Driftwood Atelier</td></tr>
<tr><td><code>vertical</code></td><td>category</td><td>Retail vertical, twelve values.</td><td>Jewellery and Accessories</td></tr>
<tr><td><code>business_model</code></td><td>category</td><td>Direct to Consumer, Omnichannel Retailer, Marketplace Seller, Subscription or B2B Wholesale.</td><td>Subscription</td></tr>
<tr><td><code>platform</code></td><td>category</td><td>Commerce platform, seven values including Custom Headless.</td><td>Shopify Plus</td></tr>
<tr><td><code>region</code></td><td>category</td><td>Main trading region.</td><td>UK and Ireland</td></tr>
<tr><td><code>store_size_band</code></td><td>category</td><td>Emerging, Growth, Scale or Enterprise.</td><td>Scale</td></tr>
<tr><td><code>fulfilment_model</code></td><td>category</td><td>In House Warehouse, Third Party Logistics, Dropship or Hybrid Fulfilment.</td><td>Third Party Logistics</td></tr>
<tr><td><code>annual_revenue_gbp</code></td><td>integer</td><td>Annual online revenue in pounds sterling.</td><td>20137000</td></tr>
<tr><td><code>monthly_sessions</code></td><td>integer</td><td>Average monthly sessions.</td><td>717024</td></tr>
<tr><td><code>average_order_value_gbp</code></td><td>float</td><td>Average order value in pounds sterling.</td><td>262.96</td></tr>
<tr><td><code>baseline_conversion_rate_pct</code></td><td>float</td><td>Normal session to order conversion rate before the incident.</td><td>0.89</td></tr>
<tr><td><code>mobile_traffic_share_pct</code></td><td>float</td><td>Share of sessions on mobile devices.</td><td>59.8</td></tr>
<tr><td><code>paid_traffic_share_pct</code></td><td>float</td><td>Share of sessions from paid media.</td><td>24.3</td></tr>
<tr><td><code>sku_count</code></td><td>integer</td><td>Number of live SKUs in the catalogue.</td><td>673</td></tr>
<tr><td><code>app_integration_count</code></td><td>integer</td><td>Third party apps, plugins and integrations installed.</td><td>27</td></tr>
<tr><td><code>payment_provider_count</code></td><td>integer</td><td>Number of payment providers connected.</td><td>2</td></tr>
<tr><td><code>release_frequency_per_month</code></td><td>integer</td><td>Code or theme releases shipped per month.</td><td>11</td></tr>
<tr><td><code>monitoring_maturity</code></td><td>category</td><td>Basic, Standard or Advanced monitoring and alerting.</td><td>Standard</td></tr>
<tr><td><code>detection_date</code></td><td>date</td><td>Date the incident was detected, YYYY/MM/DD.</td><td>2021/06/15</td></tr>
<tr><td><code>resolution_date</code></td><td>date</td><td>Date the fix was in place, YYYY/MM/DD.</td><td>2021/06/19</td></tr>
<tr><td><code>detection_channel</code></td><td>category</td><td>How the incident was noticed.</td><td>Analytics Review</td></tr>
<tr><td><code>detection_lag_hours</code></td><td>float</td><td>Hours between onset and detection.</td><td>83.0</td></tr>
<tr><td><code>incident_type</code></td><td>category</td><td>The performance incident the case is about, fifteen values.</td><td>Organic visibility loss</td></tr>
<tr><td><code>severity</code></td><td>category</td><td>Low, Medium, High or Critical.</td><td>Critical</td></tr>
<tr><td><code>affected_funnel_stage</code></td><td>category</td><td>Funnel stage where the incident hit, from Acquisition to Post Purchase.</td><td>Acquisition</td></tr>
<tr><td><code>affected_device</code></td><td>category</td><td>All Devices, Mobile, Desktop or Mobile App.</td><td>Mobile</td></tr>
<tr><td><code>root_cause</code></td><td>category</td><td>Underlying cause identified in the review.</td><td>Site migration without a redirect map</td></tr>
<tr><td><code>incident_conversion_rate_pct</code></td><td>float</td><td>Conversion rate during the incident.</td><td>0.571</td></tr>
<tr><td><code>conversion_retention_ratio</code></td><td>float</td><td>Incident conversion divided by baseline conversion. Below 1 means a drop.</td><td>0.642</td></tr>
<tr><td><code>checkout_error_rate_pct</code></td><td>float</td><td>Share of checkout attempts ending in an error or decline.</td><td>3.48</td></tr>
<tr><td><code>page_load_time_seconds</code></td><td>float</td><td>Median page load time during the incident.</td><td>3.0</td></tr>
<tr><td><code>bounce_rate_pct</code></td><td>float</td><td>Bounce rate during the incident.</td><td>62.1</td></tr>
<tr><td><code>cart_abandonment_rate_pct</code></td><td>float</td><td>Cart abandonment rate during the incident.</td><td>64.2</td></tr>
<tr><td><code>support_tickets_raised</code></td><td>integer</td><td>Customer service tickets linked to the incident.</td><td>48</td></tr>
<tr><td><code>baseline_daily_revenue_gbp</code></td><td>float</td><td>Normal daily revenue in pounds sterling.</td><td>55169.86</td></tr>
<tr><td><code>revenue_at_risk_gbp</code></td><td>integer</td><td>Estimated revenue lost while the incident lasted.</td><td>665801</td></tr>
<tr><td><code>fix_strategy</code></td><td>category</td><td>The fix the team applied, twenty two values.</td><td>Paid media spend increase</td></tr>
<tr><td><code>team_owner</code></td><td>category</td><td>Team that owned the fix.</td><td>Growth Marketing</td></tr>
<tr><td><code>time_to_fix_hours</code></td><td>float</td><td>Hours from detection until the fix was in place.</td><td>6.9</td></tr>
<tr><td><code>engineering_hours</code></td><td>float</td><td>People hours spent on the fix.</td><td>8.0</td></tr>
<tr><td><code>fix_cost_gbp</code></td><td>integer</td><td>Cost of the fix including labour, discounts given away or extra media spend.</td><td>111100</td></tr>
<tr><td><code>recovery_status</code></td><td>category</td><td>Full Recovery, Partial Recovery or Not Recovered.</td><td>Not Recovered</td></tr>
<tr><td><code>post_fix_conversion_rate_pct</code></td><td>float</td><td>Conversion rate after the fix settled.</td><td>0.598</td></tr>
<tr><td><code>conversion_recovery_ratio</code></td><td>float</td><td>Post fix conversion divided by baseline conversion.</td><td>0.672</td></tr>
<tr><td><code>revenue_recovered_gbp</code></td><td>integer</td><td>Revenue regained over the 30 days after the fix compared with incident levels.</td><td>49653</td></tr>
<tr><td><code>days_to_recover</code></td><td>integer</td><td>Days until conversion stabilised, 90 means no recovery within 90 days.</td><td>90</td></tr>
<tr><td><code>return_on_fix</code></td><td>float</td><td>Revenue recovered divided by fix cost.</td><td>0.45</td></tr>
<tr><td><code>customer_satisfaction</code></td><td>integer</td><td>Customer satisfaction during the period, 1 to 10.</td><td>5</td></tr>
<tr><td><code>repeat_incident_90d</code></td><td>category</td><td>Yes if the same incident recurred within 90 days.</td><td>No</td></tr>
<tr><td><code>incident_summary</code></td><td>text</td><td>Narrative description of the incident with its symptoms and context.</td><td>The Shopify Plus store Driftwood Atelier saw a critical severity pr...</td></tr>
<tr><td><code>resolution_narrative</code></td><td>text</td><td>Narrative of the fix and its measured result.</td><td>Growth Marketing chose paid media spend increase after 7 hours; pai...</td></tr>
<tr><td><code>lessons_learned</code></td><td>text</td><td>Retrospective lesson for future incidents.</td><td>Avoid reaching for paid media spend increase when organic visibilit...</td></tr>
<tr><td><code>tags</code></td><td>text</td><td>Semicolon separated keywords for filtering and search.</td><td>organic visibility loss;site migration without a redirect map;paid ...</td></tr>
</table>
