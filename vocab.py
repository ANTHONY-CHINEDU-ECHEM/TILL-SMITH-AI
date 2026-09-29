"""Domain knowledge for Tillsmith AI.

The single source of truth for the ecommerce world that the synthetic dataset
simulates and that the retrieval engine understands: store verticals, platforms,
business models, the fifteen store performance incidents, the fixes teams apply,
and the ground truth efficacy matrix linking each incident to the fixes that
genuinely recover conversion and revenue.

Outcomes in the generated data are driven by that matrix together with
monitoring maturity, speed of response and severity, so a system that ranks
fixes from evidence can be scored against a known truth.
"""

VERTICALS = {
    "Fashion and Apparel": {"weight": 16, "aov": 72, "conversion": 2.1, "mobile": 74, "skus": 2400, "promo": 1.0, "heavy": 0.0, "image_heavy": 1.0},
    "Beauty and Personal Care": {"weight": 10, "aov": 48, "conversion": 3.0, "mobile": 76, "skus": 900, "promo": 0.9, "heavy": 0.0, "image_heavy": 0.6},
    "Consumer Electronics": {"weight": 10, "aov": 285, "conversion": 1.5, "mobile": 58, "skus": 1600, "promo": 0.5, "heavy": 0.3, "image_heavy": 0.4},
    "Home and Furniture": {"weight": 9, "aov": 240, "conversion": 1.2, "mobile": 62, "skus": 3200, "promo": 0.6, "heavy": 1.0, "image_heavy": 1.0},
    "Grocery and Household": {"weight": 7, "aov": 58, "conversion": 4.2, "mobile": 68, "skus": 6500, "promo": 0.7, "heavy": 0.6, "image_heavy": 0.2},
    "Health and Supplements": {"weight": 7, "aov": 55, "conversion": 2.8, "mobile": 70, "skus": 600, "promo": 0.5, "heavy": 0.0, "image_heavy": 0.2},
    "Sports and Outdoors": {"weight": 8, "aov": 95, "conversion": 1.9, "mobile": 66, "skus": 2100, "promo": 0.6, "heavy": 0.4, "image_heavy": 0.7},
    "Toys and Games": {"weight": 6, "aov": 42, "conversion": 2.4, "mobile": 71, "skus": 1300, "promo": 0.8, "heavy": 0.2, "image_heavy": 0.5},
    "Jewellery and Accessories": {"weight": 7, "aov": 160, "conversion": 1.1, "mobile": 72, "skus": 1100, "promo": 0.5, "heavy": 0.0, "image_heavy": 0.9},
    "Pet Supplies": {"weight": 6, "aov": 52, "conversion": 3.4, "mobile": 69, "skus": 1800, "promo": 0.5, "heavy": 0.5, "image_heavy": 0.2},
    "Books and Media": {"weight": 5, "aov": 31, "conversion": 3.1, "mobile": 61, "skus": 12000, "promo": 0.4, "heavy": 0.1, "image_heavy": 0.1},
    "Automotive Parts": {"weight": 5, "aov": 130, "conversion": 1.6, "mobile": 55, "skus": 9000, "promo": 0.3, "heavy": 0.5, "image_heavy": 0.2},
}

SIZE_BANDS = {
    "Emerging": {"weight": 24, "revenue": 450_000, "maturity": (0.62, 0.33, 0.05), "releases": 3},
    "Growth": {"weight": 34, "revenue": 3_200_000, "maturity": (0.35, 0.50, 0.15), "releases": 6},
    "Scale": {"weight": 28, "revenue": 22_000_000, "maturity": (0.15, 0.50, 0.35), "releases": 12},
    "Enterprise": {"weight": 14, "revenue": 160_000_000, "maturity": (0.05, 0.40, 0.55), "releases": 20},
}

PLATFORMS = {
    "Shopify": {"Emerging": 45, "Growth": 30, "Scale": 8, "Enterprise": 2, "apps": 14, "self_hosted": 0.0, "headless": 0.0},
    "Shopify Plus": {"Emerging": 3, "Growth": 18, "Scale": 30, "Enterprise": 20, "apps": 18, "self_hosted": 0.0, "headless": 0.0},
    "WooCommerce": {"Emerging": 32, "Growth": 20, "Scale": 8, "Enterprise": 2, "apps": 16, "self_hosted": 1.0, "headless": 0.0},
    "Magento": {"Emerging": 5, "Growth": 12, "Scale": 18, "Enterprise": 15, "apps": 11, "self_hosted": 1.0, "headless": 0.0},
    "BigCommerce": {"Emerging": 10, "Growth": 12, "Scale": 10, "Enterprise": 6, "apps": 9, "self_hosted": 0.0, "headless": 0.0},
    "Salesforce Commerce Cloud": {"Emerging": 0.5, "Growth": 3, "Scale": 14, "Enterprise": 30, "apps": 7, "self_hosted": 0.0, "headless": 0.0},
    "Custom Headless": {"Emerging": 2, "Growth": 5, "Scale": 12, "Enterprise": 25, "apps": 6, "self_hosted": 0.5, "headless": 1.0},
}

BUSINESS_MODELS = {"Direct to Consumer": 40, "Omnichannel Retailer": 22, "Marketplace Seller": 16,
                   "Subscription": 12, "B2B Wholesale": 10}
FULFILMENT_MODELS = {"In House Warehouse": 34, "Third Party Logistics": 38, "Dropship": 12, "Hybrid Fulfilment": 16}
REGIONS = {"UK and Ireland": 24, "Western Europe": 17, "Nordics": 6, "North America": 24, "Asia Pacific": 10,
           "Australia and New Zealand": 7, "Middle East": 6, "Latin America": 6}
EU_LIKE = {"UK and Ireland", "Western Europe", "Nordics"}

MONITORING_LEVELS = ["Basic", "Standard", "Advanced"]
SEVERITIES = ["Low", "Medium", "High", "Critical"]
FUNNEL_STAGES = ["Acquisition", "Product Discovery", "Product Page", "Cart", "Checkout", "Payment", "Post Purchase"]
DEVICES = ["All Devices", "Mobile", "Desktop", "Mobile App"]
DETECTION_CHANNELS = ["Automated Alert", "Analytics Review", "Customer Complaint", "Finance Reconciliation",
                      "Marketing Report"]
RECOVERY_LEVELS = ["Full Recovery", "Partial Recovery", "Not Recovered"]
TEAMS = ["Engineering", "Growth Marketing", "Ecommerce Operations", "Payments", "Merchandising",
         "Customer Service", "Agency Partner"]

CODENAMES = [
    "Harbourline", "Kestrel", "Meridian", "Ashford", "Halcyon", "Lumen", "Solway", "Pennine", "Atlas",
    "Blackwater", "Cobalt", "Driftwood", "Elmstead", "Falcon", "Granite", "Heron", "Ivory", "Juniper",
    "Kingsway", "Larch", "Marlow", "Nimbus", "Oakridge", "Pioneer", "Redwing", "Saltire", "Thornbury",
    "Upland", "Vantage", "Westmoor", "Yarrow", "Zenith", "Bramble", "Cedar", "Dunmore", "Evergreen",
    "Foxglove", "Greystone", "Highfield", "Islay", "Keystone", "Longford", "Millbank", "Newhaven",
    "Orion", "Portland", "Riverside", "Sterling", "Tamar", "Wharfedale", "Arden", "Beacon", "Clyde",
    "Derwent", "Exmoor", "Fenwick", "Galloway", "Hadrian", "Kielder", "Lowther", "Moorcroft", "Northwick",
]
STORE_SUFFIX = {
    "Fashion and Apparel": ["Clothing", "Studio", "Wardrobe", "Label"],
    "Beauty and Personal Care": ["Beauty", "Skincare", "Apothecary"],
    "Consumer Electronics": ["Electronics", "Tech", "Digital"],
    "Home and Furniture": ["Home", "Interiors", "Living"],
    "Grocery and Household": ["Pantry", "Grocer", "Market"],
    "Health and Supplements": ["Nutrition", "Wellness", "Health"],
    "Sports and Outdoors": ["Outdoors", "Sports", "Trail"],
    "Toys and Games": ["Toys", "Games", "Playroom"],
    "Jewellery and Accessories": ["Jewellers", "Accessories", "Atelier"],
    "Pet Supplies": ["Pets", "Pet Co", "Paws"],
    "Books and Media": ["Books", "Media", "Reads"],
    "Automotive Parts": ["Motor Parts", "Autoparts", "Garage Supply"],
}

# Incidents. Each carries the language the dataset uses to describe it
# (symptoms), independent phrasing used only by the evaluation harness (queries)
# and the thesaurus the query understanding layer listens for (keywords).
INCIDENTS = {
    "checkout_failure": {
        "name": "Checkout failure",
        "stage": "Checkout",
        "root_causes": ["Theme update broke the checkout script", "Discount code logic error",
                        "Address validation service rejecting valid postcodes", "Session timeout during checkout"],
        "symptoms": [
            "checkout completion fell to {ret} percent of normal and the checkout error rate reached {err} percent",
            "customers could add items to the basket but {err} percent of checkout attempts ended in an error",
            "orders dropped sharply after a release while cart volumes stayed normal",
        ],
        "queries": [
            "customers can fill their basket but the place order button does nothing",
            "orders fell off a cliff right after our last deployment but traffic is normal",
            "people are getting an error on the final step when they try to buy",
        ],
        "keywords": ["checkout", "place order", "order button", "final step", "error at checkout", "cannot buy",
                     "checkout error", "checkout errors", "throwing errors", "after deployment", "after release",
                     "try to buy", "buy button", "orders fell", "orders collapsed", "deploy", "deployment",
                     "orders dropped", "basket"],
    },
    "payment_declines": {
        "name": "Payment decline spike",
        "stage": "Payment",
        "root_causes": ["Fraud rules set too aggressively", "Payment gateway partial outage",
                        "Strong customer authentication friction", "Card routing misconfiguration"],
        "symptoms": [
            "payment authorisation rates fell and {err} percent of card payments were declined",
            "declined transactions tripled over {days} days while fraud losses stayed flat",
            "genuine customers reported cards being refused, cutting conversion to {ret} percent of baseline",
        ],
        "queries": [
            "lots of genuine customers say their cards are being refused",
            "our payment success rate has collapsed even though fraud has not changed",
            "the bank keeps declining transactions for good customers",
        ],
        "keywords": ["payment", "payments", "card", "cards", "declined", "declines", "refused", "authorisation",
                     "authorization", "gateway", "fraud", "bank", "transaction", "transactions", "psp"],
    },
    "site_speed": {
        "name": "Site speed degradation",
        "stage": "Product Discovery",
        "root_causes": ["Third party script bloat", "Unoptimised product images", "Heavy theme update",
                        "CDN caching misconfiguration"],
        "symptoms": [
            "page load time rose to {load} seconds and bounce rate climbed to {bounce} percent",
            "mobile pages became visibly slower and conversion fell to {ret} percent of normal",
            "core web vitals failed across {pct} percent of product pages",
        ],
        "queries": [
            "the website has become painfully slow and people leave before pages load",
            "our pages take ages to appear on phones and bounce is way up",
            "site performance scores are terrible since we added new apps",
        ],
        "keywords": ["slow", "speed", "load time", "loading", "page load", "performance", "web vitals",
                     "takes ages", "take ages", "lag", "latency", "heavy pages", "performance scores",
                     "page speed", "pagespeed", "lighthouse", "slow to load"],
    },
    "mobile_ux": {
        "name": "Mobile experience regression",
        "stage": "Product Page",
        "root_causes": ["Layout broken on small screens", "Popup covering the add to cart button",
                        "Sticky banner hiding key controls", "Untested release on mobile browsers"],
        "symptoms": [
            "mobile conversion fell to {ret} percent of baseline while desktop held steady",
            "add to cart on mobile dropped by {pct} percent after a design change",
            "mobile shoppers could not reach the buy button on {pct} percent of product pages",
        ],
        "queries": [
            "desktop sales are fine but phone sales have collapsed",
            "on iphones the buy button seems to be hidden",
            "our mobile shoppers stopped adding things to their basket",
        ],
        "keywords": ["mobile", "phone", "phones", "iphone", "iphones", "android", "small screen", "responsive",
                     "popup", "hidden button", "tap", "app"],
    },
    "search_failure": {
        "name": "Onsite search failure",
        "stage": "Product Discovery",
        "root_causes": ["Search index out of sync with the catalogue", "Missing synonyms and spelling tolerance",
                        "Relevance rules promoting out of stock items", "Search provider contract lapsed"],
        "symptoms": [
            "{pct} percent of searches returned no results",
            "search led sessions converted at {ret} percent of their usual rate",
            "shoppers searching popular terms were shown irrelevant products",
        ],
        "queries": [
            "customers type product names and get nothing back",
            "people who use the search box are not finding what they want",
            "searches for our best sellers show random products",
        ],
        "keywords": ["search", "searches", "search box", "no results", "zero results", "not finding",
                     "find products", "relevance", "search bar", "search results", "nothing comes up",
                     "nothing back", "type in", "cannot find"],
    },
    "inventory_sync": {
        "name": "Inventory sync failure",
        "stage": "Product Page",
        "root_causes": ["ERP stock feed job failing silently", "Marketplace listing feed lag",
                        "Warehouse webhook failures", "Manual stock updates overwriting automation"],
        "symptoms": [
            "{pct} percent of in stock items showed as unavailable",
            "{tickets} orders were placed for items that were already out of stock",
            "stock levels lagged the warehouse by {days} days",
        ],
        "queries": [
            "products we have in the warehouse show as sold out on the site",
            "we keep selling things we do not have and then cancelling orders",
            "stock levels online do not match what is actually on the shelves",
        ],
        "keywords": ["stock", "inventory", "sold out", "out of stock", "overselling", "oversold", "warehouse",
                     "availability", "stock levels", "cancelling orders", "erp"],
    },
    "pricing_error": {
        "name": "Pricing or promotion error",
        "stage": "Cart",
        "root_causes": ["Promotion codes stacking unintentionally", "Wrong price feed imported",
                        "Currency conversion rules outdated", "Expired promotion still live"],
        "symptoms": [
            "margin collapsed as {pct} percent of orders used stacked discounts",
            "products were listed at prices {pct} percent below cost for {days} days",
            "international shoppers were charged inconsistent prices",
        ],
        "queries": [
            "a discount code is being combined with other offers and we are losing money on every order",
            "some products went live at the wrong price",
            "our sale prices did not switch off and margins are gone",
        ],
        "keywords": ["price", "prices", "pricing", "discount", "discounts", "promotion", "promo", "voucher",
                     "coupon", "margin", "wrong price", "stacking", "sale"],
    },
    "shipping_friction": {
        "name": "Shipping cost shock",
        "stage": "Checkout",
        "root_causes": ["Carrier rate increase passed straight to customers", "Free delivery threshold raised",
                        "Delivery estimate missing at checkout", "Surcharges for bulky items shown late"],
        "symptoms": [
            "cart abandonment rose to {aband} percent once delivery costs appeared",
            "{tickets} customers complained about unexpected delivery charges",
            "orders just under the free delivery threshold fell by {pct} percent",
        ],
        "queries": [
            "people abandon as soon as they see the delivery charge",
            "customers are angry about postage costs at the last step",
            "baskets are being left because shipping is too expensive",
        ],
        "keywords": ["shipping", "delivery", "delivery charge", "postage", "carrier", "free delivery",
                     "delivery cost", "courier", "surcharge", "abandon", "abandonment"],
    },
    "traffic_quality": {
        "name": "Paid traffic quality drop",
        "stage": "Acquisition",
        "root_causes": ["Bot traffic from a new ad network", "Broad match keywords draining budget",
                        "Campaign targeting changed without review", "Affiliate sending low intent visitors"],
        "symptoms": [
            "sessions rose {pct} percent but conversion fell to {ret} percent of normal",
            "paid visitors bounced at {bounce} percent",
            "return on ad spend halved within {days} days",
        ],
        "queries": [
            "we are getting loads more visitors from ads but nobody buys",
            "our ad spend is going up and sales are going down",
            "traffic looks great but it seems to be bots",
        ],
        "keywords": ["ads", "ad spend", "paid", "campaign", "campaigns", "bots", "bot", "visitors", "roas",
                     "google ads", "meta ads", "affiliate", "traffic"],
    },
    "tracking_break": {
        "name": "Analytics tracking break",
        "stage": "Post Purchase",
        "root_causes": ["Tag manager change removed the purchase event", "Consent banner blocking analytics",
                        "Duplicate tags inflating sessions", "Checkout domain change broke attribution"],
        "symptoms": [
            "reported conversion fell to {ret} percent while finance saw normal order volumes",
            "{pct} percent of purchases were missing from analytics",
            "marketing attribution showed a sudden collapse that orders did not confirm",
        ],
        "queries": [
            "analytics says sales dropped but the bank and order system disagree",
            "our dashboards show conversion crashing but the warehouse is as busy as ever",
            "numbers in our reports stopped matching real orders",
        ],
        "keywords": ["analytics", "tracking", "dashboard", "dashboards", "reports", "tag", "tags", "attribution",
                     "ga4", "consent", "numbers do not match", "reported"],
    },
    "product_content": {
        "name": "Product content gaps",
        "stage": "Product Page",
        "root_causes": ["Supplier data imported without enrichment", "Missing size and fit guidance",
                        "Images failing to load from the asset library", "Descriptions too thin for comparison"],
        "symptoms": [
            "{pct} percent of product pages lacked images or key specifications",
            "add to cart rates fell to {ret} percent of the category norm",
            "returns citing not as described rose by {pct} percent",
        ],
        "queries": [
            "our product pages look empty and nobody adds things to the basket",
            "shoppers keep asking questions that the product page should answer",
            "people return items saying they are not what they expected",
        ],
        "keywords": ["product page", "product pages", "description", "descriptions", "images", "photos",
                     "size guide", "specifications", "content", "not as described", "returns", "return items",
                     "not what they expected", "product information"],
    },
    "trust_deficit": {
        "name": "Trust and reviews deficit",
        "stage": "Product Page",
        "root_causes": ["Reviews widget stopped loading", "Negative review surge left unanswered",
                        "Security and returns information hidden", "New store lacking social proof"],
        "symptoms": [
            "product pages showed no reviews after a widget failure and conversion fell to {ret} percent",
            "a run of {tickets} negative reviews dominated the top products",
            "first time visitors converted at {ret} percent of the returning visitor rate",
        ],
        "queries": [
            "our reviews disappeared from the product pages and sales dropped",
            "a wave of bad reviews is scaring shoppers away",
            "new visitors do not trust us enough to buy",
        ],
        "keywords": ["reviews", "review", "trust", "ratings", "stars", "social proof", "bad reviews",
                     "trustpilot", "credibility", "scam"],
    },
    "platform_outage": {
        "name": "Platform or app outage",
        "stage": "Checkout",
        "root_causes": ["Third party app update failed", "Hosting capacity exhausted during a peak",
                        "API rate limits hit by an integration", "Expired SSL certificate"],
        "symptoms": [
            "the store was unavailable for {hours} hours during trading",
            "an app failure took down {pct} percent of pages",
            "the site returned errors under peak load for {hours} hours",
        ],
        "queries": [
            "the whole site went down during our biggest sale",
            "an app update broke the store and pages show errors",
            "our shop keeps crashing whenever traffic spikes",
        ],
        "keywords": ["site down", "store down", "went down", "outage", "site crashing", "store crashing", "offline",
                     "unavailable", "500 error", "app update", "hosting", "server", "certificate", "timed out"],
    },
    "seo_visibility": {
        "name": "Organic visibility loss",
        "stage": "Acquisition",
        "root_causes": ["Site migration without a redirect map", "Noindex tags left on after launch",
                        "Search algorithm update", "Duplicate content from faceted navigation"],
        "symptoms": [
            "organic sessions fell {pct} percent within {days} days",
            "{pct} percent of ranking product pages dropped out of search results",
            "revenue from unpaid search fell to {ret} percent of normal",
        ],
        "queries": [
            "since we moved to the new site we have vanished from google",
            "free traffic from search engines has halved",
            "our rankings collapsed after the replatform",
        ],
        "keywords": ["seo", "google", "organic", "rankings", "ranking", "search engine", "search engines",
                     "migration", "migrated", "replatform", "redirects", "indexing", "visibility", "vanished",
                     "new site"],
    },
    "crm_revenue_drop": {
        "name": "Email and CRM revenue drop",
        "stage": "Acquisition",
        "root_causes": ["Sending domain reputation damaged", "Abandoned basket flow switched off",
                        "Unengaged contacts never removed", "Integration stopped syncing new customers"],
        "symptoms": [
            "email revenue fell to {ret} percent of its usual share",
            "open rates dropped {pct} percent as messages landed in spam",
            "the abandoned basket flow sent nothing for {days} days",
        ],
        "queries": [
            "our emails are going to spam and newsletter sales have dried up",
            "we noticed the abandoned basket emails stopped sending",
            "revenue from our mailing list has collapsed",
        ],
        "keywords": ["email", "emails", "newsletter", "crm", "spam", "open rate", "mailing list", "klaviyo",
                     "abandoned basket email", "flows", "deliverability", "sms"],
    },
}

FIXES = {
    "checkout_rollback": {"name": "Rollback of the latest checkout change", "team": "Engineering",
                          "mechanism": "the release was reverted within hours and reapplied behind a feature flag",
                          "keywords": ["rollback", "revert"]},
    "checkout_simplification": {"name": "Checkout simplification with guest checkout", "team": "Engineering",
                                "mechanism": "form fields were cut, guest checkout enabled and errors made explicit",
                                "keywords": ["guest checkout", "simplify checkout"]},
    "payment_failover": {"name": "Payment gateway failover and smart routing", "team": "Payments",
                         "mechanism": "failed authorisations were retried through a second provider",
                         "keywords": ["failover", "smart routing", "second gateway"]},
    "fraud_rule_tuning": {"name": "Fraud rule recalibration", "team": "Payments",
                          "mechanism": "fraud thresholds were recalibrated against recent genuine orders",
                          "keywords": ["fraud rules", "recalibrate"]},
    "script_audit": {"name": "Third party script audit and deferral", "team": "Engineering",
                     "mechanism": "unused tags were removed and the rest deferred until after page render",
                     "keywords": ["script audit", "defer scripts"]},
    "image_cdn": {"name": "Image optimisation and CDN tuning", "team": "Engineering",
                  "mechanism": "images were compressed, served in modern formats and cached at the edge",
                  "keywords": ["image optimisation", "cdn"]},
    "mobile_qa": {"name": "Mobile regression testing and layout fix", "team": "Engineering",
                  "mechanism": "the layout was fixed and every release now runs on real device tests",
                  "keywords": ["device testing", "mobile testing"]},
    "search_tuning": {"name": "Search synonym and relevance tuning", "team": "Merchandising",
                      "mechanism": "zero result terms were mapped to synonyms and the index resynced hourly",
                      "keywords": ["synonyms", "search tuning"]},
    "inventory_resync": {"name": "Inventory reconciliation and sync monitoring", "team": "Ecommerce Operations",
                         "mechanism": "stock was reconciled and every feed now alerts when it stops updating",
                         "keywords": ["reconciliation", "sync monitoring"]},
    "price_guardrails": {"name": "Automated price and promotion guardrails", "team": "Ecommerce Operations",
                         "mechanism": "rules now block prices below cost and prevent codes from stacking",
                         "keywords": ["guardrails", "price rules"]},
    "shipping_transparency": {"name": "Shipping cost transparency and threshold redesign", "team": "Ecommerce Operations",
                              "mechanism": "delivery costs were shown on product pages and the free threshold reset",
                              "keywords": ["delivery threshold", "shipping transparency"]},
    "traffic_filtering": {"name": "Bot filtering and campaign retargeting", "team": "Growth Marketing",
                          "mechanism": "invalid traffic was excluded and budgets moved to proven audiences",
                          "keywords": ["bot filtering", "retargeting"]},
    "tag_audit": {"name": "Tracking audit and server side tagging", "team": "Growth Marketing",
                  "mechanism": "purchase events were restored and validated against the order system daily",
                  "keywords": ["tracking audit", "server side tagging"]},
    "content_enrichment": {"name": "Product content enrichment programme", "team": "Merchandising",
                           "mechanism": "top products gained images, size guidance and full specifications",
                           "keywords": ["content enrichment", "product content"]},
    "review_restoration": {"name": "Review widget restoration and review response programme", "team": "Customer Service",
                           "mechanism": "reviews were restored and every negative review answered within a day",
                           "keywords": ["review widget", "respond to reviews"]},
    "synthetic_monitoring": {"name": "Incident runbook with synthetic monitoring", "team": "Engineering",
                             "mechanism": "scripted journeys now test checkout every five minutes and page the team",
                             "keywords": ["synthetic monitoring", "runbook"]},
    "vendor_escalation": {"name": "App vendor escalation and version pinning", "team": "Agency Partner",
                          "mechanism": "the vendor fixed the fault and app versions are now pinned before peaks",
                          "keywords": ["vendor escalation", "version pinning"]},
    "redirect_repair": {"name": "Redirect map repair and technical SEO fix", "team": "Agency Partner",
                        "mechanism": "old URLs were redirected and indexing blockers removed",
                        "keywords": ["redirect map", "technical seo"]},
    "crm_repair": {"name": "CRM flow repair and list hygiene", "team": "Growth Marketing",
                   "mechanism": "flows were restored, unengaged contacts suppressed and the domain warmed up",
                   "keywords": ["list hygiene", "flow repair"]},
    "controlled_experiment": {"name": "Controlled experiment before full rollout", "team": "Growth Marketing",
                              "mechanism": "the fix was tested against a control group before release to all traffic",
                              "keywords": ["experiment", "ab test", "split test"]},
    "discount_blast": {"name": "Sitewide discount campaign", "team": "Growth Marketing",
                       "mechanism": "a sitewide discount was launched to lift sales",
                       "keywords": ["sitewide discount", "flash sale"]},
    "ad_spend_increase": {"name": "Paid media spend increase", "team": "Growth Marketing",
                          "mechanism": "paid media budgets were raised to push more traffic to the store",
                          "keywords": ["increase ad spend", "more ads"]},
}

EFFICACY_TIERS = {"strong": 0.85, "moderate": 0.55, "neutral": 0.25, "harmful": 0.08}

EFFICACY_MAP = {
    "checkout_failure": {"strong": ["checkout_rollback", "synthetic_monitoring"],
                         "moderate": ["checkout_simplification", "controlled_experiment"],
                         "harmful": ["discount_blast", "ad_spend_increase"]},
    "payment_declines": {"strong": ["payment_failover", "fraud_rule_tuning"],
                         "moderate": ["synthetic_monitoring"], "harmful": ["discount_blast"]},
    "site_speed": {"strong": ["script_audit", "image_cdn"], "moderate": ["synthetic_monitoring"],
                   "harmful": ["ad_spend_increase"]},
    "mobile_ux": {"strong": ["mobile_qa", "checkout_rollback"], "moderate": ["controlled_experiment"],
                  "harmful": ["ad_spend_increase", "discount_blast"]},
    "search_failure": {"strong": ["search_tuning"], "moderate": ["content_enrichment", "inventory_resync"],
                       "harmful": ["ad_spend_increase"]},
    "inventory_sync": {"strong": ["inventory_resync"], "moderate": ["synthetic_monitoring", "vendor_escalation"],
                       "harmful": ["discount_blast"]},
    "pricing_error": {"strong": ["price_guardrails"], "moderate": ["controlled_experiment"],
                      "harmful": ["discount_blast"]},
    "shipping_friction": {"strong": ["shipping_transparency"],
                          "moderate": ["checkout_simplification", "controlled_experiment"],
                          "harmful": ["ad_spend_increase"]},
    "traffic_quality": {"strong": ["traffic_filtering"], "moderate": ["tag_audit", "controlled_experiment"],
                        "harmful": ["ad_spend_increase", "discount_blast"]},
    "tracking_break": {"strong": ["tag_audit"], "moderate": ["synthetic_monitoring"],
                       "harmful": ["discount_blast", "ad_spend_increase"]},
    "product_content": {"strong": ["content_enrichment"], "moderate": ["review_restoration", "controlled_experiment"],
                        "harmful": ["discount_blast"]},
    "trust_deficit": {"strong": ["review_restoration"], "moderate": ["content_enrichment", "shipping_transparency"],
                      "harmful": ["discount_blast", "ad_spend_increase"]},
    "platform_outage": {"strong": ["vendor_escalation", "synthetic_monitoring"], "moderate": ["checkout_rollback"],
                        "harmful": ["ad_spend_increase"]},
    "seo_visibility": {"strong": ["redirect_repair"], "moderate": ["content_enrichment"],
                       "harmful": ["ad_spend_increase"]},
    "crm_revenue_drop": {"strong": ["crm_repair"], "moderate": ["controlled_experiment"],
                         "harmful": ["discount_blast"]},
}

INCIDENT_KEYS = list(INCIDENTS.keys())
FIX_KEYS = list(FIXES.keys())
INCIDENT_NAMES = [INCIDENTS[k]["name"] for k in INCIDENT_KEYS]
FIX_NAMES = [FIXES[k]["name"] for k in FIX_KEYS]
INCIDENT_BY_NAME = {INCIDENTS[k]["name"]: k for k in INCIDENT_KEYS}
FIX_BY_NAME = {FIXES[k]["name"]: k for k in FIX_KEYS}


def efficacy_tier(incident_key, fix_key):
    tiers = EFFICACY_MAP[incident_key]
    for tier in ("strong", "moderate", "harmful"):
        if fix_key in tiers[tier]:
            return tier
    return "neutral"


def efficacy_value(incident_key, fix_key):
    return EFFICACY_TIERS[efficacy_tier(incident_key, fix_key)]


def efficacy_matrix():
    return [[efficacy_value(i, j) for j in FIX_KEYS] for i in INCIDENT_KEYS]
