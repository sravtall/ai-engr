from llmkit import Client

MODELS = [
    "claude-sonnet-5",
    "claude-haiku-4-5-20251001",
    "gpt-5.6-sol",
    "gpt-5.6-luna",
]

FIELDS = {
    "invoice": [
        "vendor_name",
        "invoice_number",
        "invoice_date",
        "due_date",
        "total_amount",
        "currency",
    ],
    "ticket": [
        "ticket_id",
        "customer_name",
        "issue_summary",
        "priority",
        "status",
    ],
    "email": [
        "sender",
        "recipient",
        "subject",
        "date",
        "action_items",
    ],
}

DOCUMENTS = [
    # --- invoices ---
    (
        "invoice",
        """INVOICE
Northwind Office Supplies
123 Commerce St, Seattle, WA 98101

Invoice #: INV-20481
Date: 2026-06-03
Due Date: 2026-07-03

Bill To: Ashford Consulting Group

Item                Qty   Unit Price   Total
Ergonomic Chair      4      $189.00    $756.00
Standing Desk         2      $412.00    $824.00
USB-C Dock             6       $54.00   $324.00

Subtotal: $1,904.00
Tax (8.6%): $163.74
Total Due: $2,067.74 USD""",
    ),
    (
        "invoice",
        """Bright Cloud Hosting LLC
Invoice Number: BC-77921
Issued: March 14, 2026
Payment Due: March 28, 2026

Customer: Ferro Analytics Inc.

Description: Monthly hosting plan (Pro tier) - March 2026
Amount: $499.00
Overage (bandwidth): $38.20
Total: $537.20 USD

Thank you for your business.""",
    ),
    (
        "invoice",
        """** RECEIPT / INVOICE **
Vendor: Kettlebell & Co. Fitness Equipment
Invoice ID: KC-2026-0091
Date of Sale: 2026-01-22

Sold To: Riverside Wellness Center

1x Commercial Treadmill ........ 3,200.00 EUR
2x Adjustable Bench ............   540.00 EUR
Shipping ........................  150.00 EUR
------------------------------------------
Grand Total: 3,890.00 EUR
Terms: Net 15 (due 2026-02-06)""",
    ),
    (
        "invoice",
        """Sunrise Legal Partners
INVOICE

No: SLP-4432
Date: 05/11/2026

Client: Delacroix Textiles

Professional services rendered: Contract review and negotiation (8.5 hrs @ $310/hr)
Filing fees: $75.00

Amount Due: $2,710.00
Currency: USD
Please remit within 30 days.""",
    ),
    (
        "invoice",
        """TECH PARTS DIRECT
123 Industrial Way, Austin TX

Invoice #TPD-90210
Invoice Date: 2026-07-19
Payment Terms: Due on receipt

Sold to: Marlowe Robotics

SKU-4471 Servo Motor x12 ....... $1,080.00
SKU-2209 Control Board x3 ...... $645.00
Discount (loyalty 5%) .......... -$86.25

Total: $1,638.75 USD""",
    ),
    # --- support tickets ---
    (
        "ticket",
        """[Ticket #48213]
Submitted by: Dana Whitfield (dana.w@harborline.com)
Priority: High
Status: Open

Subject: Unable to export reports to CSV

Description: Every time I click "Export to CSV" on the Analytics dashboard,
the page freezes for about 30 seconds and then shows a generic error banner.
This started after yesterday's platform update. Tried on both Chrome and
Firefox, same result. This is blocking our end-of-month reporting.""",
    ),
    (
        "ticket",
        """Ticket ID: TCK-10029
Customer: Reyes Manufacturing
Priority: Medium
Current status: In Progress

Issue: Login page occasionally redirects to a 404 page after entering
correct credentials. Happens roughly 1 in 5 attempts. Support has
reproduced it intermittently and escalated to engineering.""",
    ),
    (
        "ticket",
        """=== SUPPORT REQUEST ===
ID: 2026-SP-771
From: Priya Nandakumar
Priority: Low
Status: Waiting on Customer

Summary: Requesting clarification on how volume discounts are applied to
multi-seat licenses. No bug, just needs an explanation from billing team.""",
    ),
    (
        "ticket",
        """Ticket #55678 - URGENT
Reported by: Client Success - Vantage Retail Group
Priority: Critical
Status: Escalated

Summary: Production API returning 500 errors for all POST /orders requests
since 14:02 UTC. Estimated impact: all checkout flows down. Needs
immediate engineering attention.""",
    ),
    (
        "ticket",
        """Helpdesk Ticket
Number: HD-3390
Submitted by: Marcus Ito
Priority: Medium
Status: Resolved

Issue: User couldn't reset password because the reset email never arrived.
Root cause: spam filter on customer's mail server. Provided a
whitelisting workaround. Confirmed resolved by customer.""",
    ),
    # --- emails ---
    (
        "email",
        """From: grace.liu@meridianfoods.com
To: procurement@harvestpartners.com
Date: 2026-04-02
Subject: Q3 Ingredient Order - Need Revised Quote

Hi team,

Following up on last week's call. Can you send a revised quote for the Q3
ingredient order with the updated volume (15% higher than Q2)? We'd also
like to lock in pricing before the end of the month if possible.

Please include updated delivery timelines as well.

Thanks,
Grace""",
    ),
    (
        "email",
        """From: ops@fleetlogix.io
To: warehouse-team@fleetlogix.io
Date: 2026-02-18
Subject: Action Required: Update dock schedules by Friday

Team,

Please update your dock scheduling sheets by end of day Friday to reflect
the new shift pattern starting March 1st. Also confirm you've reassigned
forklift certifications where needed.

Let me know if you hit any blockers.

- Ops""",
    ),
    (
        "email",
        """From: h.mercer@brightlanelaw.com
To: client.relations@oakridgecapital.com
Date: 2026-05-27
Subject: Draft NDA attached for review

Hello,

Attached is the draft NDA for the upcoming partnership discussion. Please
review sections 4 and 7 in particular, as we adjusted the confidentiality
term to 5 years per your request. Let us know if you'd like a call to
walk through changes.

Best,
Helena""",
    ),
    (
        "email",
        """From: noreply-alerts@cloudsentry.com
To: devops@brightgrid.ai
Date: 2026-03-09
Subject: [ALERT] Disk usage exceeded 90% on db-primary-02

This is an automated notice: disk usage on db-primary-02 has exceeded the
90% threshold. Recommended action: run log rotation and archive old
snapshots within 24 hours to avoid service degradation.""",
    ),
    (
        "email",
        """From: talia.brooks@sundialmedia.com
To: freelancers@sundialmedia.com
Date: 2026-06-30
Subject: Updated invoicing process starting July

Hi all,

Starting July, please submit invoices through the new portal instead of
email. Old submissions will not be processed after July 15th. Reach out
to accounts@sundialmedia.com if you need portal access set up.

Thanks for your patience during the transition,
Talia""",
    ),
]


def build_prompt(doc_type: str, text: str) -> str:
    fields = ", ".join(FIELDS[doc_type])
    return (
        f"Extract the following fields from this {doc_type} as a single JSON "
        f"object with exactly these keys: {fields}. "
        "If a field isn't present in the text, use null. "
        "Respond with only the JSON object, no other text.\n\n"
        f"---\n{text}\n---"
    )


def main() -> None:
    client = Client()
    total = len(DOCUMENTS) * len(MODELS)
    call_num = 0

    for doc_type, text in DOCUMENTS:
        prompt = build_prompt(doc_type, text)
        for model in MODELS:
            call_num += 1
            result = client.create(model=model, prompt=prompt, max_output_tokens=400)
            status = "ok" if result is not None else "FAILED"
            print(f"[{call_num}/{total}] {model:<28} {doc_type:<8} {status}")

    print(f"\nDone. {total} calls attempted across {len(MODELS)} models.")


if __name__ == "__main__":
    main()
