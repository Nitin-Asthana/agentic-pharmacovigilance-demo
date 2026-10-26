# Agentic Pharmacovigilance — Case Processing Demo

An interactive Streamlit demonstration of an **agentic AI workflow for pharmacovigilance (PV) case processing**. A team of specialized AI agents works a single safety signal through a shared case state, with a human PV reviewer in control at the decision point.

>
> - Dummy **Product names**
> - **Patients, reporters, messages, and case histories** are **invented** for the demo.
> - There is **no real patient, customer, or proprietary data** used in this project.
>
> Outputs are illustrative only and require qualified pharmacovigilance review before any operational or regulatory use.

## What it shows

**Page 1 — Case Command Center:** a sequential team of agents processes one incoming safety signal:

| Agent | Type | Role |
| --- | --- | --- |
| Intake Agent | LLM | Classifies whether the message contains a PV-relevant signal |
| Extraction Agent | LLM | Structures the case (patient, product, events, seriousness) |
| History Agent | Rules | Matches similar prior cases from a synthetic history |
| Validation Agent | Rules | Checks completeness and flags information gaps |
| Triage Agent | Rules | Recommends route and priority |
| Human PV Reviewer | You | Accepts, requests info, or overrides, then closes the case |

**Page 2 — Enterprise PV Dashboard:** a synthetic portfolio view (case counts, seriousness, routes, products) so leadership can see the operational picture.

## Authentication & configuration

The app uses **Microsoft Entra ID** via `DefaultAzureCredential` — **no API keys** are stored or required. Azure OpenAI settings are read from environment variables:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT_NAME`
- `AZURE_OPENAI_API_VERSION` (defaults to `2024-08-01-preview`)

If Azure is not configured, the app automatically runs in a deterministic **Simulation Mode** with an identical UI — so it works out of the box with no cloud setup.

## Getting started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Configure Azure OpenAI for live LLM calls
cp .env.example .env        # then edit .env with your values
az login                    # authenticate via Microsoft Entra ID

# 3. Run the app
streamlit run app.py
```

Without steps 2, the app still runs fully in Simulation Mode.

## Requirements

- Python 3.9+
- See [requirements.txt](requirements.txt)

## License

Provided as-is for demonstration purposes.
