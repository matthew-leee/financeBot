# Algo Trading Bot - Micro-Live Production Context

## Persona & Tone Requirements
- **No Financial Advice Fluff:** Skip all preachy disclaimers. Speak like a pure systems engineer.
- **Micro-Live Execution:** Assume this bot executes real capital, but in highly restricted micro-amounts.
- **Defensive Engineering:** Prioritize aggressive error handling, fail-safes, and rate-limiting over everything else.

## Tech Stack & Standards
- **Language:** Python 3.11+
- **Data & Math:** pandas, numpy, scikit-learn, xgboost
- **Exchange Protocol:** Use the official alpaca-py SDK for ALL market data and order routing (Crypto and Equities).
- **Error Handling:** Every API call must be wrapped in strict try/except blocks to handle network timeouts, HTTP 429 rate limits, and order errors cleanly.
- **Secrets Management:** ABSOLUTELY forbidden to hardcode API keys. Force the use of Windows Environment Variables (APCA_API_KEY_ID and APCA_API_SECRET_KEY).

## Hard-Coded Code Guardrails (Bot Rules)
- **Position Sizing Limit:** The code must include a hard-coded maximum dollar amount per trade that cannot be overridden by AI logic (e.g., MAX_POSITION_SIZE = 5.00).
- **Circuit Breakers:** Implement a daily loss limit function. If the bot loses more than a set microscopic threshold in a 24-hour window, execute sys.exit() and shut down.
- **Rate Limiting:** Enforce a strict delay loop (e.g., time.sleep(1)) between API calls to prevent exchange bans.

## Testing Standards
- **Pytest Framework:** Use `pytest` for all unit and integration tests.
- **Mocking External APIs:** Absolutely FORBIDDEN to hit real Alpaca live or paper endpoints during tests. Use `unittest.mock` to mock all `alpaca-py` market data fetches and order placements.
- **Deterministic Validation:** Write edge-case tests ensuring that inputs exceeding guardrails (e.g., trying to trade $10 when MAX_POSITION_SIZE = 5.00) strictly fail or clamp correctly.
- **Self-Documentation:** Every time you add a new feature, modify file structures, or change configuration variables, you must immediately update the main `README.md` to accurately reflect these changes. Keep the architecture diagram/list pristine and completely up to date.