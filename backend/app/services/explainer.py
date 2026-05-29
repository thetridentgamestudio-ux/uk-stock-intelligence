import logging

import anthropic

from ..config import settings

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic | None:
    global _client
    if _client is None and settings.anthropic_api_key:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def generate_explanation(
    ticker: str,
    name: str,
    direction: str,
    confidence: float,
    rsi: float,
    volume_ratio: float,
    return_5d: float,
    news_headlines: list[str] | None = None,
) -> str:
    client = _get_client()
    if client is None:
        return _fallback_explanation(name, ticker, direction, confidence, rsi, volume_ratio)

    headlines_block = ""
    if news_headlines:
        headlines_block = "\nRecent headlines:\n" + "\n".join(
            f"- {h}" for h in news_headlines[:3]
        )

    prompt = f"""You are a concise UK equity analyst. Write 2-3 plain-English sentences explaining why this stock is predicted to move as shown. Reference the specific data. Never guarantee returns or imply certainty.

Stock: {name} ({ticker.replace('.L', '')})
Signal: {direction} — {confidence:.0f}% model confidence
RSI (14-day): {rsi:.1f}
5-day return: {return_5d:+.1f}%
Volume vs 20-day avg: {volume_ratio:.1f}x{headlines_block}

End with one brief sentence noting that this is model output, not financial advice."""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=180,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as exc:
        logger.error("Claude API error for %s: %s", ticker, exc)
        return _fallback_explanation(name, ticker, direction, confidence, rsi, volume_ratio)


def _fallback_explanation(
    name: str, ticker: str, direction: str, confidence: float, rsi: float, volume_ratio: float
) -> str:
    return (
        f"{name} shows {direction.lower()} momentum with {confidence:.0f}% model confidence. "
        f"RSI is at {rsi:.0f} and volume is {volume_ratio:.1f}x its 20-day average. "
        "This is model output only — not financial advice."
    )
