#!/usr/bin/env python3
"""
Stock Summary Tool - AI-powered narrative summaries for stocks using Benzinga API and Ollama.

Features intelligent fallback logic when stock-specific data is limited:
- Stock-Specific: Full data available (quote + news/ratings)
- Sector Context: Quote only, uses market movers for sector context
- Market Context: Limited data, provides general market overview

Usage:
    python stock_summary.py AAPL
    python stock_summary.py MSFT --model llama3.2

Module usage:
    from stock_summary import StockSummaryTool
    tool = StockSummaryTool()
    result = tool.get_summary("AAPL")
    print(result.summary)
"""

import argparse
import logging
import os
from datetime import datetime
from typing import Optional

import ollama
import requests
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class QuoteData(BaseModel):
    """Stock quote data."""

    symbol: str
    company_name: str = ""
    price: float = 0.0
    change: float = 0.0
    change_percent: float = 0.0
    volume: int = 0
    market_cap: Optional[float] = None
    previous_close: float = 0.0


class NewsItem(BaseModel):
    """News article data."""

    title: str
    teaser: str = ""
    published_at: str = ""
    url: str = ""


class RatingItem(BaseModel):
    """Analyst rating data."""

    analyst: str = ""
    action: str = ""
    rating_current: str = ""
    rating_prior: str = ""
    price_target_current: Optional[float] = None
    price_target_prior: Optional[float] = None
    date: str = ""


class MoverItem(BaseModel):
    """Market mover data."""

    symbol: str
    company_name: str = ""
    change_percent: float = 0.0
    price: float = 0.0


class EconomicEvent(BaseModel):
    """Economic calendar event."""

    event_name: str
    country: str = "US"
    date: str = ""
    importance: int = 0


class StockData(BaseModel):
    """Aggregated stock data container."""

    ticker: str
    quote: Optional[QuoteData] = None
    news: list[NewsItem] = Field(default_factory=list)
    ratings: list[RatingItem] = Field(default_factory=list)
    gainers: list[MoverItem] = Field(default_factory=list)
    losers: list[MoverItem] = Field(default_factory=list)
    economic_events: list[EconomicEvent] = Field(default_factory=list)


class StockSummary(BaseModel):
    """Final output with summary text."""

    ticker: str
    context_level: str  # stock_specific, sector_context, market_context
    summary: str
    data: StockData
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# Constants
# =============================================================================

SECTOR_KEYWORDS = {
    "Technology": [
        "tech",
        "software",
        "semiconductor",
        "computer",
        "digital",
        "cloud",
        "cyber",
        "data",
        "ai",
        "intel",
        "microsoft",
        "apple",
        "nvidia",
        "amd",
        "oracle",
    ],
    "Healthcare": [
        "pharma",
        "biotech",
        "medical",
        "health",
        "therapeutics",
        "bio",
        "drug",
        "hospital",
        "pfizer",
        "merck",
        "johnson",
    ],
    "Financials": [
        "bank",
        "financial",
        "capital",
        "investment",
        "insurance",
        "mortgage",
        "credit",
        "asset",
        "goldman",
        "morgan",
        "chase",
    ],
    "Energy": [
        "oil",
        "gas",
        "energy",
        "petroleum",
        "solar",
        "wind",
        "power",
        "exxon",
        "chevron",
        "shell",
    ],
    "Consumer": [
        "retail",
        "consumer",
        "restaurant",
        "apparel",
        "food",
        "beverage",
        "walmart",
        "amazon",
        "target",
        "costco",
    ],
    "Industrial": [
        "industrial",
        "manufacturing",
        "aerospace",
        "defense",
        "machinery",
        "caterpillar",
        "boeing",
        "lockheed",
    ],
    "Communications": [
        "telecom",
        "media",
        "entertainment",
        "streaming",
        "wireless",
        "verizon",
        "at&t",
        "disney",
        "netflix",
    ],
    "Real Estate": [
        "reit",
        "real estate",
        "property",
        "realty",
        "housing",
    ],
}


# =============================================================================
# Benzinga API Client
# =============================================================================


class BenzingaClient:
    """HTTP client for Benzinga API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.benzinga.com"
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    def _request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with error handling."""
        if params is None:
            params = {}
        params["token"] = self.api_key

        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"API request failed for {endpoint}: {e}")
            return {}

    def get_quote(self, ticker: str) -> Optional[QuoteData]:
        """Fetch delayed quote data for a ticker."""
        data = self._request("/api/v2/quoteDelayed", {"symbols": ticker})

        if not data:
            return None

        # API returns data keyed by symbol (e.g., {"AAPL": {...}})
        quote = data.get(ticker.upper())
        if not quote:
            return None

        # Try multiple possible price fields
        price = (
            quote.get("lastTradePrice")
            or quote.get("close")
            or quote.get("last")
            or quote.get("bidPrice")
            or 0
        )
        previous_close = quote.get("previousClosePrice") or quote.get("previousClose") or 0

        return QuoteData(
            symbol=quote.get("symbol", ticker),
            company_name=quote.get("name", "") or quote.get("companyStandardName", ""),
            price=float(price),
            change=float(quote.get("change", 0) or 0),
            change_percent=float(quote.get("changePercent", 0) or 0),
            volume=int(quote.get("volume", 0) or 0),
            market_cap=float(quote.get("marketCap", 0) or 0) if quote.get("marketCap") else None,
            previous_close=float(previous_close),
        )

    def get_news(self, ticker: str, page_size: int = 5) -> list[NewsItem]:
        """Fetch recent news articles for a ticker."""
        data = self._request("/api/v2/news", {"tickers": ticker, "pageSize": page_size})

        if not data:
            return []

        news_items = []
        for article in data[:page_size]:
            news_items.append(
                NewsItem(
                    title=article.get("title", ""),
                    teaser=article.get("teaser", ""),
                    published_at=article.get("created", ""),
                    url=article.get("url", ""),
                )
            )
        return news_items

    def get_ratings(self, ticker: str) -> list[RatingItem]:
        """Fetch analyst ratings for a ticker."""
        data = self._request(
            "/api/v2.1/calendar/ratings", {"parameters[tickers]": ticker, "pageSize": 5}
        )

        if not data or "ratings" not in data:
            return []

        ratings = []
        for rating in data["ratings"][:5]:
            ratings.append(
                RatingItem(
                    analyst=rating.get("analyst", "") or rating.get("analyst_name", ""),
                    action=rating.get("action_company", ""),
                    rating_current=rating.get("rating_current", ""),
                    rating_prior=rating.get("rating_prior", ""),
                    price_target_current=float(rating.get("pt_current", 0) or 0)
                    if rating.get("pt_current")
                    else None,
                    price_target_prior=float(rating.get("pt_prior", 0) or 0)
                    if rating.get("pt_prior")
                    else None,
                    date=rating.get("date", ""),
                )
            )
        return ratings

    def get_movers(self, session: str = "REGULAR") -> tuple[list[MoverItem], list[MoverItem]]:
        """Fetch market gainers and losers."""
        data = self._request("/api/v1/market/movers", {"session": session})

        gainers = []
        losers = []

        if not data:
            return gainers, losers

        result = data.get("result", {})

        # Parse gainers
        if result.get("gainers"):
            for item in result["gainers"][:5]:
                gainers.append(
                    MoverItem(
                        symbol=item.get("symbol", ""),
                        company_name=item.get("companyName", ""),
                        change_percent=float(item.get("changePercent", 0) or 0),
                        price=float(item.get("price", 0) or 0),
                    )
                )

        # Parse losers
        if result.get("losers"):
            for item in result["losers"][:5]:
                losers.append(
                    MoverItem(
                        symbol=item.get("symbol", ""),
                        company_name=item.get("companyName", ""),
                        change_percent=float(item.get("changePercent", 0) or 0),
                        price=float(item.get("price", 0) or 0),
                    )
                )

        return gainers, losers

    def get_economics(self) -> list[EconomicEvent]:
        """Fetch upcoming economic calendar events."""
        data = self._request(
            "/api/v2.1/calendar/economics", {"parameters[country]": "US", "pagesize": 5}
        )

        if not data or "economics" not in data:
            return []

        events = []
        for event in data["economics"][:5]:
            events.append(
                EconomicEvent(
                    event_name=event.get("event_name", "") or "",
                    country=event.get("country", "US") or "US",
                    date=event.get("date", "") or "",
                    importance=int(event.get("importance", 0) or 0),
                )
            )
        return events


# =============================================================================
# Context Detection and Sector Derivation
# =============================================================================


def derive_sector(company_name: str) -> str:
    """Derive sector from company name using keyword matching."""
    if not company_name:
        return "General Market"

    name_lower = company_name.lower()
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return sector
    return "General Market"


def determine_context_level(data: StockData) -> str:
    """Determine which fallback level to use based on available data."""
    has_quote = data.quote is not None and data.quote.price > 0
    has_news = len(data.news) > 0
    has_ratings = len(data.ratings) > 0

    if has_quote and (has_news or has_ratings):
        return "stock_specific"
    elif has_quote:
        return "sector_context"
    else:
        return "market_context"


# =============================================================================
# Prompt Templates
# =============================================================================


def build_stock_specific_prompt(data: StockData) -> str:
    """Build prompt for stock-specific context level."""
    quote = data.quote
    ticker = data.ticker
    company = quote.company_name if quote else ticker

    # Format news bullets
    news_bullets = ""
    if data.news:
        news_bullets = "\n".join(f"- {n.title}" for n in data.news[:5])
    else:
        news_bullets = "No recent news available."

    # Format ratings bullets
    ratings_bullets = ""
    if data.ratings:
        for r in data.ratings[:3]:
            pt_str = f" (PT: ${r.price_target_current:.2f})" if r.price_target_current else ""
            ratings_bullets += f"- {r.analyst}: {r.action} - {r.rating_current}{pt_str}\n"
    else:
        ratings_bullets = "No recent analyst activity."

    return f"""Generate a concise stock summary for investors.

Stock: {ticker} ({company})
Price: ${quote.price:.2f} ({quote.change_percent:+.2f}%)
Volume: {quote.volume:,}
{f'Market Cap: ${quote.market_cap/1e9:.2f}B' if quote.market_cap else ''}

Recent News:
{news_bullets}

Analyst Activity:
{ratings_bullets.strip()}

Generate a 3-5 sentence narrative summary covering:
1. Current price action and what's driving it
2. Key news developments if notable
3. Analyst sentiment if available

Be factual and avoid speculation. Write in a professional financial news style."""


def build_sector_context_prompt(data: StockData) -> str:
    """Build prompt for sector context level (quote only, no news/ratings)."""
    quote = data.quote
    ticker = data.ticker
    company = quote.company_name if quote else ticker
    sector = derive_sector(company)

    # Format movers
    gainers_str = ", ".join(f"{m.symbol} ({m.change_percent:+.1f}%)" for m in data.gainers[:3])
    losers_str = ", ".join(f"{m.symbol} ({m.change_percent:+.1f}%)" for m in data.losers[:3])

    return f"""Generate a concise stock summary with sector context.

Stock: {ticker} ({company})
Price: ${quote.price:.2f} ({quote.change_percent:+.2f}%)
Sector: {sector}
Volume: {quote.volume:,}

No recent company-specific news or analyst activity available.

Market Context:
- Top gainers today: {gainers_str or 'N/A'}
- Top losers today: {losers_str or 'N/A'}

Generate a 3-5 sentence summary that:
1. Describes the stock's current price movement
2. Contextualizes performance relative to broader market/sector trends
3. Notes the lack of recent news while remaining informative

Be factual and professional. Acknowledge limited company-specific data."""


def build_market_context_prompt(data: StockData) -> str:
    """Build prompt for market context level (minimal ticker data)."""
    ticker = data.ticker

    # Format movers
    gainers_str = ", ".join(f"{m.symbol} ({m.change_percent:+.1f}%)" for m in data.gainers[:3])
    losers_str = ", ".join(f"{m.symbol} ({m.change_percent:+.1f}%)" for m in data.losers[:3])

    # Format economic events
    events_str = ""
    if data.economic_events:
        events_str = ", ".join(e.event_name for e in data.economic_events[:3])
    else:
        events_str = "No major events scheduled"

    return f"""Generate a market context summary for a stock with limited data.

Stock: {ticker}
Limited quote or company data available for this ticker.

Market Overview:
- Top gainers: {gainers_str or 'N/A'}
- Top losers: {losers_str or 'N/A'}
- Upcoming economic events: {events_str}

Generate a 3-5 sentence summary that:
1. Acknowledges limited data for this specific ticker
2. Provides current market conditions that could affect stocks generally
3. Notes key economic events or market themes

Be factual and professional. Focus on providing useful market context."""


# =============================================================================
# LLM Integration
# =============================================================================


def generate_summary_with_ollama(prompt: str, model: str = "qwen2.5-coder:7b") -> str:
    """Generate summary using Ollama."""
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 300},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Ollama generation failed: {e}")
        return ""


def generate_fallback_summary(data: StockData, context_level: str) -> str:
    """Generate a structured summary without LLM when Ollama is unavailable."""
    if context_level == "stock_specific" and data.quote:
        quote = data.quote
        summary = f"{data.ticker} ({quote.company_name}) is trading at ${quote.price:.2f}, "
        summary += f"{'up' if quote.change >= 0 else 'down'} {abs(quote.change_percent):.2f}% today. "

        if data.news:
            summary += f"Recent headline: {data.news[0].title}. "
        if data.ratings:
            r = data.ratings[0]
            summary += f"Latest analyst action: {r.analyst} - {r.action}."
        return summary

    elif context_level == "sector_context" and data.quote:
        quote = data.quote
        sector = derive_sector(quote.company_name)
        summary = f"{data.ticker} ({quote.company_name}) in the {sector} sector "
        summary += f"is trading at ${quote.price:.2f} ({quote.change_percent:+.2f}%). "
        summary += "No recent company-specific news available."
        return summary

    else:
        summary = f"Limited data available for {data.ticker}. "
        if data.gainers:
            summary += f"Market gainers include {data.gainers[0].symbol}. "
        if data.losers:
            summary += f"Market losers include {data.losers[0].symbol}."
        return summary


# =============================================================================
# Main Tool Class
# =============================================================================


class StockSummaryTool:
    """Main interface for generating stock summaries."""

    def __init__(self, model: str = "qwen2.5-coder:7b", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.environ.get("BENZINGA_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "Benzinga API key required. Set BENZINGA_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = BenzingaClient(self.api_key)

    def _fetch_data(self, ticker: str) -> StockData:
        """Fetch all available data for a ticker."""
        ticker = ticker.upper().strip()

        # Always fetch quote first
        quote = self.client.get_quote(ticker)

        # Fetch news and ratings
        news = self.client.get_news(ticker)
        ratings = self.client.get_ratings(ticker)

        # Fetch market context for fallback
        gainers, losers = self.client.get_movers()
        economics = self.client.get_economics()

        return StockData(
            ticker=ticker,
            quote=quote,
            news=news,
            ratings=ratings,
            gainers=gainers,
            losers=losers,
            economic_events=economics,
        )

    def _build_prompt(self, data: StockData, context_level: str) -> str:
        """Build appropriate prompt based on context level."""
        if context_level == "stock_specific":
            return build_stock_specific_prompt(data)
        elif context_level == "sector_context":
            return build_sector_context_prompt(data)
        else:
            return build_market_context_prompt(data)

    def _generate(self, prompt: str) -> str:
        """Generate summary using LLM."""
        return generate_summary_with_ollama(prompt, self.model)

    def get_summary(self, ticker: str) -> StockSummary:
        """Generate stock summary with automatic fallback."""
        logger.info(f"Fetching data for {ticker}...")

        # Fetch all data
        data = self._fetch_data(ticker)

        # Determine context level
        context_level = determine_context_level(data)
        logger.info(f"Context level: {context_level}")

        # Build prompt
        prompt = self._build_prompt(data, context_level)

        # Generate summary
        summary_text = self._generate(prompt)

        # Fallback if LLM failed
        if not summary_text:
            logger.info("Using fallback summary (LLM unavailable)")
            summary_text = generate_fallback_summary(data, context_level)

        return StockSummary(
            ticker=ticker.upper(),
            context_level=context_level,
            summary=summary_text,
            data=data,
        )


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate AI-powered stock summaries using Benzinga data and Ollama."
    )
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL, MSFT)")
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:7b",
        help="Ollama model to use (default: qwen2.5-coder:7b)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show additional details"
    )

    args = parser.parse_args()

    tool = StockSummaryTool(model=args.model)
    result = tool.get_summary(args.ticker)

    print(f"\n{'='*60}")
    print(f"Stock Summary: {result.ticker}")
    print(f"Context Level: {result.context_level}")
    print(f"{'='*60}\n")
    print(result.summary)

    if args.verbose and result.data.quote:
        q = result.data.quote
        print(f"\n{'─'*60}")
        print(f"Quote Data: ${q.price:.2f} ({q.change_percent:+.2f}%)")
        print(f"Volume: {q.volume:,}")
        if result.data.news:
            print(f"News items: {len(result.data.news)}")
        if result.data.ratings:
            print(f"Ratings: {len(result.data.ratings)}")

    print()


if __name__ == "__main__":
    main()
