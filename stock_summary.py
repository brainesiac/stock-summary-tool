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
from datetime import datetime
from typing import Optional

import ollama
import requests
from pydantic import BaseModel, Field

import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
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
    sector: str = ""
    industry: str = ""


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
# Benzinga API Client
# =============================================================================


class BenzingaClient:
    """HTTP client for Benzinga API."""

    def __init__(self, api_key: str, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url or config.BENZINGA_BASE_URL
        self.session = requests.Session()
        self.session.headers.update(config.HTTP_HEADERS)

    def _request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with error handling."""
        if params is None:
            params = {}
        params["token"] = self.api_key

        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=config.BENZINGA_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"API request failed for {endpoint}: {e}")
            return {}

    def get_quote(self, ticker: str) -> Optional[QuoteData]:
        """Fetch delayed quote data for a ticker."""
        data = self._request(config.ENDPOINTS["quote"], {"symbols": ticker})

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
            sector=quote.get("sector", "") or "",
            industry=quote.get("industry", "") or "",
        )

    def get_news(self, ticker: str, page_size: int = None) -> list[NewsItem]:
        """Fetch recent news articles for a ticker."""
        page_size = page_size or config.DEFAULT_NEWS_COUNT
        data = self._request(config.ENDPOINTS["news"], {"tickers": ticker, "pageSize": page_size})

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
            config.ENDPOINTS["ratings"],
            {"parameters[tickers]": ticker, "pageSize": config.DEFAULT_RATINGS_COUNT},
        )

        if not data or "ratings" not in data:
            return []

        ratings = []
        for rating in data["ratings"][: config.DEFAULT_RATINGS_COUNT]:
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
        data = self._request(config.ENDPOINTS["movers"], {"session": session})

        gainers = []
        losers = []

        if not data:
            return gainers, losers

        result = data.get("result", {})

        # Parse gainers
        if result.get("gainers"):
            for item in result["gainers"][: config.DEFAULT_MOVERS_COUNT]:
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
            for item in result["losers"][: config.DEFAULT_MOVERS_COUNT]:
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
            config.ENDPOINTS["economics"],
            {"parameters[country]": "US", "pagesize": config.DEFAULT_ECONOMICS_COUNT},
        )

        if not data or "economics" not in data:
            return []

        events = []
        for event in data["economics"][: config.DEFAULT_ECONOMICS_COUNT]:
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


def get_sector(quote: QuoteData) -> str:
    """Get sector from quote data, with keyword fallback."""
    # Use API-provided sector if available
    if quote and quote.sector:
        return quote.sector

    # Fallback to keyword matching on company name
    if quote and quote.company_name:
        name_lower = quote.company_name.lower()
        for sector, keywords in config.SECTOR_KEYWORDS.items():
            if any(kw in name_lower for kw in keywords):
                return sector

    return "General Market"


def determine_context_level(data: StockData) -> str:
    """Determine which fallback level to use based on available data."""
    has_quote = data.quote is not None and data.quote.price > 0
    has_news = len(data.news) > 0
    has_ratings = len(data.ratings) > 0

    if has_quote and (has_news or has_ratings):
        return config.CONTEXT_STOCK_SPECIFIC
    elif has_quote:
        return config.CONTEXT_SECTOR
    else:
        return config.CONTEXT_MARKET


# =============================================================================
# Prompt Building
# =============================================================================


def build_stock_specific_prompt(data: StockData) -> str:
    """Build prompt for stock-specific context level."""
    quote = data.quote
    ticker = data.ticker
    company = quote.company_name if quote else ticker

    # Format news bullets
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
        ratings_bullets = ratings_bullets.strip()
    else:
        ratings_bullets = "No recent analyst activity."

    # Format market cap line
    market_cap_line = f"Market Cap: ${quote.market_cap / 1e9:.2f}B" if quote.market_cap else ""

    return config.STOCK_SPECIFIC_PROMPT.format(
        ticker=ticker,
        company=company,
        price=quote.price,
        change_percent=quote.change_percent,
        volume=quote.volume,
        market_cap_line=market_cap_line,
        news_bullets=news_bullets,
        ratings_bullets=ratings_bullets,
    )


def build_sector_context_prompt(data: StockData) -> str:
    """Build prompt for sector context level (quote only, no news/ratings)."""
    quote = data.quote
    ticker = data.ticker
    company = quote.company_name if quote else ticker
    sector = get_sector(quote)

    # Format movers
    gainers_str = ", ".join(f"{m.symbol} ({m.change_percent:+.1f}%)" for m in data.gainers[:3])
    losers_str = ", ".join(f"{m.symbol} ({m.change_percent:+.1f}%)" for m in data.losers[:3])

    return config.SECTOR_CONTEXT_PROMPT.format(
        ticker=ticker,
        company=company,
        price=quote.price,
        change_percent=quote.change_percent,
        sector=sector,
        volume=quote.volume,
        gainers_str=gainers_str or "N/A",
        losers_str=losers_str or "N/A",
    )


def build_market_context_prompt(data: StockData) -> str:
    """Build prompt for market context level (minimal ticker data)."""
    ticker = data.ticker

    # Format movers
    gainers_str = ", ".join(f"{m.symbol} ({m.change_percent:+.1f}%)" for m in data.gainers[:3])
    losers_str = ", ".join(f"{m.symbol} ({m.change_percent:+.1f}%)" for m in data.losers[:3])

    # Format economic events
    if data.economic_events:
        events_str = ", ".join(e.event_name for e in data.economic_events[:3])
    else:
        events_str = "No major events scheduled"

    return config.MARKET_CONTEXT_PROMPT.format(
        ticker=ticker,
        gainers_str=gainers_str or "N/A",
        losers_str=losers_str or "N/A",
        events_str=events_str,
    )


# =============================================================================
# LLM Integration
# =============================================================================


def generate_summary_with_ollama(prompt: str, model: str = None) -> str:
    """Generate summary using Ollama."""
    model = model or config.OLLAMA_MODEL
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": config.LLM_TEMPERATURE,
                "num_predict": config.LLM_MAX_TOKENS,
            },
        )
        return response["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Ollama generation failed: {e}")
        return ""


def generate_fallback_summary(data: StockData, context_level: str) -> str:
    """Generate a structured summary without LLM when Ollama is unavailable."""
    if context_level == config.CONTEXT_STOCK_SPECIFIC and data.quote:
        quote = data.quote
        summary = f"{data.ticker} ({quote.company_name}) is trading at ${quote.price:.2f}, "
        summary += f"{'up' if quote.change >= 0 else 'down'} {abs(quote.change_percent):.2f}% today. "

        if data.news:
            summary += f"Recent headline: {data.news[0].title}. "
        if data.ratings:
            r = data.ratings[0]
            summary += f"Latest analyst action: {r.analyst} - {r.action}."
        return summary

    elif context_level == config.CONTEXT_SECTOR and data.quote:
        quote = data.quote
        sector = get_sector(quote)
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

    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or config.OLLAMA_MODEL
        self.api_key = api_key or config.BENZINGA_API_KEY
        if not self.api_key:
            raise ValueError(
                "Benzinga API key required. Set BENZINGA_API_KEY environment variable, "
                "add it to .env file, or pass api_key parameter."
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
        if context_level == config.CONTEXT_STOCK_SPECIFIC:
            return build_stock_specific_prompt(data)
        elif context_level == config.CONTEXT_SECTOR:
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
        default=config.OLLAMA_MODEL,
        help=f"Ollama model to use (default: {config.OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show additional details"
    )

    args = parser.parse_args()

    tool = StockSummaryTool(model=args.model)
    result = tool.get_summary(args.ticker)

    print(f"\n{'=' * 60}")
    print(f"Stock Summary: {result.ticker}")
    print(f"Context Level: {result.context_level}")
    print(f"{'=' * 60}\n")
    print(result.summary)

    if args.verbose and result.data.quote:
        q = result.data.quote
        print(f"\n{'-' * 60}")
        print(f"Quote Data: ${q.price:.2f} ({q.change_percent:+.2f}%)")
        print(f"Volume: {q.volume:,}")
        if result.data.news:
            print(f"News items: {len(result.data.news)}")
        if result.data.ratings:
            print(f"Ratings: {len(result.data.ratings)}")

    print()


if __name__ == "__main__":
    main()
