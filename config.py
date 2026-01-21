"""
Configuration settings for Stock Summary Tool.

All prompts, constants, and environment-based settings are centralized here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")


# =============================================================================
# API Configuration
# =============================================================================

BENZINGA_API_KEY = os.environ.get("BENZINGA_API_KEY", "")
BENZINGA_BASE_URL = os.environ.get("BENZINGA_BASE_URL", "https://api.benzinga.com")
BENZINGA_TIMEOUT = int(os.environ.get("BENZINGA_TIMEOUT", "10"))

# API Endpoints
ENDPOINTS = {
    "quote": "/api/v2/quoteDelayed",
    "news": "/api/v2/news",
    "ratings": "/api/v2.1/calendar/ratings",
    "movers": "/api/v1/market/movers",
    "economics": "/api/v2.1/calendar/economics",
}

# Default pagination
DEFAULT_NEWS_COUNT = 5
DEFAULT_RATINGS_COUNT = 5
DEFAULT_MOVERS_COUNT = 5
DEFAULT_ECONOMICS_COUNT = 5


# =============================================================================
# Ollama Configuration
# =============================================================================

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# LLM Generation Parameters
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "300"))


# =============================================================================
# Sector Keywords for Classification
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
        "google",
        "meta",
        "amazon",
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
        "abbvie",
        "lilly",
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
        "wells fargo",
        "citi",
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
        "conocophillips",
    ],
    "Consumer Discretionary": [
        "retail",
        "consumer",
        "restaurant",
        "apparel",
        "automotive",
        "hotel",
        "leisure",
        "walmart",
        "target",
        "costco",
        "nike",
        "tesla",
    ],
    "Consumer Staples": [
        "food",
        "beverage",
        "household",
        "tobacco",
        "grocery",
        "procter",
        "coca-cola",
        "pepsi",
        "kraft",
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
        "honeywell",
        "3m",
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
        "comcast",
    ],
    "Real Estate": [
        "reit",
        "real estate",
        "property",
        "realty",
        "housing",
        "mortgage",
    ],
    "Materials": [
        "mining",
        "chemical",
        "steel",
        "aluminum",
        "gold",
        "copper",
        "lumber",
    ],
    "Utilities": [
        "utility",
        "electric",
        "water",
        "gas utility",
        "renewable",
    ],
}


# =============================================================================
# Prompt Templates
# =============================================================================

STOCK_SPECIFIC_PROMPT = """Generate a concise stock summary for investors.

Current Date/Time: {current_datetime}

Stock: {ticker} ({company})
Price: ${price:.2f} ({change_percent:+.2f}%)
Volume: {volume:,}
{market_cap_line}

Recent News:
{news_bullets}

Analyst Activity:
{ratings_bullets}

Generate a 3-5 sentence narrative summary covering:
1. Current price action and what's driving it. Use the current date provided above.
2. Key news developments if notable
3. Analyst sentiment if available

Be factual and avoid speculation. Write in a professional financial news style."""


SECTOR_CONTEXT_PROMPT = """Generate a concise stock summary with sector context.

Current Date/Time: {current_datetime}

Stock: {ticker} ({company})
Price: ${price:.2f} ({change_percent:+.2f}%)
Sector: {sector}
Volume: {volume:,}

No recent company-specific news or analyst activity available.

Market Context:
- Top gainers today: {gainers_str}
- Top losers today: {losers_str}

Generate a 3-5 sentence summary that:
1. Describes the stock's current price movement. Use the current date provided above.
2. Contextualizes performance relative to broader market/sector trends
3. Notes the lack of recent news while remaining informative

Be factual and professional. Acknowledge limited company-specific data."""


MARKET_CONTEXT_PROMPT = """Generate a market context summary for a stock with limited data.

Current Date/Time: {current_datetime}

Stock: {ticker}
Limited quote or company data available for this ticker.

Market Overview:
- Top gainers: {gainers_str}
- Top losers: {losers_str}
- Upcoming economic events: {events_str}

Generate a 3-5 sentence summary that:
1. Acknowledges limited data for this specific ticker
2. Provides current market conditions that could affect stocks generally. Use the current date provided above.
3. Notes key economic events or market themes

Be factual and professional. Focus on providing useful market context."""


# =============================================================================
# Logging Configuration
# =============================================================================

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(levelname)s: %(message)s"


# =============================================================================
# HTTP Client Configuration
# =============================================================================

HTTP_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
}


# =============================================================================
# Context Level Constants
# =============================================================================

CONTEXT_STOCK_SPECIFIC = "stock_specific"
CONTEXT_SECTOR = "sector_context"
CONTEXT_MARKET = "market_context"
