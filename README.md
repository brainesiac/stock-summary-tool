# Stock Summary Tool

AI-powered narrative summaries for stocks using Benzinga API data and local Ollama LLMs.

## Features

- **Real-time stock data** via Benzinga API (quotes, news, analyst ratings)
- **AI-generated summaries** using local Ollama models
- **Intelligent fallback logic** when stock-specific data is limited:
  - **Stock-Specific**: Full analysis with quote, news, and analyst ratings
  - **Sector Context**: Quote data with market mover context
  - **Market Context**: General market overview for unknown tickers
- **Type-safe data models** using Pydantic
- **CLI and module interfaces**

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally
- Benzinga API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-summary-tool.git
cd stock-summary-tool
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure Ollama is running with a compatible model:
```bash
ollama pull qwen2.5-coder:7b
ollama serve
```

## Configuration

The tool uses a Benzinga API key. You can configure it in two ways:

1. **Environment variable** (recommended):
```bash
export BENZINGA_API_KEY="your-api-key"
```

2. **Pass directly** when instantiating the tool:
```python
tool = StockSummaryTool(api_key="your-api-key")
```

## Usage

### Command Line

```bash
# Basic usage
python stock_summary.py AAPL

# With verbose output (shows quote data, news count, ratings count)
python stock_summary.py NVDA --verbose

# Use a different Ollama model
python stock_summary.py MSFT --model llama3.2
```

### As a Module

```python
from stock_summary import StockSummaryTool

# Initialize the tool
tool = StockSummaryTool(model="qwen2.5-coder:7b")

# Generate a summary
result = tool.get_summary("AAPL")

# Access the data
print(f"Ticker: {result.ticker}")
print(f"Context Level: {result.context_level}")
print(f"Summary: {result.summary}")

# Access underlying data
if result.data.quote:
    print(f"Price: ${result.data.quote.price:.2f}")
    print(f"Change: {result.data.quote.change_percent:+.2f}%")

for news in result.data.news:
    print(f"- {news.title}")
```

## Output Examples

### Stock-Specific Context (AAPL)
```
============================================================
Stock Summary: AAPL
Context Level: stock_specific
============================================================

Apple Inc. (AAPL) closed at $246.70, down slightly by 0.13%, following
a quiet trading session. Citigroup maintains a "Buy" rating on the stock
with a price target of $315, while Evercore ISI Group and Wedbush both
reiterate their "Outperform" stance, setting higher price targets at
$330 and $350 respectively.
```

### Market Context (Invalid Ticker)
```
============================================================
Stock Summary: XXXXX
Context Level: market_context
============================================================

Limited data is available for the stock XXXXX. The top gainers in the
market include PAVM (+233.8%), GITS (+219.1%), and SLGB (+117.5%),
while the top losers are VERO (-54.3%), IBG (-37.0%), and SURG (-29.4%).
Upcoming economic events include Challenger Job Cuts and Import Price Index.
```

## Fallback Logic

The tool automatically selects the appropriate context level based on available data:

| Level | Trigger | Data Used |
|-------|---------|-----------|
| `stock_specific` | Quote + (news OR ratings) available | Quote, news, ratings |
| `sector_context` | Quote only, no news/ratings | Quote + market movers |
| `market_context` | Limited/no ticker data | Market movers, economic calendar |

## Data Models

### StockSummary
The main output model containing:
- `ticker`: Stock symbol
- `context_level`: One of `stock_specific`, `sector_context`, `market_context`
- `summary`: AI-generated narrative
- `data`: StockData object with all fetched data
- `generated_at`: ISO timestamp

### StockData
Aggregated data container with:
- `quote`: QuoteData (price, change, volume, market cap)
- `news`: List of NewsItem objects
- `ratings`: List of RatingItem objects
- `gainers`: List of MoverItem objects
- `losers`: List of MoverItem objects
- `economic_events`: List of EconomicEvent objects

## API Endpoints Used

- `/api/v2/quoteDelayed` - Stock quotes
- `/api/v2/news/` - News articles
- `/api/v2.1/calendar/ratings` - Analyst ratings
- `/api/v1/market/movers` - Market gainers/losers
- `/api/v2.1/calendar/economics` - Economic calendar

## Supported Ollama Models

Any Ollama model works, but recommended models for Apple Silicon:
- `qwen2.5-coder:7b` (default) - Fast, good at structured output
- `llama3.2` - Good general purpose
- `mistral:7b` - Balanced performance

## Error Handling

- **API errors**: Logged as warnings, tool continues with available data
- **No quote data**: Falls back to market context
- **LLM unavailable**: Returns structured data summary without narrative
- **Invalid ticker**: Returns market context summary

## License

MIT License
