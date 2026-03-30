import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.search.web_search import run_web_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Run web search using src.tools.search.run_web_search")
    parser.add_argument("query", nargs="+", help="Search query text.")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum number of results (1-20).")
    parser.add_argument("--region", default="us-en", help="Search region.")
    parser.add_argument("--safesearch", default="moderate", choices=["on", "moderate", "off"], help="Safe search mode.")
    parser.add_argument("--backend", default="duckduckgo", help="DDGS backend (if supported).")
    parser.add_argument("--json", action="store_true", help="Print full JSON response.")
    args = parser.parse_args()

    query = " ".join(args.query)
    response = run_web_search(
        query=query,
        max_results=args.max_results,
        region=args.region,
        safesearch=args.safesearch,
        backend=args.backend,
    )

    if args.json:
        print(json.dumps(response, indent=2, ensure_ascii=False))
        return

    print(f"Provider: {response.get('provider')}")
    if response.get("warnings"):
        print(f"Warnings: {response['warnings']}")
    if response.get("error"):
        print(f"Error: {response['error']}")
        print(f"Details: {response.get('details', {})}")
        return

    for item in response.get("results", []):
        print(f"\n[{item.get('position')}] {item.get('title')}")
        print(f"URL: {item.get('url')}")
        print(f"Snippet: {item.get('snippet')}")

# python scripts/web_search_test.py "latest qwen model" --max-results 5 --json
# python scripts/web_search_test.py "qwen agent mcp"


if __name__ == "__main__":
    main()
