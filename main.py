from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.lc_agent import LangChainSearchAgent
from seed_data_large import ensure_seed_data


def main() -> None:
    print("=" * 72)
    print("  Agentic Search - Enterprise Search Assistant")
    print("  Multi-source search with Qwen-compatible tool calling")
    print("=" * 72)

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("\n[!] DASHSCOPE_API_KEY is not set.")
        print("    The app can still use the local fallback search path.")
        print("    Optional: set DASHSCOPE_BASE_URL and DASHSCOPE_MODEL.")

    print("\n[*] Preparing demo data...")
    ensure_seed_data()

    agent = LangChainSearchAgent()

    print("\n" + "=" * 72)
    print("  System ready. Enter a question to search, or type 'quit' to exit.")
    print("  Searchable sources:")
    print("    1. SQLite structured records: employees, departments, projects, contracts, products")
    print("    2. Chroma vector collections: company profile, technical docs, meeting notes")
    print("    3. Whoosh keyword indexes: policies and engineering articles")
    print("    4. Sample code repository")
    print("    5. Simulated enterprise systems: HR, finance, project, wiki")
    print("    6. Operation logs")
    print("=" * 72)

    while True:
        try:
            question = input("\n[?] Question: ").strip()
            if not question:
                continue
            if question.lower() in {"quit", "exit", "q"}:
                print("[*] Goodbye.")
                break

            answer = agent.search(question, verbose=True)
            print("\n" + answer)
        except KeyboardInterrupt:
            print("\n[*] Goodbye.")
            break
        except Exception as exc:
            print(f"\n[!] Error: {exc}")


if __name__ == "__main__":
    main()
