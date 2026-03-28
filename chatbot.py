"""
chatbot.py
----------
Interactive command-line chatbot for interrogating Prof. Devreotes' papers.

Usage:
    python chatbot.py

Special commands (type at the prompt):
    /papers     — List all papers in the corpus
    /sources    — Show the source chunks used for the last answer
    /reset      — Clear conversation history
    /help       — Show these commands
    /quit       — Exit
"""

import os
import sys
import traceback
import textwrap
from rag_engine import RAGEngine


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
WIDTH = 80


def hr(char: str = "─") -> None:
    print(char * WIDTH)


def print_wrapped(text: str, indent: int = 0) -> None:
    prefix = " " * indent
    for line in text.split("\n"):
        if line.strip():
            print(textwrap.fill(line, width=WIDTH - indent, initial_indent=prefix,
                                subsequent_indent=prefix))
        else:
            print()


def print_header() -> None:
    os.system("clear" if os.name == "posix" else "cls")
    hr("═")
    print("  Devreotes Lab — Research Chatbot")
    print("  Ask questions about Prof. Peter Devreotes' papers")
    hr("═")
    print("  Commands: /papers  /sources  /reset  /help  /quit")
    hr()
    print()


def print_sources(chunks: list[dict]) -> None:
    hr("·")
    print(f"  Sources retrieved ({len(chunks)} chunks):\n")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] {c['title']} ({c['year']}) — score: {c['score']:.4f}")
        print(f"       {', '.join(c['authors'][:2])}"
              f"{'  et al.' if len(c['authors']) > 2 else ''}")
        print(f"       Chunk {c['chunk_index']+1}/{c['total_chunks']}")
        print()
    hr("·")


def print_papers(papers: list[dict]) -> None:
    hr("·")
    print(f"  Papers in corpus ({len(papers)}):\n")
    for p in papers:
        print(f"  • {p['year']} — {p['title']}")
        print(f"    {p['journal']}")
        print(f"    {', '.join(p['authors'][:3])}"
              f"{'  et al.' if len(p['authors']) > 3 else ''}")
        if p["topics"]:
            print(f"    Topics: {', '.join(p['topics'])}")
        print()
    hr("·")


def print_help() -> None:
    hr("·")
    print("""
  Available commands:

  /papers   Show all papers in the corpus with metadata
  /sources  Show the source excerpts used to answer the last question
  /reset    Clear conversation history (start fresh)
  /help     Show this help message
  /quit     Exit the chatbot

  Tips:
  • Ask about specific findings, methods, or concepts
  • Ask comparative questions across papers
  • Ask about collaborators, timelines, or research themes
  • Follow-up questions work naturally (conversation is remembered)

  Example questions:
  • "What is the half-life of acetylcholine receptors?"
  • "Where are newly synthesized receptors before they reach the surface?"
  • "How did Devreotes show that receptor synthesis is de novo?"
  • "What methods were used to study receptor degradation?"
  • "Which papers deal with the Golgi apparatus?"
""")
    hr("·")


# ---------------------------------------------------------------------------
# Main chatbot loop
# ---------------------------------------------------------------------------
def main():
    # Honour a DEBUG env var for verbose error output
    debug = os.environ.get("CHATBOT_DEBUG", "0") == "1"

    print_header()

    try:
        engine = RAGEngine()
        engine.load()
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print("  Run `python build_index.py` first to build the index.\n")
        sys.exit(1)
    except EnvironmentError as e:
        print(f"\n  ERROR: {e}\n")
        sys.exit(1)

    last_chunks: list[dict] = []

    print("  Ready! Type your question below.\n")

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!\n")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("/quit", "/exit", "/q"):
            print("\n  Goodbye!\n")
            break

        elif cmd == "/help":
            print_help()
            continue

        elif cmd == "/papers":
            print_papers(engine.get_paper_list())
            continue

        elif cmd == "/sources":
            if last_chunks:
                print_sources(last_chunks)
            else:
                print("\n  No sources yet — ask a question first.\n")
            continue

        elif cmd == "/reset":
            engine.reset_conversation()
            last_chunks = []
            continue

        # --------------- RAG query ---------------
        print()
        print("  Thinking...", end="", flush=True)

        # Evaluate the guard BEFORE the try-block: Python does not allow
        # function calls directly in an except clause.
        _openai_api_error = openai_import_error_guard()
        try:
            answer, chunks = engine.ask(user_input)
            last_chunks = chunks
        except _openai_api_error as e:
            # Narrow handler: surface OpenAI API errors clearly
            print(f"\r  ERROR: OpenAI API call failed — {e}\n")
            if debug:
                traceback.print_exc()
            continue
        except Exception as e:
            # Broad handler: always show the error type so failures are diagnosable
            print(f"\r  ERROR [{type(e).__name__}]: {e}\n")
            if debug:
                traceback.print_exc()
            continue

        print(f"\r  {'Assistant':─<{WIDTH-2}}")
        print()
        print_wrapped(answer, indent=2)
        print()

        unique_papers = {}
        for c in chunks:
            if c["source"] not in unique_papers:
                unique_papers[c["source"]] = c["title"]
        print("  Sources: " + " | ".join(
            f"{v[:45]}..." if len(v) > 45 else v
            for v in unique_papers.values()
        ))
        print()
        hr()
        print()


def openai_import_error_guard():
    """Return the openai.APIError class, or Exception as a safe fallback."""
    try:
        import openai
        return openai.APIError
    except ImportError:
        return Exception


if __name__ == "__main__":
    main()
