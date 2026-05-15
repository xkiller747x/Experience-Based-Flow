"""Unified entry point for Logistic-AI experiments.

Usage:
    python run.py <command> [args...]

Commands:
    generate-cases       Generate historical case bank (default 30k)
    generate-validation  Generate validation query set (default 1000)
    evaluate-baselines   Run standard baselines (no_rag, bm25_rag, tfidf_rag, dense_rag)
    evaluate-se-rag      Run SE-RAG evaluation

Examples:
    python run.py generate-cases --count 30000
    python run.py generate-validation --count 1000 --seed 20260428
    python run.py evaluate-baselines --query-count 1000 --seed 20260428
    python run.py evaluate-se-rag --query-count 1000 --seed 20260428
    python run.py evaluate-baselines --methods no_rag bm25_rag
    python run.py --help
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


COMMANDS = {
    "generate-cases": {
        "script": "parallel_case_generator.py",
        "help": "Generate historical case bank (30k by default)",
    },
    "generate-validation": {
        "script": "generate_validation_queries.py",
        "help": "Generate validation query set (1000 by default)",
    },
    "evaluate-baselines": {
        "script": "eval_standard_baselines.py",
        "help": "Run standard baselines (no_rag, bm25_rag, tfidf_rag, dense_rag)",
    },
    "evaluate-se-rag": {
        "script": "eval_se_rag.py",
        "help": "Run SE-RAG evaluation",
    },
}


def print_help():
    print("Usage: python run.py <command> [args...]")
    print()
    print("Commands:")
    for name, info in COMMANDS.items():
        print(f"  {name:<25s} {info['help']}")
    print()
    print("Examples:")
    print("  python run.py generate-cases --count 30000")
    print("  python run.py generate-validation --count 1000")
    print("  python run.py evaluate-baselines --query-count 1000 --seed 20260428")
    print("  python run.py evaluate-se-rag --query-count 1000 --seed 20260428")
    print()
    print("Pass --help to any command for full options.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print_help()
        sys.exit(0 if len(sys.argv) > 1 else 1)

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(COMMANDS.keys())}")
        sys.exit(1)

    script = SCRIPTS / COMMANDS[cmd]["script"]
    if not script.exists():
        print(f"Error: script not found {script}")
        print(f"Run the setup script first to generate {COMMANDS[cmd]['script']}")
        sys.exit(1)

    script_args = sys.argv[2:]

    # Forward --help to the underlying script
    if "-h" in script_args or "--help" in script_args:
        subprocess.call([sys.executable, str(script), "--help"])
        sys.exit(0)

    cmdline = [sys.executable, str(script)] + script_args
    display = " ".join(str(a) if " " not in str(a) else f'"{a}"' for a in cmdline)
    print(f"[run.py] $ {display}")
    sys.exit(subprocess.call(cmdline))


if __name__ == "__main__":
    main()
