"""
test_connection.py

STEP 1 — run this first.
Confirms your ANTHROPIC_API_KEY works and the SDK is wired correctly.

Usage:
    python test_connection.py
"""

from claude_client import call_claude


def main():
    print("Testing connection to Claude API...\n")

    response = call_claude(
        system="You are a helpful assistant. Be brief.",
        user="Reply with exactly: 'Connection successful. Pipeline ready.'",
        max_tokens=50,
    )

    print("Response received:")
    print("─" * 50)
    print(response)
    print("─" * 50)
    print("\nIf you see a response above, your setup is working.")
    print("Paste this output back to continue to Step 2.")


if __name__ == "__main__":
    main()
