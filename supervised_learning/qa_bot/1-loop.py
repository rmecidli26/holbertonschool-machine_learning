#!/usr/bin/env python3
"""
A script that takes user input in a loop and responds accordingly.
"""

if __name__ == "__main__":
    while True:
        try:
            user_input = input("Q: ")
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if user_input.strip().lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break
        
        print("A:")
