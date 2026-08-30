#!/usr/bin/env python3
"""
Interactive Question Answering loop module
"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Continuously prompts the user for questions and answers them
    using the reference text.
    """
    while True:
        try:
            user_input = input("Q: ")
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not user_input:
            continue

        if user_input.strip().lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break

        answer = question_answer(user_input, reference)
        if not answer:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
