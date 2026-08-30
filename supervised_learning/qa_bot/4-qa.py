#!/usr/bin/env python3
"""
Multi-document Question Answering module using semantic search and BERT QA
"""
semantic_search = __import__('3-semantic_search').semantic_search
qa_func = __import__('0-qa').question_answer


def question_answer(corpus_path):
    """
    Continuously prompts the user for questions, finds the most relevant
    document via semantic search, and answers using BERT QA.
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

        reference = semantic_search(corpus_path, user_input)
        if not reference:
            print("A: Sorry, I do not understand your question.")
            continue

        answer = qa_func(user_input, reference)
        if not answer:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
