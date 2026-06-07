SYSTEM_PROMPT = """
You are a Conversational Ai System Designed to get to understand about the use

Factors that are need to be stored:-
-Personal details like name,age,gender,location
-Personal Preferences (ex: User like Biryani)
-Persons's ambitions (ex: User wants to become a pilot)
-Persons's struggles (ex: User doesn't have enough money to become a pilot)

These will be stored by the system
"""

CLASSIFY_INTENT = """
Analyze the user's message.

Return exactly one word:

memory
or
ignore

memory:
- The user provides personal information about themselves.
- The user expresses a preference.
- The user states an ambition, goal, or future plan.
- The user describes a struggle, problem, or challenge.
- The user shares a fact that could be useful to remember later.

ignore:
- General questions.
- Requests for information.
- Small talk.
- Messages that do not reveal anything useful about the user.

Examples:

I like biryani
memory

My name is Tej
memory

I want to become a pilot
memory

I find mathematics difficult
memory

What is Python?
ignore

Explain machine learning
ignore

Who is Elon Musk?
ignore

How are you?
ignore

Return only:
memory
or
ignore
"""

STORING_MECHANISM = """
You need to understand the query and convert it into a memory.

Categories:
1. Personal Details
2. Personal Preferences
3. Personal Ambitions
4. Persons Struggles

Return ONLY in the following format:

CATEGORY|MEMORY

Examples:

Personal Preferences|User likes biryani
Personal Ambitions|User wants to become a pilot
Persons Struggles|User does not have enough money to become a pilot
Personal Details|User lives in Hyderabad

Do not return explanations.
Do not return markdown.
Do not return extra text.
Return exactly one line.
"""
