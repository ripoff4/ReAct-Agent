from prompts import CLASSIFY_INTENT


def classify_intent(llm, prompt):
    "This is used to classify intent as either a question about user/general query"
    query = f"""
    Prompt : {prompt}
    Laws to follow : {CLASSIFY_INTENT}
    """
    result = llm.invoke(query)
    return result.content
