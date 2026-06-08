from tools import classify_intent
from state import AgentState
from prompts import STORING_MECHANISM

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

import json
import os
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)


def store_memory(state: AgentState):

    filepath = "memories/data.json"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    prompt = state["messages"][-1].content

    intent = classify_intent(llm=llm, prompt=prompt)

    if intent.lower().strip() == "ignore":
        state["intent"] = "ignore"
        return state
    elif intent.lower().strip() == "memory":

        query = f"""
        Query : {prompt}
        Answer_Mechanism : {STORING_MECHANISM}
        """
        response = llm.invoke(query).content.strip()

        role, answer, importance = response.split("|", 2)
        embeddings = OpenAIEmbeddings()
        vector = embeddings.embed_query(answer)
        try:
            with open(filepath, "r") as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        list = ["Personal Details", "Personal Preferences",
                "Personal Ambitions", "Persons Struggles"]
        if role in list:
            memory_record = {
                "memory": answer,
                "category": role,
                "importance": int(importance),
                "embedding": vector,
                "access_count": 0,
                "created_at": datetime.now().isoformat(),
                "last_accessed": None
            }
            data.append(memory_record)
            with open("memories/data.json", "w") as file:
                json.dump(data, file, indent=4)
        else:
            print("Internal Problem as role is not given properly by llm")
    else:
        print("Internal Problem as intent is not classified properly by llm")
    state["intent"] = "memory"
    return state


def load_memory(state: AgentState):

    if not os.path.isfile("memories/data.json"):
        print("Json file not found!")
        return state
    with open("memories/data.json") as file:
        data = json.load(file)

    query = state["messages"][-1].content
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed_query(query)

    memory_scores = []
    for memory in data:

        memory_embedding = memory["embedding"]

        created_at = datetime.fromisoformat(
            memory["created_at"]
        )

        days_old = (
            datetime.now() - created_at
        ).days

        recency_score = max(0, 0.3-(days_old*0.01))
        frequency_score = min(memory["access_count"]*0.02, 0.03)
        similarity = cosine_similarity(
            [query_embedding],
            [memory_embedding]
        )[0][0]

        importance_score = memory["importance"] * 0.05

        score = (
            similarity
            + importance_score
            + recency_score
            + frequency_score
        )

        memory_scores.append(
            (memory, score)
        )

    memory_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_memories = memory_scores[:5]

    retrieved_memories = []

    for memory, score in top_memories:

        memory["access_count"] += 1
        memory["last_accessed"] = datetime.now().isoformat()
        retrieved_memories.append(
            memory["memory"]
        )
    with open("memories/data.json", "w") as file:
        json.dump(data, file, indent=4)

    memory_context = "\n".join(retrieved_memories)

    prompt = f"""
    memory context : {memory_context}
    query : {query}
    From the information given answer accordingly
    """

    state["messages"] = state["messages"] + [HumanMessage(content=prompt)]
    return state


def should_load_memory(state: AgentState):
    if state["intent"] == "ignore":
        return "load_memory"
    else:
        return "answer_question"


def answer_question(state: AgentState):

    answer = llm.invoke(state["messages"][-1].content)
    print(f"AI : {answer.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("store_memory", store_memory)
graph.add_node("load_memory", load_memory)
graph.add_node("answer_question", answer_question)
graph.add_edge(START, "store_memory")
graph.add_conditional_edges("store_memory", should_load_memory, {
                            "load_memory": "load_memory", "answer_question": "answer_question"})
graph.add_edge("load_memory", "answer_question")
graph.add_edge("answer_question", END)

agent = graph.compile()
