from tools import classify_intent
from state import AgentState
from prompts import STORING_MECHANISM

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

import json
import os
from dotenv import load_dotenv
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
        return state
    elif intent.lower().strip() == "memory":

        query = f"""
        Query : {prompt}
        Answer_Mechanism : {STORING_MECHANISM}
        """
        response = llm.invoke(query).content.strip()

        role, answer = response.split("|", 1)
        try:
            with open(filepath, "r") as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        list = ["Personal Details", "Personal Preferences",
                "Personal Ambitions", "Persons Struggles"]
        if role in list:
            data.setdefault(role, []).append(answer)
            with open("memories/data.json", "w") as file:
                json.dump(data, file, indent=4)
        else:
            print("Internal Problem as role is not given properly by llm")
    else:
        print("Internal Problem as intent is not classified properly by llm")
    return state


def load_memory(state: AgentState):

    if not os.path.isfile("memories/data.json"):
        print("Json file not found!")
        return state
    with open("memories/data.json") as file:
        data = json.load(file)

    personal_details = data.setdefault("Personal Details", [])
    personal_preferences_list = data.setdefault("Personal Preferences", [])
    if len(personal_preferences_list) >= 3:
        personal_preferences = personal_preferences_list[-3:]
    else:
        personal_preferences = personal_preferences_list

    personal_ambitions_list = data.setdefault("Personal Ambitions", [])
    if len(personal_ambitions_list) >= 3:
        personal_ambitions = personal_ambitions_list[-3:]
    else:
        personal_ambitions = personal_ambitions_list

    personal_struggles_list = data.setdefault("Persons Struggles", [])
    if len(personal_struggles_list) >= 3:
        personal_struggles = personal_struggles_list[-3:]
    else:
        personal_struggles = personal_struggles_list

    memory = f"""
    details : {personal_details}
    preferences : {personal_preferences}
    ambitions : {personal_ambitions}
    struggles : {personal_struggles}
    """

    query = f"""
    memory : {memory}
    query : {state["messages"][-1].content}
    From the information given answer accordingly
    """

    answer = llm.invoke(query)
    print(f"AI : {answer.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("store_memory", store_memory)
graph.add_node("load_memory", load_memory)
graph.add_edge(START, "store_memory")
graph.add_edge("store_memory", "load_memory")
graph.add_edge("load_memory", END)

agent = graph.compile()
