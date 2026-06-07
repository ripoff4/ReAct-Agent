from graph import agent
from langchain_core.messages import HumanMessage

while True:
    user_input = input("User : ")
    if user_input.lower().strip() == "exit":
        break
    result = agent.invoke({
        "messages": [HumanMessage(content=user_input)]
    })
