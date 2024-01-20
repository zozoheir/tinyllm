from tinyllm.agent.tool.tool import Tool


def think_and_plan(execution_plan: str):
    return f"Execution plan: \n{execution_plan}"


def get_think_and_plan_tool():
    return Tool(
        name="think_and_plan",
        description="Use this tool to plan complex execution of a task using tools",
        python_lambda=think_and_plan,
        parameters={
            "type": "object",
            "required": ["symbol"],
            "properties": {
                "execution_plan": {
                    "type": "string",
                    "description": "Think step by step about the execution plan, and output a numbered list of tools to execute. Eg: 1. tool1: <text>> 2. tool2: <text>",
                },
            }
        }
    )
