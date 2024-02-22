import ast
import astor
import subprocess
import sys

from tinyllm.agent.tool.tool import Tool


def code_interpreter(code: str):
    modified_code = modify_code_to_print_last_expression(code)
    result = run_code(modified_code)
    if result.stdout == b'' and result.stderr == b'':
        return "The code did not return anything. Did you forget to print?"

    return f"StdOut:\n{result.stdout.decode('utf-8')}\nStdErr:\n{result.stderr.decode('utf-8')}"


def run_code(code):
    return subprocess.run(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def modify_code_to_print_last_expression(code):
    """
    Parse the code and modify it to print the last expression or variable assignment.
    """
    try:
        tree = ast.parse(code)
        last_node = tree.body[-1]

        # Check if the last node is an expression or a variable assignment
        if isinstance(last_node, (ast.Expr, ast.Assign)):
            # Create a print node
            if isinstance(last_node, ast.Assign):
                # For variable assignment, print the variable
                print_node = ast.Expr(value=ast.Call(func=ast.Name(id='print', ctx=ast.Load()),
                                                     args=[last_node.targets[0]],
                                                     keywords=[]))
            else:
                # For direct expressions, print the expression
                print_node = ast.Expr(value=ast.Call(func=ast.Name(id='print', ctx=ast.Load()),
                                                     args=[last_node.value],
                                                     keywords=[]))

            # Add the print node to the AST
            tree.body.append(print_node)

        # Convert the AST back to code
        return astor.to_source(tree)

    except SyntaxError as e:
        return f"SyntaxError: {e}"



def get_code_interpreter_tool():
    return Tool(
        name="code_interpreter",
        description="""
        Use this tool to run python code. 
        """,
        python_lambda=code_interpreter,
        parameters={
            "type": "object",
            "required": ["code"],
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
            }
        }
    )
