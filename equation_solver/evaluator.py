import ast


def extract_expression(equation_string):
    """Extract the math expression part (before '=')."""
    if '=' in equation_string:
        equation_string = equation_string.rsplit('=', 1)[0]
    return equation_string.strip()


def is_safe_expression(expr_str):
    """
    Check if expression is safe to evaluate using AST whitelist.
    Only allow arithmetic operations on constants.
    """
    try:
        tree = ast.parse(expr_str, mode='eval')
    except SyntaxError:
        return False

    # Whitelist of allowed node types
    allowed_types = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.USub,
        ast.UAdd,
    }

    for node in ast.walk(tree):
        if type(node) not in allowed_types:
            return False

    return True


def evaluate_expression(equation_string):
    """
    Safely evaluate a mathematical expression string.
    Returns: numeric result (int/float) or error string.
    """
    expr_str = extract_expression(equation_string)

    if not expr_str:
        return "ERROR: empty expression"

    if not is_safe_expression(expr_str):
        return "ERROR: unsafe or invalid expression"

    try:
        # Compile and evaluate with no builtins
        code = compile(ast.parse(expr_str, mode='eval'), '<string>', 'eval')
        result = eval(code, {"__builtins__": {}}, {})

        # Convert whole-number floats to int
        if isinstance(result, float) and result == int(result):
            result = int(result)

        return result

    except ZeroDivisionError:
        return "ERROR: division by zero"
    except Exception as e:
        return f"ERROR: {type(e).__name__}"
