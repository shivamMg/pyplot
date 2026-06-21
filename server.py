import ast
import base64
import json
import math
import os
import tempfile
from io import BytesIO
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import azure.functions as func
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


MAX_EXPRESSION_LENGTH = 300
MAX_POINTS = 10000
MAX_ABS_NUMBER = 1_000_000
MAX_POWER_EXPONENT = 20
DEFAULT_START = -10.0
DEFAULT_END = 10.0
DEFAULT_STEP = 0.1

ALLOWED_FUNCTIONS = {
    "abs": np.abs,
    "absolute": np.absolute,
    "acos": np.arccos,
    "arccos": np.arccos,
    "arcsin": np.arcsin,
    "arctan": np.arctan,
    "arctan2": np.arctan2,
    "asin": np.arcsin,
    "atan": np.arctan,
    "atan2": np.arctan2,
    "ceil": np.ceil,
    "cos": np.cos,
    "cosh": np.cosh,
    "degrees": np.degrees,
    "exp": np.exp,
    "floor": np.floor,
    "log": np.log,
    "log10": np.log10,
    "log2": np.log2,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "radians": np.radians,
    "round": np.round,
    "sin": np.sin,
    "sinh": np.sinh,
    "sqrt": np.sqrt,
    "tan": np.tan,
    "tanh": np.tanh,
}

ALLOWED_CONSTANTS = {
    "e": np.e,
    "pi": np.pi,
}

ALLOWED_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)
ALLOWED_UNARY_OPS = (ast.UAdd, ast.USub)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


class ExpressionValidationError(ValueError):
    pass


class ExpressionValidator(ast.NodeVisitor):
    def __init__(self):
        self.has_x = False

    def generic_visit(self, node):
        raise ExpressionValidationError(f"Unsupported expression element: {type(node).__name__}")

    def visit_Expression(self, node):
        self.visit(node.body)
        if not self.has_x:
            raise ExpressionValidationError("Expression must use x")

    def visit_BinOp(self, node):
        if not isinstance(node.op, ALLOWED_BIN_OPS):
            raise ExpressionValidationError(f"Operator {type(node.op).__name__} is not allowed")
        if isinstance(node.op, ast.Pow):
            self._validate_power(node)
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node):
        if not isinstance(node.op, ALLOWED_UNARY_OPS):
            raise ExpressionValidationError(f"Operator {type(node.op).__name__} is not allowed")
        self.visit(node.operand)

    def visit_Constant(self, node):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionValidationError("Only numeric constants are allowed")
        if not math.isfinite(float(node.value)) or abs(float(node.value)) > MAX_ABS_NUMBER:
            raise ExpressionValidationError(f"Numeric constants must be finite and <= {MAX_ABS_NUMBER:g}")

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            raise ExpressionValidationError("Only reading values is allowed")
        if node.id == "x":
            self.has_x = True
            return
        if node.id in ALLOWED_FUNCTIONS or node.id in ALLOWED_CONSTANTS:
            return
        raise ExpressionValidationError(f"Unknown name: {node.id}")

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ExpressionValidationError("Only direct function calls are allowed")
        if node.func.id not in ALLOWED_FUNCTIONS:
            raise ExpressionValidationError(f"Function is not allowed: {node.func.id}")
        if node.keywords:
            raise ExpressionValidationError("Keyword arguments are not allowed")
        if not 1 <= len(node.args) <= 2:
            raise ExpressionValidationError("Functions must use one or two arguments")
        for arg in node.args:
            self.visit(arg)

    def _validate_power(self, node):
        if not contains_x(node):
            raise ExpressionValidationError("Constant-only exponentiation is not allowed")
        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
            exponent = float(node.right.value)
            if not math.isfinite(exponent) or abs(exponent) > MAX_POWER_EXPONENT:
                raise ExpressionValidationError(f"Power exponents must be between {-MAX_POWER_EXPONENT} and {MAX_POWER_EXPONENT}")


def contains_x(node):
    return any(isinstance(child, ast.Name) and child.id == "x" for child in ast.walk(node))


def parse_expression(expression):
    if not isinstance(expression, str) or not expression.strip():
        raise ExpressionValidationError("Expression is required")
    expression = expression.strip()
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ExpressionValidationError(f"Expression must be {MAX_EXPRESSION_LENGTH} characters or fewer")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionValidationError("Expression is not valid Python syntax") from exc
    ExpressionValidator().visit(tree)
    return expression, compile(tree, "<expression>", "eval")


def parse_float(value, default, name):
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def parse_range(data):
    start = parse_float(data.get("start"), DEFAULT_START, "start")
    end = parse_float(data.get("end"), DEFAULT_END, "end")
    step = parse_float(data.get("step"), DEFAULT_STEP, "step")
    if step <= 0:
        raise ValueError("step must be greater than 0")
    if start >= end:
        raise ValueError("start must be less than end")
    points = math.floor((end - start) / step) + 1
    if points < 2:
        raise ValueError("range must include at least two points")
    if points > MAX_POINTS:
        raise ValueError(f"range produces too many points; maximum is {MAX_POINTS}")
    x = np.linspace(start, start + step * (points - 1), points)
    return start, end, step, points, x


def evaluate_expression(compiled_expression, x):
    env = {"x": x, **ALLOWED_FUNCTIONS, **ALLOWED_CONSTANTS}
    with np.errstate(all="ignore"):
        y = eval(compiled_expression, {"__builtins__": {}}, env)
    y = np.asarray(y, dtype=float)
    if y.shape == ():
        y = np.full_like(x, float(y))
    if y.shape != x.shape:
        raise ValueError("Expression must return one value for each x")
    finite = np.isfinite(y)
    if not np.any(finite):
        raise ValueError("Expression did not produce any finite y values in this range")
    return np.where(finite, y, np.nan), int(np.count_nonzero(finite))


def render_plot(expression, x, y):
    figure = Figure(figsize=(8, 5), dpi=120, tight_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(x, y, color="#2563eb", linewidth=2)
    axis.axhline(0, color="#94a3b8", linewidth=0.8)
    axis.axvline(0, color="#94a3b8", linewidth=0.8)
    axis.grid(True, color="#e2e8f0", linewidth=0.8)
    axis.set_title(f"y = {expression}")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def json_response(payload, status_code=200):
    return func.HttpResponse(
        json.dumps(payload),
        status_code=status_code,
        mimetype="application/json",
    )


def get_request_data(req):
    if req.method == "GET":
        return dict(req.params)
    try:
        data = req.get_json()
    except ValueError:
        raise ValueError("Request body must be JSON")
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")
    return data


@app.route(route="{*path}", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def index(req: func.HttpRequest) -> func.HttpResponse:
    path = (req.route_params.get("path") or "").strip("/")
    if path and path != "index.html":
        return func.HttpResponse("Not found", status_code=404)
    html = Path(__file__).with_name("index.html").read_text(encoding="utf-8")
    return func.HttpResponse(html, mimetype="text/html")


@app.route(route="api/plot", methods=["GET", "POST"], auth_level=func.AuthLevel.ANONYMOUS)
def plot(req: func.HttpRequest) -> func.HttpResponse:
    try:
        data = get_request_data(req)
        expression, compiled_expression = parse_expression(data.get("expression"))
        start, end, step, points, x = parse_range(data)
        y, finite_points = evaluate_expression(compiled_expression, x)
        image_base64 = render_plot(expression, x, y)
        return json_response({
            "imageBase64": image_base64,
            "mimeType": "image/png",
            "expression": expression,
            "range": {
                "start": start,
                "end": end,
                "step": step,
                "actualEnd": float(x[-1]),
                "points": points,
            },
            "finitePoints": finite_points,
        })
    except (ExpressionValidationError, ValueError, FloatingPointError) as exc:
        return json_response({"error": str(exc)}, status_code=400)
