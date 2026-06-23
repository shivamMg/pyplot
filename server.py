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
import sympy
from sympy.calculus.util import (
    continuous_domain,
    function_range,
    periodicity,
    singularities,
)
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

SYMPY_FUNCTIONS = {
    "abs": sympy.Abs,
    "absolute": sympy.Abs,
    "acos": sympy.acos,
    "arccos": sympy.acos,
    "arcsin": sympy.asin,
    "arctan": sympy.atan,
    "arctan2": sympy.atan2,
    "asin": sympy.asin,
    "atan": sympy.atan,
    "atan2": sympy.atan2,
    "ceil": sympy.ceiling,
    "cos": sympy.cos,
    "cosh": sympy.cosh,
    "degrees": lambda a: a * 180 / sympy.pi,
    "exp": sympy.exp,
    "floor": sympy.floor,
    "log": sympy.log,
    "log10": lambda a: sympy.log(a, 10),
    "log2": lambda a: sympy.log(a, 2),
    "maximum": sympy.Max,
    "minimum": sympy.Min,
    "radians": lambda a: a * sympy.pi / 180,
    "round": lambda a: sympy.floor(a + sympy.Rational(1, 2)),
    "sin": sympy.sin,
    "sinh": sympy.sinh,
    "sqrt": sympy.sqrt,
    "tan": sympy.tan,
    "tanh": sympy.tanh,
}

SYMPY_CONSTANTS = {
    "e": sympy.E,
    "pi": sympy.pi,
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


TRIG_FUNCTIONS = (sympy.sin, sympy.cos, sympy.tan, sympy.cot, sympy.sec, sympy.csc)


def build_sympy_expr(compiled_expression):
    x = sympy.Symbol("x", real=True)
    env = {"x": x, **SYMPY_FUNCTIONS, **SYMPY_CONSTANTS}
    expr = eval(compiled_expression, {"__builtins__": {}}, env)
    return x, sympy.sympify(expr)


def _expr_repr(value):
    return sympy.sstr(value), sympy.latex(value)


def _solve_real(equation, x):
    solutions = sympy.solve(equation, x)
    real = []
    for solution in solutions:
        try:
            if sympy.im(solution) != 0:
                continue
        except TypeError:
            continue
        simplified = sympy.simplify(solution)
        if simplified not in real:
            real.append(simplified)
    return real


def _solutions_repr(solutions, variable="x"):
    if not solutions:
        return "no real solutions", r"\text{no real solutions}"
    text = ", ".join(f"{variable} = {sympy.sstr(s)}" for s in solutions)
    latex = r",\quad ".join(f"{variable} = {sympy.latex(s)}" for s in solutions)
    return text, latex


def _to_exact(value):
    if isinstance(value, float) and value.is_integer():
        return sympy.Integer(int(value))
    return sympy.Rational(value).limit_denominator(1_000_000)


def _bound_repr(value):
    if value == sympy.oo:
        return "+\u221e", r"+\infty"
    if value == -sympy.oo:
        return "\u2212\u221e", r"-\infty"
    return _expr_repr(value)


def _limit_repr(expr, x, point):
    value = sympy.limit(expr, x, point)
    if isinstance(value, sympy.AccumBounds) or value.has(sympy.AccumBounds):
        return "does not exist (oscillates)", r"\text{does not exist (oscillates)}"
    return _bound_repr(value)


def compute_analysis(expr, x, start, end):
    analysis = {"latexEquation": None, "groups": []}
    try:
        analysis["latexEquation"] = "y = " + sympy.latex(expr)
    except Exception:
        analysis["latexEquation"] = None

    def add(props, label, description, producer):
        try:
            produced = producer()
        except Exception:
            return
        if not produced:
            return
        text, latex = produced
        if text in (None, ""):
            return
        props.append({
            "label": label,
            "description": description,
            "value": text,
            "latex": latex,
        })

    has_trig = expr.has(*TRIG_FUNCTIONS)

    # Group 1: Shape & Behavior
    first_derivative = sympy.diff(expr, x)
    second_derivative = sympy.diff(expr, x, 2)
    g1 = []
    add(g1, "First derivative", "Slope / rate of change (dy/dx).",
        lambda: _expr_repr(first_derivative))
    add(g1, "Second derivative", "Concavity (d\u00b2y/dx\u00b2).",
        lambda: _expr_repr(second_derivative))
    add(g1, "Critical points", "x-values where the slope is zero (turning points).",
        lambda: _solutions_repr(_solve_real(first_derivative, x)))
    add(g1, "Inflection points", "x-values where concavity changes.",
        lambda: _solutions_repr(_solve_real(second_derivative, x)))
    analysis["groups"].append({"id": 1, "title": "Shape & Behavior", "properties": g1})

    # Range is reused by groups 2 and 3.
    value_range = None
    try:
        domain = continuous_domain(expr, x, sympy.S.Reals)
        value_range = function_range(expr, x, domain)
    except Exception:
        value_range = None

    def maxmin(which):
        if value_range is None:
            return None
        bound = value_range.sup if which == "max" else value_range.inf
        return _bound_repr(bound)

    def y_intercept():
        value = expr.subs(x, 0)
        if value in (sympy.zoo, sympy.nan) or value.has(sympy.oo, sympy.zoo, sympy.nan):
            return None
        return _expr_repr(sympy.simplify(value))

    # Group 2: Key Points
    g2 = []
    add(g2, "x-intercepts (roots)", "Where the curve crosses y = 0.",
        lambda: _solutions_repr(_solve_real(expr, x)))
    add(g2, "y-intercept", "Value where x = 0.", y_intercept)
    add(g2, "Maximum value", "Largest y over all real x.", lambda: maxmin("max"))
    add(g2, "Minimum value", "Smallest y over all real x.", lambda: maxmin("min"))
    analysis["groups"].append({"id": 2, "title": "Key Points", "properties": g2})

    def period_repr():
        period = periodicity(expr, x)
        if period is None:
            return "not periodic", r"\text{not periodic}"
        return _expr_repr(period)

    def asymptotes_repr():
        points = singularities(expr, x, sympy.S.Reals)
        if points == sympy.S.EmptySet or points == sympy.EmptySet:
            return "none", r"\text{none}"
        return _expr_repr(points)

    # Group 3: Bounds & Asymptotics
    g3 = []
    add(g3, "Range", "All possible y-values.",
        lambda: _expr_repr(value_range) if value_range is not None else None)
    add(g3, "Limit as x \u2192 +\u221e", "Value approached as x grows large.",
        lambda: _limit_repr(expr, x, sympy.oo))
    add(g3, "Limit as x \u2192 \u2212\u221e", "Value approached as x decreases.",
        lambda: _limit_repr(expr, x, -sympy.oo))
    add(g3, "Periodicity", "Interval over which the curve repeats.", period_repr)
    add(g3, "Vertical asymptotes", "x-values where the curve diverges.", asymptotes_repr)
    analysis["groups"].append({"id": 3, "title": "Bounds & Asymptotics", "properties": g3})

    def indefinite_integral():
        antiderivative = sympy.integrate(expr, x)
        if antiderivative.has(sympy.Integral):
            return None
        return (sympy.sstr(antiderivative) + " + C",
                sympy.latex(antiderivative) + " + C")

    def definite_integral():
        lower, upper = _to_exact(start), _to_exact(end)
        value = sympy.integrate(expr, (x, lower, upper))
        if value.has(sympy.Integral) or not value.is_finite:
            return None
        approx = sympy.N(value, 6)
        return (f"{sympy.sstr(value)} \u2248 {approx}",
                f"{sympy.latex(value)} \\approx {sympy.latex(approx)}")

    def series_repr():
        series = sympy.series(expr, x, 0, 6)
        return _expr_repr(series)

    # Group 4: Calculus
    g4 = []
    add(g4, "Indefinite integral", "Antiderivative (\u222b y dx).", indefinite_integral)
    add(g4, f"Definite integral [{start:g}, {end:g}]",
        "Signed area under the curve over the plotted range.", definite_integral)
    add(g4, "Series expansion at x = 0",
        "Polynomial approximation near x = 0 (Taylor series).", series_repr)
    analysis["groups"].append({"id": 4, "title": "Calculus", "properties": g4})

    is_polynomial = bool(expr.is_polynomial(x))

    def trig_form():
        if not has_trig:
            return None
        combined = sympy.simplify(sympy.trigsimp(expr))
        if combined == expr:
            return None
        return _expr_repr(combined)

    def degree_repr():
        if not is_polynomial:
            return None
        degree = sympy.degree(expr, x)
        return str(degree), str(degree)

    # Group 5: Algebraic Structure
    g5 = []
    add(g5, "Simplified form", "Algebraically simplified expression.",
        lambda: _expr_repr(sympy.simplify(expr)))
    add(g5, "Factored form", "Expression written as a product of factors.",
        lambda: _expr_repr(sympy.factor(expr)))
    add(g5, "Trigonometric form", "Combined single-wave form (amplitude / phase).", trig_form)
    add(g5, "Polynomial degree", "Highest power of x (polynomials only).", degree_repr)
    add(g5, "Is a polynomial?", "Whether the expression is a polynomial in x.",
        lambda: (("yes", r"\text{yes}") if is_polynomial else ("no", r"\text{no}")))
    analysis["groups"].append({"id": 5, "title": "Algebraic Structure", "properties": g5})

    return analysis


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
        analysis = None
        try:
            x_sym, sym_expr = build_sympy_expr(compiled_expression)
            analysis = compute_analysis(sym_expr, x_sym, start, end)
        except Exception:
            analysis = None
        return json_response({
            "imageBase64": image_base64,
            "mimeType": "image/png",
            "expression": expression,
            "latexEquation": analysis.get("latexEquation") if analysis else None,
            "analysis": {"groups": analysis.get("groups")} if analysis else None,
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
