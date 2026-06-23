# Pyplot

Simple web app that plots and analyzes an equation in `x` with NumPy, Matplotlib and SymPy.

Allowed functions: `abs`, `absolute`, `acos`, `arccos`, `arcsin`, `arctan`, `arctan2`, `asin`, `atan`, `atan2`, `ceil`, `cos`, `cosh`, `degrees`, `exp`, `floor`, `log`, `log10`, `log2`, `maximum`, `minimum`, `radians`, `round`, `sin`, `sinh`, `sqrt`, `tan`, `tanh`.

Allowed constants: `e`, `pi`.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
func start
```

Open `http://localhost:7071/`.

## Deploy to Azure

```powershell
az login
az account set --subscription "YOUR_SUBSCRIPTION"
func azure functionapp publish <function-app-name> --python
```
