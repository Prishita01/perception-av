from fastapi import FastAPI, HTTPException
import subprocess

app = FastAPI()


def run_script(script_name):
    result = subprocess.run(
        ["python", f"inference/{script_name}"],
        capture_output=True,
        text=True,
        cwd='/opt/airflow'
    )
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)
    return {"output": result.stdout}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect")
def detect():
    return run_script("detect.py")


@app.post("/depth")
def depth():
    return run_script("depth.py")


@app.post("/segment")
def segment():
    return run_script("segment.py")