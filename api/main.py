from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from agents import run_career_optimization

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/analyze", response_class=JSONResponse)
async def analyze_profile(
    profile_url: str = Form(...),
    target_role: str = Form(...),
    thread_id: int = Form(default=1)
):
    final_state = run_career_optimization(profile_url, target_role, thread_id)
    summary = final_state.values["messages"][-1].content  # Last message from graph
    return {"response": summary}
