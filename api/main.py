from fastapi import FastAPI, Request, Form, Cookie
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from uuid import uuid4
from agents.runner import run_career_conversation_step

app = FastAPI()
app.mount("/static", StaticFiles(directory="api/static"), name="static")

templates = Jinja2Templates(directory="templates")

THREAD_COOKIE_NAME = "thread_id"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    thread_id = request.cookies.get(THREAD_COOKIE_NAME)
    if not thread_id:
        thread_id = str(uuid4())
    response = templates.TemplateResponse("chat.html", {"request": request, "thread_id": thread_id})
    response.set_cookie(key=THREAD_COOKIE_NAME, value=thread_id)
    return response

@app.post("/chat", response_class=JSONResponse)
async def chat(
    request: Request,
    message: str = Form(...),
    thread_id: str = Cookie(None)
):
    bot_reply = run_career_conversation_step(user_input=message, thread_id=thread_id)
    return {"user": message, "bot": bot_reply}
