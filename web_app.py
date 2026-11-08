from __future__ import annotations
import asyncio,hmac,secrets
from pathlib import Path
from aiohttp import web
from config import config
from loop_agent.agent import agent
from loop_agent.connector import connector
from sandbox.sandbox import sandbox
COOKIE="opencoder_session"
def authorized(request):
    value=request.cookies.get(COOKIE,"")
    return bool(value) and hmac.compare_digest(value,request.app["session"])
async def login(request):
    try: data=await request.json()
    except Exception: return web.json_response({"error":"Solicitud no válida"},status=400)
    if not hmac.compare_digest(str(data.get("token","")),config.web_access_token or ""):
        return web.json_response({"error":"Token incorrecto"},status=401)
    response=web.json_response({"ok":True})
    response.set_cookie(COOKIE,request.app["session"],httponly=True,samesite="Strict",secure=request.secure,max_age=43200)
    return response
async def chat(request):
    if not authorized(request): return web.json_response({"error":"Sesión caducada. Vuelve a entrar."},status=401)
    try: data=await request.json()
    except Exception: return web.json_response({"error":"Solicitud no válida"},status=400)
    prompt=str(data.get("prompt","")).strip()
    if not prompt or len(prompt)>20000: return web.json_response({"error":"Mensaje vacío o demasiado largo."},status=400)
    try:
        answer,image_url,files=await agent.ask(user_id=0,channel_id=0,prompt=prompt)
        return web.json_response({"answer":answer,"image_url":image_url,"files":files or []})
    except Exception as exc:
        return web.json_response({"error":f"{type(exc).__name__}: {exc}"},status=500)
async def start(app): await agent.init()
async def stop(app):
    await connector.close()
    await agent.close()
    await sandbox.close()
async def run():
    app=web.Application(client_max_size=1024*1024);app["session"]=secrets.token_urlsafe(32)
    app.router.add_get("/",lambda r:web.FileResponse("web/index.html"))
    app.router.add_get("/static/app.css",lambda r:web.FileResponse("web/app.css"))
    app.router.add_get("/static/app.js",lambda r:web.FileResponse("web/app.js"))
    app.router.add_get("/static/opencoder.gif",lambda r:web.FileResponse("imagenes/opencoder.gif"))
    app.router.add_post("/api/login",login);app.router.add_post("/api/chat",chat)
    app.on_startup.append(start);app.on_cleanup.append(stop)
    runner=web.AppRunner(app);await runner.setup()
    await web.TCPSite(runner,host=config.web_host,port=config.port).start()
    print(f"OpenCoder listo en http://localhost:{config.port}")
    await asyncio.Event().wait()
