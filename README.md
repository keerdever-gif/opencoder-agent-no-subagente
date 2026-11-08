# OpenCoder-Agent (Web, sin subagentes)

<div align="center"><img src="imagenes/opencoder.gif" width="100%" alt="OpenCoder-Agent"></div>

Agente de programación Python con interfaz web. Esta edición se usa desde el navegador y no requiere Discord, Telegram, WhatsApp ni Railway. No incluye subagentes.

## Instalación
```bash
git clone https://github.com/keerdever-gif/opencoder-agent-no-subagente.git
cd opencoder-agent-no-subagente
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```
Configura `AI_API_KEY` y un secreto largo en `WEB_ACCESS_TOKEN`.

## Ejecutar
```bash
python main.py
```
Abre http://localhost:8080 e introduce el token. Docker: `docker compose up --build`.

## Variables
- `AI_API_KEY` / `AI_API_KEY_1`: proveedor LLM.
- `WEB_ACCESS_TOKEN`: protege el acceso web.
- `WEB_HOST` y `PORT`: interfaz y puerto.
- `OPENCODER_WORKSPACE`: directorio de trabajo.
- `DATABASE_URL`: PostgreSQL opcional.

## Módulos
`web/`, `web_app.py`, `loop_agent/`, `sandbox/`, `search/`, `tools/`, `tests/`.

Licencia Apache-2.0.
