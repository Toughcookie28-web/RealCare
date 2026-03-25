from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from api.routes.chat import router as chat_router
from api.routes.health import router as health_router
from api.routes.history import router as history_router
from core.settings import get_settings
from db.session import init_database
from observability.logging import configure_logging
from observability.middleware import ObservabilityMiddleware
from observability.tracing import configure_tracing


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_database()
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    configure_logging()
    if settings.otel_enabled:
        configure_tracing(service_name='medigenius')

    app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)
    app.add_middleware(ObservabilityMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )
    app.add_middleware(SessionMiddleware, secret_key=settings.resolved_session_secret())

    app.mount('/static', StaticFiles(directory='static'), name='static')
    templates = Jinja2Templates(directory='templates')

    app.include_router(chat_router)
    app.include_router(history_router)
    app.include_router(health_router)

    @app.get('/', response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse(request, 'index.html')

    return app


app = create_app()


if __name__ == '__main__':
    import uvicorn

    cfg = get_settings()
    uvicorn.run('app:app', host=cfg.api_host, port=cfg.api_port, reload=cfg.debug)
