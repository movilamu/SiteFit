import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

# ---------------------------------------------------------------------------
# Sentry initialization
# Read SENTRY_DSN from environment variables.
# Demographic data privacy: send_default_pii=False ensures no PII is captured.
# ---------------------------------------------------------------------------
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[
            StarletteIntegration(transaction_style="endpoint"),
            FastApiIntegration(transaction_style="endpoint"),
        ],
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0")),
        send_default_pii=False,  # Explicitly disabled for demographic data privacy
    )


# ---------------------------------------------------------------------------
# Router registration — File_XX fallbacks support this loose-file workspace.
# ---------------------------------------------------------------------------
def _load_router(module_names: list[str]):
    last_error: Exception | None = None
    for name in module_names:
        try:
            module = __import__(name, fromlist=["router"])
            return getattr(module, "router")
        except Exception as exc:  # ImportError or missing deps in a given layout
            last_error = exc
    raise ImportError(f"Could not import router from {module_names}: {last_error}")


sites_routes = _load_router(["routers.sites", "api.sites_routes", "File_43"])
score_routes = _load_router(["routers.scores", "api.score_routes", "File_44"])
decomposition_routes = _load_router(["routers.decomposition", "api.decomposition_routes", "File_45"])
scenario_routes = _load_router(["routers.scenario", "api.scenario_routes", "File_46"])
export_routes = _load_router(["routers.export", "api.export_routes", "File_47"])

try:
    elicitation_routes = _load_router(["routers.elicitation", "api.elicitation_routes", "File_37"])
except ImportError:
    elicitation_routes = None

try:
    sensitivity_routes = _load_router(["routers.sensitivity", "api.sensitivity_routes", "File_41"])
except ImportError:
    sensitivity_routes = None

app = FastAPI(
    title="Retail Site Platform API",
    version="0.1.0",
)


@app.middleware("http")
async def sentry_http_middleware(request: Request, call_next):
    """Capture failed HTTP responses and unhandled exceptions to Sentry."""
    try:
        response = await call_next(request)
        if response.status_code >= 500 and SENTRY_DSN:
            sentry_sdk.capture_message(
                f"HTTP {response.status_code} on {request.method} {request.url.path}",
                level="error",
            )
        return response
    except Exception as exc:
        if SENTRY_DSN:
            sentry_sdk.capture_exception(exc)
        raise exc


# CORS: Next.js on localhost during development.
# Replace the Vercel placeholder with the production frontend URL at deploy time.
_DEV_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_PRODUCTION_ORIGIN = "https://YOUR-VERCEL-APP.vercel.app"  # fill in at deployment

app.add_middleware(
    CORSMiddleware,
    allow_origins=[*_DEV_ORIGINS, _PRODUCTION_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sites_routes)           # File 43
app.include_router(score_routes)           # File 44
app.include_router(decomposition_routes)   # File 45
app.include_router(scenario_routes)        # File 46
app.include_router(export_routes)          # File 47
if elicitation_routes is not None:
    app.include_router(elicitation_routes)  # File 37
if sensitivity_routes is not None:
    app.include_router(sensitivity_routes)  # File 41


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
