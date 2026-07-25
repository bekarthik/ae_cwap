"""HTTP routers, one per bounded context.

Each router maps 1:1 to a backend service in the target microservice topology,
so splitting the gateway apart later is a routing change rather than a rewrite.
"""

from api_gateway.routers import admin, auth, diagnose, knowledge, runs, runtime, workflows

ALL_ROUTERS = (
    auth.router,
    workflows.router,
    diagnose.router,
    knowledge.router,
    runs.router,
    runtime.router,
    admin.router,
)

__all__ = [
    "ALL_ROUTERS",
    "admin",
    "auth",
    "diagnose",
    "knowledge",
    "runs",
    "runtime",
    "workflows",
]
