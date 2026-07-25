"""HTTP routers, one per bounded context.

Each router maps 1:1 to a backend service in the target microservice topology,
so splitting the gateway apart later is a routing change rather than a rewrite.
"""

from api_gateway.routers import (
    admin,
    agents,
    auth,
    design,
    knowledge,
    memories,
    runs,
    runtime,
    skills,
    workflows,
)

ALL_ROUTERS = (
    auth.router,
    workflows.router,
    design.router,
    agents.router,
    skills.router,
    memories.router,
    knowledge.router,
    runs.router,
    runtime.router,
    admin.router,
)

__all__ = [
    "ALL_ROUTERS",
    "admin",
    "agents",
    "auth",
    "design",
    "knowledge",
    "memories",
    "runs",
    "runtime",
    "skills",
    "workflows",
]
