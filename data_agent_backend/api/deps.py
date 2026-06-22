from __future__ import annotations

from fastapi import Request

from data_agent_backend.services.factory import BackendServices, create_backend_services


def get_backend_services(request: Request) -> BackendServices:
    services = getattr(request.app.state, "services", None)
    if services is None:
        services = create_backend_services()
        request.app.state.services = services
    return services

