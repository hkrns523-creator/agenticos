"""HTTP API surface (FastAPI), used for containerized deployment (Docker/AWS)
where a Streamlit UI isn't the right interface — e.g. behind an ALB, called
from another service, or health-checked by ECS/App Runner."""
