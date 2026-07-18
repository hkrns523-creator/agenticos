#!/bin/sh
set -e
python scripts/seed_db.py --if-missing
exec streamlit run src/agenticos/ui/streamlit_app.py --server.port="${PORT:-10000}" --server.address=0.0.0.0