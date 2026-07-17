# Deploying AgenticOS to AWS

This deploys the FastAPI service (`src/agenticos/api/app.py`) — the
container built from the repo root `Dockerfile` — to ECS Fargate behind an
ALB. The Streamlit UI can run as a second service from the same image (see
`docker-compose.yml` for the command override) if you want an interactive
front end in addition to the API.

## Architecture

```
Internet -> ALB -> ECS Fargate (agenticos-api, N tasks)
                        |
                        ├─> EFS (SQLite DB + Chroma vector store — shared, persistent)
                        └─> Ollama backend (see "LLM backend" below)
```

## 1. Build and push the image

```bash
aws ecr create-repository --repository-name agenticos
aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com

docker build -t agenticos:latest .
docker tag agenticos:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/agenticos:latest
docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/agenticos:latest
```

## 2. Persistent storage (EFS)

The SQLite database and Chroma vector store need to survive task restarts
and (if you scale beyond one task) be visible to every task. An EFS file
system mounted into the container at `/mnt/agenticos-data` covers both —
`task-definition.json` already wires this up via `efsVolumeConfiguration`.

```bash
aws efs create-file-system --tags Key=Name,Value=agenticos-data
aws efs create-access-point --file-system-id <FS_ID> \
  --posix-user Uid=1000,Gid=1000 \
  --root-directory Path=/agenticos,CreationInfo="{OwnerUid=1000,OwnerGid=1000,Permissions=755}"
```

**Concurrency note:** SQLite is not a great fit for multiple writers across
NFS. Keep `desiredCount: 1` for the API service (fine for this workload —
it's I/O-bound on Ollama, not CPU-bound), or migrate `db/connection.py` and
`memory/store.py` to RDS/Aurora Postgres if you need to scale horizontally.
Read-heavy scaling can still work by pointing extra tasks at a read replica.

## 3. LLM backend (Ollama)

AgenticOS talks to Ollama over HTTP (`AGENTICOS_OLLAMA_BASE_URL`); it doesn't
bundle Ollama itself. Two reasonable options on AWS:

- **Self-hosted Ollama on ECS/EC2**: run `ollama/ollama` as its own service
  (GPU instance recommended — `g4dn.xlarge` or similar — for acceptable
  latency), expose it on an internal ALB or Cloud Map service-discovery
  name, and set `AGENTICOS_OLLAMA_BASE_URL` to that internal endpoint.
- **Swap to a hosted API**: `llm/client.py`'s `build_chat_model` is the one
  place the model is constructed — pointing it at Amazon Bedrock or another
  hosted provider (via `langchain-aws` or an OpenAI-compatible endpoint) is
  a change to that one function, not a codebase-wide rewrite.

Store `AGENTICOS_OLLAMA_BASE_URL` in SSM Parameter Store (or Secrets
Manager) rather than a plaintext task-definition environment variable if
it points at something you'd rather not leave in CloudTrail/console
history — `task-definition.json` already references it as a `secrets`
entry.

## 4. IAM roles

- **Execution role** (`executionRoleArn`): pulls the ECR image, writes to
  CloudWatch Logs, reads the SSM parameter — attach
  `AmazonECSTaskExecutionRolePolicy` plus an inline policy for
  `ssm:GetParameters` on the specific parameter ARN.
- **Task role** (`taskRoleArn`): what the running container can do at
  runtime — for the current codebase, EFS client access
  (`elasticfilesystem:ClientMount`, `ClientWrite`) is the main thing
  needed beyond default networking.

## 5. Deploy

```bash
aws logs create-log-group --log-group-name /ecs/agenticos-api

# Fill in <ACCOUNT_ID> / <REGION> / <EFS_*> / <ATTRIBUTE> placeholders in
# task-definition.json first.
aws ecs register-task-definition --cli-input-json file://deploy/aws/task-definition.json

aws ecs create-service \
  --cluster <CLUSTER_NAME> \
  --service-name agenticos-api \
  --task-definition agenticos-api \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<SUBNET_IDS>],securityGroups=[<SG_ID>],assignPublicIp=DISABLED}" \
  --load-balancers "targetGroupArn=<TARGET_GROUP_ARN>,containerName=agenticos-api,containerPort=8000"
```

Point the ALB target group's health check at `GET /ready` (not `/health` —
see the docstrings in `api/app.py` for why they're split: `/health` is a
liveness probe that never touches Ollama/EFS, `/ready` actually confirms
the graph/model came up cleanly at startup).

## 6. Simpler alternative: AWS App Runner

For a lower-ops path than ECS+ALB+EFS, App Runner can run the same image
directly from ECR with an autoscaling HTTP service and a built-in health
check, at the cost of no EFS support — you'd need to point
`AGENTICOS_DB_PATH`/`AGENTICOS_VECTOR_DB_DIR` at S3-backed storage (e.g. via
a sync sidecar) or an RDS/Postgres-backed store instead of local SQLite/Chroma.
Reasonable for a demo or low-traffic internal tool; ECS+EFS is the more
direct fit for the SQLite-based design as shipped.
