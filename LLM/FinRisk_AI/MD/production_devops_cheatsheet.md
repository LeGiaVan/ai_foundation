# 🚀 CẨM NANG PRODUCTION DEVOPS CHO AI ENGINEER
## Từ Laptop đến VPS Thực tế: Hạ tầng, CI/CD, Bảo mật & Vận hành

---

## 1. HẠ TẦNG & TRIỂN KHAI

---

### 1.1. VPS vs PaaS — Chọn gì cho AI System?

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│        PaaS (Render, Railway,         │    VPS (DigitalOcean, Vultr, Akamai) │
│         Heroku, Google Cloud Run)     │         Ubuntu 22.04 raw server      │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ ✅ Deploy bằng 1 lệnh git push        │ ✅ Full control mọi thứ               │
│ ✅ Scale tự động                       │ ✅ Dữ liệu BCTC ở trên máy mình      │
│ ✅ Không cần quản lý server           │ ✅ Rẻ hơn 3-5x khi load ổn định       │
│ ❌ Không có persistent disk           │ ✅ Self-host Qdrant, Langfuse, Redis  │
│ ❌ Cold start (container ngủ đông)    │ ❌ Phải tự cài Nginx, SSL             │
│ ❌ Đắt khi có nhiều service nặng      │ ❌ Phải tự vá lỗi bảo mật hệ điều hành│
│ ❌ KHÔNG ĐƯỢC dùng cho Ngân hàng      │ ✅ BẮT BUỘC với data tài chính nhạy  │
│    (data bay ra cloud bên thứ 3)      │   cảm (on-premise requirement)       │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

**→ FinRisk AI dùng VPS** vì: Qdrant + PostgreSQL + Langfuse cần persistent disk, data tài chính không được rời khỏi hạ tầng công ty.

**Cấu hình VPS tối thiểu cho FinRisk AI:**
```
CPU:  4 vCPU
RAM:  8 GB (Qdrant ~2GB, Langfuse ~1GB, FastAPI ~1GB, còn lại cho hệ điều hành)
Disk: 50 GB SSD (Vector DB collection, PDF DocStore, PostgreSQL, ClickHouse logs)
OS:   Ubuntu 22.04 LTS (Long Term Support — vá lỗi đến năm 2027)
```

---

### 1.2. Reverse Proxy & Nginx

#### Vấn đề không có Reverse Proxy:
Nếu bạn expose thẳng FastAPI (chạy ở cổng `:8000`) ra internet, người dùng phải vào `http://45.123.456.78:8000`. Điều này:
* Không có HTTPS → browser cảnh báo "Not Secure"
* Không thể chạy nhiều service trên cùng 1 VPS (FastAPI:8000, Langfuse:3000, Grafana:3001)
* Không có bộ lọc bảo mật ở tầng đầu vào

#### Nginx là gì?
**Nginx** là một Web Server / Reverse Proxy đứng phía trước tất cả các service của bạn. Nó nhận request từ internet (port 80/443), xử lý SSL, rồi chuyển tiếp (proxy pass) vào đúng service.

```
Internet
  │
  ▼
[Nginx :443 HTTPS]  ← Làm SSL Termination, Rate Limit, CORS tại đây
  │
  ├─ /api/*   → FastAPI :8000
  ├─ /traces/* → Langfuse :3000
  └─ /grafana/ → Grafana :3001
```

#### File cấu hình Nginx cơ bản (`/etc/nginx/sites-available/finriskai`):
```nginx
server {
    listen 80;
    server_name finriskai.yourdomain.com;

    # Redirect toàn bộ HTTP → HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name finriskai.yourdomain.com;

    # SSL Certificate từ Let's Encrypt
    ssl_certificate     /etc/letsencrypt/live/finriskai.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/finriskai.yourdomain.com/privkey.pem;

    # Rate Limiting: Tối đa 10 request/giây từ 1 IP
    limit_req zone=api_limit burst=20 nodelay;

    # Proxy toàn bộ traffic vào FastAPI
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;    # Chờ LLM generate tối đa 2 phút
    }

    # File upload size limit (chặn PDF > 20MB)
    client_max_body_size 20M;
}
```

---

### 1.3. SSL/TLS & Let's Encrypt

* **SSL/TLS:** Giao thức mã hóa lớp truyền tải. `HTTPS = HTTP + SSL/TLS`. Toàn bộ traffic giữa trình duyệt và server được mã hóa — không ai nghe lén được nội dung BCTC tài chính của khách hàng.
* **Let's Encrypt:** Tổ chức phi lợi nhuận cấp **SSL Certificate miễn phí** cho mọi domain. Certificate có hiệu lực 90 ngày và **tự động gia hạn** qua Certbot.

```bash
# Cài Certbot
sudo apt install certbot python3-certbot-nginx

# Cấp certificate cho domain (Certbot tự động sửa Nginx config)
sudo certbot --nginx -d finriskai.yourdomain.com

# Kiểm tra cron job tự gia hạn mỗi 2 tháng
sudo systemctl list-timers | grep certbot
```

---

### 1.4. Domain & DNS Record

* **Domain:** Tên dễ nhớ (`finriskai.com`) thay cho địa chỉ IP số (`45.67.89.12`). Mua tại Namecheap, GoDaddy, hoặc Cloudflare.
* **DNS Record:** Bảng tra cứu chuyển Domain → IP. Cấu hình tại nhà cung cấp domain:

| Loại Record | Ví dụ | Mục đích |
| :--- | :--- | :--- |
| **A Record** | `finriskai.com → 45.67.89.12` | Trỏ domain chính vào VPS |
| **CNAME** | `www.finriskai.com → finriskai.com` | Alias, trỏ `www` về domain gốc |
| **TXT** | Dùng xác minh Let's Encrypt, Google Search Console | Xác thực quyền sở hữu domain |

---

### 1.5. Docker Compose Restart Policy & Health Check

```yaml
# docker-compose.yml
version: "3.9"

services:
  api:
    build: .
    restart: unless-stopped     # Tự khởi động lại nếu crash (trừ khi bạn tự `docker stop`)
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s             # Kiểm tra mỗi 30 giây
      timeout: 10s              # Timeout sau 10 giây nếu không phản hồi
      retries: 3                # Sau 3 lần thất bại → đánh dấu service là "unhealthy"
      start_period: 60s         # Chờ 60 giây sau khi start (để warmup model)
    ports:
      - "8000:8000"

  qdrant:
    image: qdrant/qdrant:v1.9.0
    restart: unless-stopped     # CRITICAL: Không có cái này, VPS reboot là mất Vector DB!
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru

volumes:
  qdrant_data:
```

**Health Check Endpoint trong FastAPI:**
```python
@app.get("/health")
async def health_check():
    # Kiểm tra kết nối đến tất cả dependency
    try:
        await qdrant_client.get_collections()
        await redis_client.ping()
        return {
            "status": "ok",
            "qdrant": "connected",
            "redis": "connected",
            "langfuse": "reachable",
            "uptime_seconds": time.time() - START_TIME
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service degraded: {str(e)}")
```

---

## 2. CẤU HÌNH & BẢO MẬT VẬN HÀNH

---

### 2.1. Phân tách Môi trường: dev / staging / production

**Vì sao phải tách:**
* **dev:** Thử nghiệm tính năng mới, có thể dùng model rẻ, log verbose, không cần SSL
* **staging:** Giống production 100% nhưng không có dữ liệu thật — nơi QA team kiểm thử trước khi release
* **production:** Dữ liệu thật của khách hàng, cấu hình strict, monitoring đầy đủ

```bash
# Cấu trúc file .env theo môi trường
.env.dev          # Temperature=0.7, LOG_LEVEL=DEBUG, MODEL=gpt-4o-mini
.env.staging      # Temperature=0.0, LOG_LEVEL=INFO, MODEL=claude-3-5-haiku
.env.production   # Temperature=0.0, LOG_LEVEL=WARNING, MODEL=claude-3-5-haiku

# Chạy với môi trường cụ thể
docker compose --env-file .env.production up -d
```

```bash
# .env.production (KHÔNG BAO GIỜ COMMIT FILE NÀY LÊN GIT!)
OPENAI_API_KEY=sk-xxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx
DATABASE_URL=postgresql://finrisk:secret@postgres:5432/finrisk_prod
REDIS_URL=redis://redis:6379/0
ENVIRONMENT=production
LOG_LEVEL=WARNING
MAX_UPLOAD_SIZE_MB=20
```

---

### 2.2. Secrets Management

**Nguyên tắc vàng: API Key không bao giờ được nằm trong code Git!**

```
# .gitignore — LUÔN LUÔN có những dòng này
.env
.env.*
*.pem
secrets/
```

**Quy trình quản lý Secret:**

```
[Bạn nhập secret vào GitHub Settings → Secrets → Actions]
                    │
                    ▼
[GitHub Actions CI/CD đọc secret → inject vào quá trình build]
                    │
                    ▼
[Deploy lên VPS → Ghi vào file /opt/finriskai/.env trên server]
                    │
                    ▼
[Docker Compose đọc .env → inject vào container qua environment]
```

---

### 2.3. API Key Rotation (Đổi key không gây downtime)

Khi phát hiện key bị lộ hoặc đổi định kỳ mỗi 3 tháng:
1. Tạo key mới trên OpenAI/Anthropic Dashboard.
2. Cập nhật trong GitHub Secrets (để CI/CD dùng).
3. SSH vào VPS → chỉnh sửa `/opt/finriskai/.env`.
4. Chạy `docker compose up -d api` (chỉ restart service `api`, không ảnh hưởng Qdrant/DB).
5. Vô hiệu hóa key cũ sau 5 phút để đảm bảo không có request đang dùng key cũ.

---

### 2.4. CORS, Rate Limiting & Input Size Limit

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address

app = FastAPI()

# CORS: Chỉ cho phép frontend chính thức gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://finriskai.yourdomain.com"],  # Không dùng "*" trong production!
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Rate Limiting: Giới hạn 10 request/phút cho mỗi IP
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/v1/analyze")
@limiter.limit("10/minute")
async def analyze_report(request: Request, ...):
    ...
```

**Input Size Limit (Chặn OOM & Injection):**
```python
from fastapi import UploadFile, HTTPException

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

async def validate_upload(file: UploadFile):
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "File quá lớn. Tối đa 20MB.")
    if not file.filename.endswith((".pdf", ".docx")):
        raise HTTPException(400, "Chỉ chấp nhận file PDF và DOCX.")
    return content
```

---

## 3. CHI PHÍ & HIỆU NĂNG LLM TRONG PRODUCTION

---

### 3.1. Token Usage Tracking theo User

```python
from langfuse.decorators import observe, langfuse_context

@observe(name="generate_credit_report")
async def generate_report(user_id: str, question: str):
    # Gắn user_id vào Trace để Langfuse phân tích chi phí theo từng người dùng
    langfuse_context.update_current_trace(
        user_id=user_id,
        tags=["credit-report", "production"]
    )
    # LLM call...
    response = await llm.ainvoke(messages)
    # Langfuse tự động tính tiền từ token usage
    return response.content
```

---

### 3.2. Budget Alert — Cảnh báo khi vượt ngân sách

**Trên Langfuse Dashboard:** Vào Settings → Alerts → Tạo alert:
* Điều kiện: `Total Cost (USD) per user_id > 5.00 in last 24h`
* Hành động: Gửi Webhook → Slack Channel → Bật cờ `user_locked=True` trong Redis

---

### 3.3. Response Caching với Redis (Giảm 60-80% gọi API trùng lặp)

```python
import hashlib, json
import redis.asyncio as redis

r = redis.from_url("redis://redis:6379")
CACHE_TTL = 86400  # 24 giờ

async def cached_analyze(user_query: str, context_chunks: list) -> str:
    # Tạo cache key duy nhất từ query + context
    cache_key = f"llm:{hashlib.sha256((user_query + str(context_chunks)).encode()).hexdigest()}"
    
    # Thử lấy từ cache trước
    cached = await r.get(cache_key)
    if cached:
        return json.loads(cached)["answer"]
    
    # Cache miss → gọi LLM thật
    answer = await llm.ainvoke(f"Context: {context_chunks}\nQuestion: {user_query}")
    
    # Lưu vào cache 24 giờ
    await r.setex(cache_key, CACHE_TTL, json.dumps({"answer": answer.content}))
    return answer.content
```

---

### 3.4. Streaming Response (Giảm Perceived Latency)

Không cần chờ LLM generate xong 500 chữ (5 giây) mới thấy kết quả. SSE Stream hiển thị từng token ngay khi sinh ra:

```python
from fastapi.responses import StreamingResponse

@app.post("/api/v1/analyze/stream")
async def analyze_stream(request: AnalysisRequest):
    async def generate():
        async for chunk in llm.astream(request.question):
            yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

### 3.5. Timeout & Retry Policy cho LLM Call

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),           # Thử lại tối đa 3 lần
    wait=wait_exponential(min=1, max=30), # Chờ 1s, 2s, 4s... giữa các lần thử
    reraise=True
)
async def call_llm_with_retry(prompt: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await llm.ainvoke(prompt)
            return response.content
    except httpx.TimeoutException:
        raise ValueError("LLM timeout sau 60 giây. Thử lại sau.")
```

---

## 4. CI/CD VỚI GITHUB ACTIONS

---

### 4.1. Tổng quan Pipeline CI/CD cho FinRisk AI

```
Git push to main
     │
     ▼
[Job 1: Test & Lint]
  • pytest tests/unit/
  • ruff check (code style)
     │ ✅ Pass
     ▼
[Job 2: Ragas Quality Gate]
  • Chạy 50 câu Golden Testset
  • Faithfulness ≥ 0.85?
  • Context Recall ≥ 0.80?
     │ ✅ Pass (❌ FAIL → block deploy!)
     ▼
[Job 3: Build Docker Image]
  • docker build -t finriskai:$GIT_SHA .
  • docker push ghcr.io/youruser/finriskai:$GIT_SHA
     │
     ▼
[Job 4: Deploy to VPS via SSH]
  • SSH vào VPS
  • docker compose pull api
  • docker compose up -d --no-deps --build api
  • Kiểm tra /health trong 60 giây
     │ ✅ Healthy → Done
     │ ❌ Unhealthy → Auto rollback!
```

---

### 4.2. File `.github/workflows/deploy.yml` hoàn chỉnh

```yaml
name: CI/CD Deploy FinRisk AI

on:
  push:
    branches: [main]

env:
  IMAGE_NAME: ghcr.io/${{ github.repository_owner }}/finriskai

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/unit/ -v

  eval-gate:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Ragas evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          QDRANT_URL: ${{ secrets.QDRANT_STAGING_URL }}
        run: |
          pip install ragas pytest
          pytest tests/eval/test_ragas_metrics.py -v --tb=short
      # Nếu bước này fail → pipeline dừng, không deploy!

  build-and-push:
    needs: eval-gate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build & push Docker image
        run: |
          echo "${{ secrets.GITHUB_TOKEN }}" | docker login ghcr.io -u ${{ github.actor }} --password-stdin
          docker build -t $IMAGE_NAME:${{ github.sha }} .
          docker push $IMAGE_NAME:${{ github.sha }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - name: Deploy via SSH (Zero-Downtime)
        uses: appleboy/ssh-action@v1.0.0
        with:
          host: ${{ secrets.VPS_HOST }}
          username: ${{ secrets.VPS_USER }}
          key: ${{ secrets.VPS_SSH_KEY }}
          script: |
            cd /opt/finriskai
            
            # Lưu lại version cũ để rollback nếu cần
            echo "${{ github.sha }}" > .current_version
            
            # Pull image mới và restart chỉ service api (Zero-Downtime)
            export IMAGE_TAG=${{ github.sha }}
            docker compose pull api
            docker compose up -d --no-deps api
            
            # Chờ health check xanh trong 60 giây
            for i in $(seq 1 12); do
              sleep 5
              STATUS=$(curl -sf http://localhost:8000/health | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null)
              if [ "$STATUS" = "ok" ]; then
                echo "✅ Deploy thành công!"
                exit 0
              fi
            done
            
            # Nếu sau 60 giây vẫn unhealthy → Rollback tự động!
            echo "❌ Health check thất bại! Đang rollback..."
            PREV_SHA=$(cat .prev_version 2>/dev/null || echo "latest")
            export IMAGE_TAG=$PREV_SHA
            docker compose up -d --no-deps api
            exit 1
```

---

## 5. GIÁM SÁT & ĐỘ TIN CẬY

---

### 5.1. Structured Logging (JSON Logs)

Log dạng text thô khó tìm kiếm và filter. Log dạng JSON có thể đẩy vào Grafana/Loki và query như database:

```python
import logging, json, time
from pythonjsonlogger import jsonlogger

# Cấu hình JSON logger
handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter(
    "%(asctime)s %(levelname)s %(name)s %(message)s"
))
logger = logging.getLogger("finriskai")
logger.addHandler(handler)

# Mọi log đều dưới dạng JSON
logger.info("LLM request completed", extra={
    "user_id": "analyst_007",
    "request_id": "req_abc123",
    "latency_ms": 1240,
    "input_tokens": 450,
    "output_tokens": 120,
    "model": "claude-3-5-haiku",
    "cost_usd": 0.0034
})
```

Output: `{"timestamp": "2024-09-18 14:00:00", "level": "INFO", "message": "LLM request completed", "user_id": "analyst_007", "latency_ms": 1240, ...}`

---

### 5.2. Uptime Monitoring

**Công cụ:** UptimeRobot (miễn phí đến 50 monitors) hoặc Better Uptime.

* Ping endpoint `GET /health` mỗi **1 phút**
* Khi `/health` trả về non-200 lần đầu tiên → gửi email/SMS cảnh báo ngay lập tức
* Dashboard public: `https://status.finriskai.yourdomain.com`

---

### 5.3. Backup Strategy

```bash
# 1. Backup Qdrant Collection Snapshot (chạy hàng ngày qua cron)
curl -X POST "http://localhost:6333/collections/financial_docs/snapshots"
# File snapshot được lưu tại /var/lib/qdrant/snapshots/

# 2. Upload snapshot lên Cloudflare R2 (S3-compatible, rẻ)
aws s3 cp /var/lib/qdrant/snapshots/ s3://finriskai-backup/ \
  --recursive --endpoint-url https://<account_id>.r2.cloudflarestorage.com

# 3. Backup PostgreSQL
pg_dump -U finrisk finrisk_prod | gzip > /backup/postgres_$(date +%Y%m%d).sql.gz

# Thêm vào crontab (chạy 2am mỗi ngày)
# 0 2 * * * /opt/finriskai/scripts/backup.sh
```

---

### 5.4. Rollback Strategy

```bash
# Scenario: Deploy mới bị lỗi, cần quay lại phiên bản cũ ngay lập tức

# Cách 1: Rollback Docker Image theo Git commit hash
cd /opt/finriskai
export IMAGE_TAG=<git-sha-cũ>
docker compose up -d --no-deps api  # Chỉ restart service api, giữ nguyên DB/Redis

# Cách 2: Rollback database migration (nếu có thay đổi schema)
alembic downgrade -1

# Cách 3: Rollback Prompt version trên Langfuse
# → Vào Langfuse Web UI → Prompts → Chọn version cũ → Set label "production"
# → Không cần restart bất kỳ service nào!
```

---

## 6. ASYNC & TÁC VỤ NỀN (BACKGROUND TASKS)

---

### 6.1. FastAPI BackgroundTasks (Cho việc đơn giản)

Phù hợp cho tác vụ nhanh (<30 giây), không cần theo dõi tiến trình chi tiết:

```python
from fastapi import BackgroundTasks

async def index_document_background(file_content: bytes, collection: str):
    """Chạy ngầm sau khi API đã trả response 202 cho người dùng."""
    chunks = parse_pdf(file_content)
    await qdrant_client.upsert(collection, chunks)
    logger.info(f"Indexed {len(chunks)} chunks into {collection}")

@app.post("/api/v1/documents/upload", status_code=202)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    content = await file.read()
    # Trả về 202 Accepted NGAY LẬP TỨC (không chờ xử lý xong)
    background_tasks.add_task(index_document_background, content, "financial_docs")
    return {"message": "Tài liệu đang được xử lý. Sẽ sẵn sàng trong 1-2 phút."}
```

---

### 6.2. Celery + Redis (Cho việc phức tạp, chạy lâu)

Phù hợp cho tác vụ >30 giây, cần retry khi lỗi, cần theo dõi tiến trình (progress bar):

```python
# tasks.py
from celery import Celery
import redis

celery_app = Celery("finriskai", broker="redis://redis:6379/0", backend="redis://redis:6379/1")

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def index_large_document(self, document_id: str, s3_url: str):
    """Task chạy trên Celery Worker riêng biệt."""
    try:
        content = download_from_s3(s3_url)       # Download từ S3
        chunks = parse_pdf_with_tables(content)   # OCR + bảng biểu (5-10 phút)
        embed_and_store(chunks, document_id)      # Embed và lưu vào Qdrant
        notify_user(document_id, status="completed")
    except Exception as exc:
        # Tự động retry sau 60 giây nếu lỗi
        raise self.retry(exc=exc)

# Gọi task từ FastAPI
@app.post("/api/v1/documents/upload-large")
async def upload_large_document(file: UploadFile):
    s3_url = await upload_to_s3(file)
    task = index_large_document.delay(document_id, s3_url)
    return {"task_id": task.id, "status": "queued"}

# Kiểm tra tiến trình
@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    result = celery_app.AsyncResult(task_id)
    return {"task_id": task_id, "status": result.status}
```

**Khi nào chọn BackgroundTasks vs Celery:**

| Tiêu chí | FastAPI BackgroundTasks | Celery + Redis |
| :--- | :--- | :--- |
| Thời gian xử lý | < 30 giây | > 30 giây |
| Retry khi lỗi | Không có | Tự động retry |
| Theo dõi tiến trình | Không | Có (task.status, task.result) |
| Chạy khi server restart | Mất hết queue | Vẫn còn trong Redis |
| Độ phức tạp setup | Rất đơn giản | Cần Celery Worker service riêng |
| **Khi nào dùng** | Gửi email, webhook, log đơn giản | Index PDF lớn, xử lý batch, job định kỳ |

---

## 7. CHI PHÍ THỰC TẾ ĐỂ XÂY FINRISK AI

---

### 7.1. Tổng hợp: Cái gì phải trả tiền, cái gì miễn phí?

#### 🔴 BẮT BUỘC trả tiền

| Hạng mục | Dịch vụ gợi ý | Chi phí ước tính |
| :--- | :--- | :--- |
| **VPS** — server chạy toàn bộ hệ thống | DigitalOcean / Vultr / Akamai | ~$24–40/tháng (4vCPU, 8GB RAM) |
| **Domain** — địa chỉ web của bạn | Namecheap / Cloudflare | ~$10–15/năm |
| **LLM API** — gọi OpenAI / Anthropic | Claude Haiku hoặc GPT-4o-mini | ~$5–20/tháng (tùy lượng dùng) |

> **Tổng ước tính khi chạy production: ~$35–60/tháng** (~900k–1.5tr VND/tháng)

#### 🟢 MIỄN PHÍ hoàn toàn

| Hạng mục | Lý do miễn phí |
| :--- | :--- |
| **SSL/TLS** (HTTPS) | Let's Encrypt cấp miễn phí, tự gia hạn |
| **Nginx** | Open source |
| **Docker / Docker Compose** | Open source |
| **GitHub Actions** | Miễn phí với repo public; 2000 phút/tháng với repo private |
| **Cloudflare DNS** | Free tier đủ dùng cho cá nhân / startup |
| **Langfuse** (Self-hosted) | Chạy trên VPS của mình → $0 |
| **Qdrant** (Self-hosted) | Chạy trên VPS của mình → $0 |
| **Redis, PostgreSQL** | Open source, chạy trên VPS |
| **Ragas / DeepEval** | Open source Python library |
| **BGE-M3 Embedding** | Tải về một lần, chạy local trên VPS |
| **UptimeRobot** | Free tier: 50 monitors, check mỗi 5 phút |

---

### 7.2. Chiến lược chi phí theo từng giai đoạn học

```
Giai đoạn 1–3 (Tuần 1–6): HỌC TRÊN MÁY LOCAL
────────────────────────────────────────────
  VPS:     $0  → Chạy toàn bộ bằng docker compose up trên laptop
  Domain:  $0  → Dùng localhost hoặc Ngrok (miễn phí, tạm expose ra internet)
  LLM API: ~$2–5 total → Dùng tiết kiệm trong quá trình học
  ─────────────────────────────────────────
  Chi phí: ~$2–5 trong cả 6 tuần

Giai đoạn 4–5 (Tuần 7–12): DEPLOY THẬT LÊN VPS
────────────────────────────────────────────
  VPS:     ~$24/tháng → Thuê theo tháng, không cam kết năm
  Domain:  ~$1/tháng  → Chia đều $12/năm
  LLM API: ~$5–10/tháng → Có Redis Cache nên giảm ~70% gọi LLM
  ─────────────────────────────────────────
  Chi phí: ~$30–35/tháng (~750k–900k VND)
```

---

### 7.3. Mẹo tiết kiệm chi phí LLM

| Kỹ thuật | Tiết kiệm ước tính | Cách làm |
| :--- | :--- | :--- |
| **Redis Response Cache** | 60–80% gọi API | Hash(query + context) → cache 24h |
| **Claude Haiku thay GPT-4o** | 5–10x rẻ hơn, chất lượng tương đương | `model="claude-3-5-haiku-20241022"` |
| **Anthropic Prompt Caching** | Giảm 80% chi phí System Prompt dài | Cache prefix dài hơn 1024 tokens |
| **Context Compression (LLMLingua)** | Cắt 30–50% token thừa trong context | Dùng trước khi gửi context vào LLM |
| **Token Budget per User** | Tránh bị một user "ăn hết" quota | Langfuse alert + Redis counter |

