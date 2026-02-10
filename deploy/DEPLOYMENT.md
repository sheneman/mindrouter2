# MindRouter2 Deployment Guide - Rocky Linux 8

## Prerequisites

- Rocky Linux 8 VM with root/sudo access
- Docker and Docker Compose installed
- Apache httpd installed
- Git installed
- SSL certificate (self-signed for testing, or real cert for production)

## Step 1: Install Dependencies (if needed)

```bash
# Install Docker
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER

# Install Apache, Git, and modules
sudo dnf install -y httpd mod_ssl git
sudo systemctl enable httpd
```

## Step 2: Clone Repository

```bash
# Create deployment directory and clone
sudo mkdir -p /opt/mindrouter
sudo chown $USER:$USER /opt/mindrouter
cd /opt/mindrouter

# Clone from GitHub
git clone https://github.com/sheneman/mindrouter2.git .
```

## Step 3: Configure Environment

```bash
# Copy and edit production environment file
cp .env.prod.example .env.prod

# Generate secure passwords and secret key
python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))"
python3 -c "import secrets; print('DB_PASSWORD=' + secrets.token_urlsafe(24))"
python3 -c "import secrets; print('REDIS_PASSWORD=' + secrets.token_urlsafe(24))"

# Edit .env.prod with your values
nano .env.prod
```

**Important: Update these values in .env.prod:**
- `SECRET_KEY` - Generated secret
- `MYSQL_ROOT_PASSWORD` - Secure root password
- `MYSQL_PASSWORD` and in `DATABASE_URL` - Same secure password
- `REDIS_PASSWORD` and in `REDIS_URL` - Same secure password
- `CORS_ORIGINS` - Your actual domain

## Step 4: Configure SSL Certificate

### Option A: Self-signed certificate (testing only)
```bash
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/pki/tls/private/mindrouter.key \
  -out /etc/pki/tls/certs/mindrouter.crt \
  -subj "/CN=mindrouter.example.com"
```

### Option B: Let's Encrypt (production)
```bash
sudo dnf install -y certbot python3-certbot-apache
sudo certbot --apache -d mindrouter.example.com
```

## Step 5: Configure Apache

```bash
# Copy Apache config
sudo cp deploy/apache-mindrouter.conf /etc/httpd/conf.d/mindrouter.conf

# Edit to match your domain
sudo nano /etc/httpd/conf.d/mindrouter.conf
# Update ServerName to your actual domain
# Update SSL certificate paths if using Let's Encrypt

# Enable required modules
sudo dnf install -y mod_proxy_html

# Test Apache config
sudo apachectl configtest

# Restart Apache
sudo systemctl restart httpd
```

## Step 6: Configure Firewall

```bash
# Open HTTP and HTTPS ports
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

## Step 7: Configure SELinux (if enabled)

```bash
# Allow Apache to connect to backend
sudo setsebool -P httpd_can_network_connect 1

# If you have issues with Docker, you may need:
sudo setsebool -P container_manage_cgroup 1
```

## Step 8: Start the Application

```bash
cd /opt/mindrouter

# Build and start containers
docker compose -f docker-compose.prod.yml up -d --build

# Check status
docker compose -f docker-compose.prod.yml ps

# View logs
docker compose -f docker-compose.prod.yml logs -f app
```

## Step 9: Run Database Migrations

```bash
# Run Alembic migrations
docker compose -f docker-compose.prod.yml exec app alembic upgrade head

# Seed initial admin user (optional)
docker compose -f docker-compose.prod.yml exec app python scripts/seed_dev_data.py
```

## Step 10: Deploy GPU Sidecar Agents

Each GPU inference node needs a sidecar agent running to report GPU metrics back to MindRouter2.

### On each GPU server:

```bash
# Option A: Run via Docker (recommended)
# Copy the sidecar directory to the GPU server, or pull the repo
scp -r sidecar/ user@gpu-server:/opt/mindrouter-sidecar/

ssh user@gpu-server
cd /opt/mindrouter-sidecar
docker build -t mindrouter-sidecar -f Dockerfile.sidecar .
docker run -d --name gpu-sidecar \
  --gpus all \
  -p 9101:9101 \
  --restart unless-stopped \
  mindrouter-sidecar

# Verify it's working
curl http://localhost:9101/health
curl http://localhost:9101/gpu-info
```

```bash
# Option B: Run directly (no Docker)
pip install fastapi uvicorn nvidia-ml-py
GPU_AGENT_PORT=9101 python gpu_agent.py
```

### Register the node in MindRouter2:

```bash
# Via API
curl -X POST https://mindrouter.example.com/api/admin/nodes/register \
  -H "Authorization: Bearer admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpu-server-1",
    "hostname": "gpu1.example.com",
    "sidecar_url": "http://gpu1.example.com:9101"
  }'
```

Or use the admin dashboard at `/admin/nodes`.

### Register backends on the node:

```bash
# Backend using all GPUs on the node
curl -X POST https://mindrouter.example.com/api/admin/backends/register \
  -H "Authorization: Bearer admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ollama-gpu1",
    "url": "http://gpu1.example.com:11434",
    "engine": "ollama",
    "max_concurrent": 4,
    "node_id": 1
  }'

# Backend using specific GPUs (for multi-backend nodes)
curl -X POST https://mindrouter.example.com/api/admin/backends/register \
  -H "Authorization: Bearer admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "vllm-gpu1-01",
    "url": "http://gpu1.example.com:8000",
    "engine": "vllm",
    "max_concurrent": 16,
    "node_id": 1,
    "gpu_indices": [0, 1]
  }'
```

**Note:** `gpu_indices` is optional. Omit it to assign all GPUs on the node to the backend. Use it when multiple backends share the same physical server and you want each to report telemetry only for its assigned GPUs.

## Step 11: Verify Deployment

```bash
# Test health endpoint directly
curl http://127.0.0.1:8000/healthz

# Test through Apache (replace with your domain)
curl -k https://mindrouter.example.com/healthz

# Check all services are healthy
docker compose -f docker-compose.prod.yml ps

# Verify sidecar connectivity (from MindRouter2 server)
curl http://gpu1.example.com:9101/health

# Verify node appears in telemetry
curl -H "Authorization: Bearer admin-api-key" \
  https://mindrouter.example.com/api/admin/telemetry/overview
```

**Firewall note:** The MindRouter2 server needs network access to each GPU node's sidecar port (default 9101). Ensure firewall rules allow this traffic between the gateway and GPU nodes.

## Ongoing Operations

### View Logs
```bash
# Application logs
docker compose -f docker-compose.prod.yml logs -f app

# Apache logs
sudo tail -f /var/log/httpd/mindrouter_error.log
sudo tail -f /var/log/httpd/mindrouter_access.log
```

### Restart Services
```bash
# Restart app only
docker compose -f docker-compose.prod.yml restart app

# Restart everything
docker compose -f docker-compose.prod.yml restart
```

### Update Application
```bash
cd /opt/mindrouter
git pull origin master

# Rebuild and restart
docker compose -f docker-compose.prod.yml up -d --build

# Run any new migrations
docker compose -f docker-compose.prod.yml exec app alembic upgrade head
```

### Backup Database
```bash
# Backup
docker compose -f docker-compose.prod.yml exec mariadb \
  mysqldump -u root -p mindrouter > backup_$(date +%Y%m%d).sql

# Restore
docker compose -f docker-compose.prod.yml exec -T mariadb \
  mysql -u root -p mindrouter < backup.sql
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker compose -f docker-compose.prod.yml logs app

# Check if ports are in use
sudo ss -tlnp | grep 8000
```

### Apache 502 Bad Gateway
```bash
# Check if app is running
curl http://127.0.0.1:8000/healthz

# Check SELinux
sudo ausearch -m avc -ts recent
sudo setsebool -P httpd_can_network_connect 1
```

### Database connection issues
```bash
# Check MariaDB is healthy
docker compose -f docker-compose.prod.yml ps mariadb

# Check connection from app container
docker compose -f docker-compose.prod.yml exec app \
  python -c "from backend.app.db.session import engine; print('OK')"
```

## Security Checklist

- [ ] Changed all default passwords in .env.prod
- [ ] Generated unique SECRET_KEY
- [ ] SSL certificate installed and working
- [ ] Firewall configured (only 80/443 open)
- [ ] SELinux properly configured
- [ ] Database not exposed externally
- [ ] Redis not exposed externally
- [ ] CORS_ORIGINS set to actual domain
- [ ] Disabled DEBUG mode
- [ ] GPU sidecar ports (9101) not exposed to public internet
- [ ] Sidecar agents running on all GPU nodes
