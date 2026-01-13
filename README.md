# Pupero Moderation Service

A comprehensive moderation system for the Pupero platform, providing user moderation, content moderation, trade dispute resolution, and Matrix chat integration.

## Features

### User Moderation
- **Warn** - Issue warnings to users
- **Mute** - Temporarily prevent users from chatting (with duration)
- **Ban** - Permanently ban users from the platform
- **Suspend** - Temporarily suspend user accounts (with duration)
- **Freeze Funds** - Freeze user funds during disputes or investigations

### Content Moderation
- Delete/hide offers, reviews, and profile content
- Edit inappropriate content
- Restore removed content

### Reporting System
- Users can report other users or content
- Report categories: scam, harassment, spam, inappropriate content, fraud, impersonation, illegal activity
- Non-anonymous reports (reporter identity visible to moderators)
- Automatic queue management

### Trade Dispute System
- Buyers and sellers can open disputes on transactions
- Threaded messaging within disputes
- Evidence submission from both parties
- Moderator resolution with fund distribution
- Resolution types: buyer favor, seller favor, split

### Appeals System
- Users can appeal moderation decisions
- Moderators review and approve/deny appeals
- Approved appeals automatically reverse the original action

### Notifications
- In-app notifications for all moderation actions
- Matrix message notifications
- Notification types: warnings, bans, mutes, content removal, dispute updates, appeal results

### Audit Logging
- Complete audit trail of all moderation actions
- Tracks: who, what, when, why, IP address, user agent
- Admin-only access to audit logs

### Dashboard & Statistics
- Pending reports, disputes, and appeals counts
- Active bans, mutes, and suspensions
- Daily action statistics
- Moderation queue management

### Matrix Integration
- Ban users from Matrix rooms
- Send direct messages via Matrix
- Sync moderation actions with Matrix

## API Endpoints

### Health
- `GET /healthz` - Liveness check
- `GET /health` - Health status

### User Status
- `GET /api/v1/users/{user_id}/check` - Check if user can perform actions (for other services)
- `GET /api/v1/users/{user_id}/status` - Get full moderation status (moderator only)

### Reports
- `POST /api/v1/reports` - Create a report
- `GET /api/v1/reports` - List reports (moderator only)
- `GET /api/v1/reports/{report_id}` - Get report details
- `PATCH /api/v1/reports/{report_id}` - Update report status

### Moderation Actions
- `POST /api/v1/actions` - Execute moderation action
- `GET /api/v1/actions` - List actions (moderator only)
- `GET /api/v1/actions/{action_id}` - Get action details
- `GET /api/v1/users/{user_id}/actions` - Get user's moderation history

### Disputes
- `POST /api/v1/disputes` - Create a dispute
- `GET /api/v1/disputes` - List disputes (moderator only)
- `GET /api/v1/disputes/{dispute_id}` - Get dispute details
- `PATCH /api/v1/disputes/{dispute_id}` - Update dispute
- `POST /api/v1/disputes/{dispute_id}/resolve` - Resolve dispute (moderator only)
- `GET /api/v1/users/{user_id}/disputes` - Get user's disputes
- `POST /api/v1/disputes/{dispute_id}/messages` - Add message to dispute
- `GET /api/v1/disputes/{dispute_id}/messages` - Get dispute messages

### Appeals
- `POST /api/v1/appeals` - Create an appeal
- `GET /api/v1/appeals` - List appeals (moderator only)
- `GET /api/v1/appeals/{appeal_id}` - Get appeal details
- `PATCH /api/v1/appeals/{appeal_id}` - Update/resolve appeal

### Notifications
- `GET /api/v1/notifications` - Get user's notifications
- `POST /api/v1/notifications/{notification_id}/read` - Mark as read
- `POST /api/v1/notifications/read-all` - Mark all as read

### Queue
- `GET /api/v1/queue` - Get moderation queue (moderator only)
- `POST /api/v1/queue/{queue_id}/claim` - Claim queue item
- `POST /api/v1/queue/{queue_id}/complete` - Complete queue item

### Dashboard
- `GET /api/v1/dashboard` - Get dashboard statistics (moderator only)
- `GET /api/v1/audit-logs` - Get audit logs (admin only)

### Matrix Integration
- `POST /api/v1/matrix/ban` - Ban user from Matrix
- `POST /api/v1/matrix/message` - Send Matrix message

### Fund Management
- `POST /api/v1/funds/increase` - Increase user funds (admin only)
- `POST /api/v1/funds/freeze` - Freeze user funds
- `POST /api/v1/funds/unfreeze` - Unfreeze user funds

## Environment Variables

```env
# Database
DATABASE_URL=mariadb+mariadbconnector://root:password@localhost:3306/pupero_moderation

# Service port
MODERATION_PORT=8020

# JWT (must match LoginBackend)
JWT_SECRET_KEY=your-secret-key

# Service URLs
LOGIN_SERVICE_URL=http://pupero-login:8001
OFFERS_SERVICE_URL=http://pupero-offers:8002
WALLET_SERVICE_URL=http://pupero-WalletManager:8004
TRANSACTIONS_SERVICE_URL=http://pupero-transactions:8003

# Matrix
MATRIX_HS_URL=http://pupero-matrix-synapse:8008
MATRIX_ADMIN_SECRET=your-admin-secret
MATRIX_SERVER_NAME=Pupero

# Logging
LOG_FILE=/var/log/pupero/moderation.log
```

## User Roles

- **user** - Can create reports, disputes, appeals; view own notifications
- **moderator** - Can review reports, resolve disputes, take moderation actions
- **admin** - Full access including fund adjustments and audit logs

## Database Tables

- `report` - User reports
- `moderation_action` - Log of all moderation actions
- `user_moderation_status` - Current status for each user
- `dispute` - Trade disputes
- `dispute_message` - Messages within disputes
- `appeal` - Appeals against moderation decisions
- `audit_log` - Complete audit trail
- `notification` - User notifications
- `moderation_queue` - Queue of pending items
- `moderation_stats` - Daily statistics

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="mariadb+mariadbconnector://root:password@localhost:3306/pupero_moderation"
export JWT_SECRET_KEY="your-secret-key"

# Run the service
uvicorn app.main:app --host 0.0.0.0 --port 8020
```

## Docker

```bash
# Build
docker build -t pupero-moderation -f Pupero-Moderation/Dockerfile .

# Run
docker run -p 8020:8020 --env-file Pupero-Moderation/.env pupero-moderation
```

## Integration with Other Services

Other Pupero services should call `/api/v1/users/{user_id}/check` before allowing user actions:

```python
import httpx

async def check_user_can_trade(user_id: int) -> bool:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://pupero-moderation:8020/api/v1/users/{user_id}/check")
        data = response.json()
        return data["can_trade"]
```

## License

Proprietary - Pupero Platform
