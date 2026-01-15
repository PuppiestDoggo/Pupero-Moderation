"""
Pupero Moderation Service
Full moderation system with reports, disputes, appeals, and Matrix integration.
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Depends, HTTPException, Body, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select, func, SQLModel
from jose import jwt, JWTError

from app.database import engine, get_session
from app.config import (
    JWT_SECRET_KEY, JWT_ALGORITHM, LOGIN_SERVICE_URL, OFFERS_SERVICE_URL,
    WALLET_SERVICE_URL, MATRIX_HS_URL, MATRIX_ADMIN_SECRET, MATRIX_SERVER_NAME,
    LOG_FILE
)
from app.models import (
    Report, ModerationAction, UserModerationStatus, Dispute, DisputeMessage,
    Appeal, AuditLog, Notification, ModerationQueue, ModerationStats,
    ReportStatus, ModerationActionType, DisputeStatus, AppealStatus,
    NotificationType, NotificationChannel, ContentType, ReportCategory
)
from app.schemas import (
    ReportCreate, ReportUpdate, ReportRead, ReportList,
    ModerationActionCreate, ModerationActionRead, ModerationActionList,
    UserModerationStatusRead, UserModerationCheck,
    DisputeCreate, DisputeUpdate, DisputeResolve, DisputeRead, DisputeList,
    DisputeMessageCreate, DisputeMessageRead,
    AppealCreate, AppealUpdate, AppealRead, AppealList,
    AuditLogRead, AuditLogList, NotificationRead, NotificationList,
    QueueItemRead, QueueList, DashboardStats, FundAdjustment, FundFreeze,
    MatrixBanRequest, MatrixKickRequest, MatrixMessageRequest,
    BulkActionRequest, BulkActionResponse
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moderation")
if LOG_FILE:
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)


@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    logger.info("Moderation service started, tables created")
    yield
    logger.info("Moderation service shutting down")


app = FastAPI(
    title="Pupero Moderation Service",
    description="Full moderation system for Pupero platform",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== AUTHENTICATION ==============

def get_current_user(request: Request, session: Session = Depends(get_session)):
    """Extract and validate JWT token from request"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"user_id": int(user_id), "role": payload.get("role", "user")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def require_moderator(current_user: dict = Depends(get_current_user)):
    """Require moderator or admin role"""
    if current_user["role"] not in ["moderator", "admin"]:
        raise HTTPException(status_code=403, detail="Moderator access required")
    return current_user


def require_admin(current_user: dict = Depends(get_current_user)):
    """Require admin role"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# ============== HEALTH ENDPOINTS ==============

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "moderation"}


# ============== HELPER FUNCTIONS ==============

def get_or_create_user_status(session: Session, user_id: int) -> UserModerationStatus:
    """Get or create moderation status for a user"""
    status = session.exec(select(UserModerationStatus).where(UserModerationStatus.user_id == user_id)).first()
    if not status:
        status = UserModerationStatus(user_id=user_id)
        session.add(status)
        session.commit()
        session.refresh(status)
    return status


def create_audit_log(session: Session, actor_id: int, actor_role: str, action: str,
                     target_type: str, target_id: str, details: dict, request: Request = None):
    """Create an audit log entry"""
    log = AuditLog(
        actor_user_id=actor_id,
        actor_role=actor_role,
        action=action,
        target_type=target_type,
        target_id=target_id,
        details=json.dumps(details),
        ip_address=request.client.host if request else None,
        user_agent=request.headers.get("User-Agent") if request else None
    )
    session.add(log)
    session.commit()


def create_notification(session: Session, user_id: int, notif_type: NotificationType,
                        title: str, message: str, **kwargs):
    """Create a notification for a user"""
    notif = Notification(
        user_id=user_id,
        notification_type=notif_type,
        title=title,
        message=message,
        channel=NotificationChannel.BOTH,
        **kwargs
    )
    session.add(notif)
    session.commit()
    return notif


def add_to_queue(session: Session, item_type: str, item_id: int, priority: int = 0):
    """Add item to moderation queue"""
    queue_item = ModerationQueue(item_type=item_type, item_id=item_id, priority=priority)
    session.add(queue_item)
    session.commit()


# ============== USER STATUS CHECK (for other services) ==============

@app.get("/api/v1/users/{user_id}/check", response_model=UserModerationCheck)
def check_user_status(user_id: int, session: Session = Depends(get_session)):
    """Check if a user can perform actions (called by other services)"""
    status = session.exec(select(UserModerationStatus).where(UserModerationStatus.user_id == user_id)).first()
    
    if not status:
        return UserModerationCheck(
            user_id=user_id, can_login=True, can_trade=True, can_chat=True,
            is_banned=False, is_muted=False, is_suspended=False, funds_frozen=False
        )
    
    # Check if temporary actions have expired
    now = datetime.utcnow()
    if status.is_muted and status.mute_expires_at and status.mute_expires_at < now:
        status.is_muted = False
        status.mute_reason = None
        session.commit()
    if status.is_suspended and status.suspend_expires_at and status.suspend_expires_at < now:
        status.is_suspended = False
        status.suspend_reason = None
        session.commit()
    
    can_login = not status.is_banned
    can_trade = not (status.is_banned or status.is_suspended or status.funds_frozen)
    can_chat = not (status.is_banned or status.is_muted)
    
    message = None
    if status.is_banned:
        message = f"Account banned: {status.ban_reason}"
    elif status.is_suspended:
        message = f"Account suspended until {status.suspend_expires_at}: {status.suspend_reason}"
    elif status.is_muted:
        message = f"Muted until {status.mute_expires_at}: {status.mute_reason}"
    
    return UserModerationCheck(
        user_id=user_id, can_login=can_login, can_trade=can_trade, can_chat=can_chat,
        is_banned=status.is_banned, is_muted=status.is_muted, is_suspended=status.is_suspended,
        funds_frozen=status.funds_frozen, message=message
    )


@app.get("/api/v1/users/{user_id}/status", response_model=UserModerationStatusRead)
def get_user_moderation_status(user_id: int, session: Session = Depends(get_session),
                                current_user: dict = Depends(require_moderator)):
    """Get full moderation status for a user (moderator only)"""
    status = get_or_create_user_status(session, user_id)
    return status


# ============== REPORT ENDPOINTS ==============

@app.post("/api/v1/reports", response_model=ReportRead)
def create_report(report_in: ReportCreate, request: Request, session: Session = Depends(get_session),
                  current_user: dict = Depends(get_current_user)):
    """Create a new report (any authenticated user)"""
    report = Report(
        reporter_user_id=current_user["user_id"],
        reported_user_id=report_in.reported_user_id,
        content_type=report_in.content_type,
        content_id=report_in.content_id,
        category=report_in.category,
        description=report_in.description,
        evidence=report_in.evidence
    )
    session.add(report)
    session.commit()
    session.refresh(report)
    add_to_queue(session, "report", report.id, priority=1)
    create_audit_log(session, current_user["user_id"], current_user["role"], "create_report",
                     "report", str(report.id), {"category": report_in.category.value}, request)
    return report


@app.get("/api/v1/reports", response_model=ReportList)
def list_reports(status: Optional[ReportStatus] = None, page: int = 1, per_page: int = 20,
                 session: Session = Depends(get_session), current_user: dict = Depends(require_moderator)):
    """List reports (moderator only)"""
    query = select(Report)
    if status:
        query = query.where(Report.status == status)
    query = query.order_by(Report.created_at.desc())
    total = session.exec(select(func.count()).select_from(Report)).one()
    reports = session.exec(query.offset((page-1)*per_page).limit(per_page)).all()
    return ReportList(reports=reports, total=total, page=page, per_page=per_page)


@app.get("/api/v1/reports/{report_id}", response_model=ReportRead)
def get_report(report_id: int, session: Session = Depends(get_session),
               current_user: dict = Depends(require_moderator)):
    """Get a specific report"""
    report = session.get(Report, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@app.patch("/api/v1/reports/{report_id}", response_model=ReportRead)
def update_report(report_id: int, update_in: ReportUpdate, request: Request,
                  session: Session = Depends(get_session), current_user: dict = Depends(require_moderator)):
    """Update a report (moderator only)"""
    report = session.get(Report, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    update_data = update_in.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(report, key, value)
    
    if update_in.status in [ReportStatus.RESOLVED, ReportStatus.DISMISSED]:
        report.resolved_at = datetime.utcnow()
        create_notification(session, report.reporter_user_id, NotificationType.REPORT_RESOLVED,
                           "Report Resolved", f"Your report has been {update_in.status.value}",
                           related_report_id=report.id)
    
    session.commit()
    session.refresh(report)
    create_audit_log(session, current_user["user_id"], current_user["role"], "update_report",
                     "report", str(report_id), update_data, request)
    return report


# ============== MODERATION ACTION ENDPOINTS ==============

@app.post("/api/v1/actions", response_model=ModerationActionRead)
def create_moderation_action(action_in: ModerationActionCreate, request: Request,
                              session: Session = Depends(get_session),
                              current_user: dict = Depends(require_moderator)):
    """Execute a moderation action (moderator only)"""
    # Admin-only actions
    admin_only_actions = [ModerationActionType.INCREASE_FUNDS, ModerationActionType.DECREASE_FUNDS]
    if action_in.action_type in admin_only_actions and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required for this action")
    
    expires_at = None
    if action_in.duration_minutes and action_in.action_type in [ModerationActionType.MUTE, ModerationActionType.SUSPEND]:
        expires_at = datetime.utcnow() + timedelta(minutes=action_in.duration_minutes)
    
    action = ModerationAction(
        moderator_user_id=current_user["user_id"],
        target_user_id=action_in.target_user_id,
        action_type=action_in.action_type,
        content_type=action_in.content_type,
        content_id=action_in.content_id,
        reason=action_in.reason,
        details=action_in.details,
        related_report_id=action_in.related_report_id,
        related_dispute_id=action_in.related_dispute_id,
        duration_minutes=action_in.duration_minutes,
        expires_at=expires_at
    )
    session.add(action)
    session.commit()
    session.refresh(action)
    
    # Apply action to user status
    if action_in.target_user_id:
        status = get_or_create_user_status(session, action_in.target_user_id)
        _apply_action_to_status(session, status, action_in, expires_at)
        _send_action_notification(session, action_in, action.id)
    
    create_audit_log(session, current_user["user_id"], current_user["role"], 
                     f"moderation_action_{action_in.action_type.value}",
                     "user" if action_in.target_user_id else "content",
                     str(action_in.target_user_id or action_in.content_id),
                     {"action_type": action_in.action_type.value, "reason": action_in.reason}, request)
    
    return action


def _apply_action_to_status(session: Session, status: UserModerationStatus, 
                            action_in: ModerationActionCreate, expires_at: Optional[datetime]):
    """Apply moderation action to user status"""
    action_type = action_in.action_type
    
    if action_type == ModerationActionType.WARN:
        status.warning_count += 1
    elif action_type == ModerationActionType.MUTE:
        status.is_muted = True
        status.mute_reason = action_in.reason
        status.mute_expires_at = expires_at
    elif action_type == ModerationActionType.UNMUTE:
        status.is_muted = False
        status.mute_reason = None
        status.mute_expires_at = None
    elif action_type == ModerationActionType.BAN:
        status.is_banned = True
        status.ban_reason = action_in.reason
    elif action_type == ModerationActionType.UNBAN:
        status.is_banned = False
        status.ban_reason = None
    elif action_type == ModerationActionType.SUSPEND:
        status.is_suspended = True
        status.suspend_reason = action_in.reason
        status.suspend_expires_at = expires_at
    elif action_type == ModerationActionType.UNSUSPEND:
        status.is_suspended = False
        status.suspend_reason = None
        status.suspend_expires_at = None
    elif action_type == ModerationActionType.FREEZE_FUNDS:
        status.funds_frozen = True
        if action_in.amount:
            status.frozen_amount = action_in.amount
    elif action_type == ModerationActionType.UNFREEZE_FUNDS:
        status.funds_frozen = False
        status.frozen_amount = 0.0
    
    status.last_action_at = datetime.utcnow()
    session.commit()


def _send_action_notification(session: Session, action_in: ModerationActionCreate, action_id: int):
    """Send notification to user about moderation action"""
    if not action_in.target_user_id:
        return
    
    notif_map = {
        ModerationActionType.WARN: (NotificationType.WARNING_RECEIVED, "Warning Received", "You have received a warning"),
        ModerationActionType.MUTE: (NotificationType.MUTED, "Account Muted", "Your account has been muted"),
        ModerationActionType.UNMUTE: (NotificationType.UNMUTED, "Mute Lifted", "Your mute has been lifted"),
        ModerationActionType.BAN: (NotificationType.BANNED, "Account Banned", "Your account has been banned"),
        ModerationActionType.UNBAN: (NotificationType.UNBANNED, "Ban Lifted", "Your ban has been lifted"),
        ModerationActionType.SUSPEND: (NotificationType.SUSPENDED, "Account Suspended", "Your account has been suspended"),
        ModerationActionType.UNSUSPEND: (NotificationType.UNSUSPENDED, "Suspension Lifted", "Your suspension has been lifted"),
        ModerationActionType.FREEZE_FUNDS: (NotificationType.FUNDS_FROZEN, "Funds Frozen", "Your funds have been frozen"),
        ModerationActionType.UNFREEZE_FUNDS: (NotificationType.FUNDS_UNFROZEN, "Funds Unfrozen", "Your funds have been unfrozen"),
    }
    
    if action_in.action_type in notif_map:
        notif_type, title, message = notif_map[action_in.action_type]
        create_notification(session, action_in.target_user_id, notif_type, title,
                           f"{message}. Reason: {action_in.reason}", related_action_id=action_id)


@app.get("/api/v1/actions", response_model=ModerationActionList)
def list_moderation_actions(target_user_id: Optional[int] = None, action_type: Optional[ModerationActionType] = None,
                            page: int = 1, per_page: int = 20, session: Session = Depends(get_session),
                            current_user: dict = Depends(require_moderator)):
    """List moderation actions (moderator only)"""
    query = select(ModerationAction)
    if target_user_id:
        query = query.where(ModerationAction.target_user_id == target_user_id)
    if action_type:
        query = query.where(ModerationAction.action_type == action_type)
    query = query.order_by(ModerationAction.created_at.desc())
    total = session.exec(select(func.count()).select_from(ModerationAction)).one()
    actions = session.exec(query.offset((page-1)*per_page).limit(per_page)).all()
    return ModerationActionList(actions=actions, total=total, page=page, per_page=per_page)


@app.get("/api/v1/actions/{action_id}", response_model=ModerationActionRead)
def get_moderation_action(action_id: int, session: Session = Depends(get_session),
                          current_user: dict = Depends(require_moderator)):
    """Get a specific moderation action"""
    action = session.get(ModerationAction, action_id)
    if not action:
        raise HTTPException(status_code=404, detail="Action not found")
    return action


@app.get("/api/v1/users/{user_id}/actions", response_model=ModerationActionList)
def get_user_actions(user_id: int, page: int = 1, per_page: int = 20,
                     session: Session = Depends(get_session), current_user: dict = Depends(require_moderator)):
    """Get all moderation actions for a user"""
    query = select(ModerationAction).where(ModerationAction.target_user_id == user_id)
    query = query.order_by(ModerationAction.created_at.desc())
    total = session.exec(select(func.count()).select_from(ModerationAction).where(
        ModerationAction.target_user_id == user_id)).one()
    actions = session.exec(query.offset((page-1)*per_page).limit(per_page)).all()
    return ModerationActionList(actions=actions, total=total, page=page, per_page=per_page)


@app.get("/users/search")
def search_users(q: str = Query(..., description="Search query for username, email, or user ID"),
                 limit: int = Query(50, ge=1, le=100),
                 session: Session = Depends(get_session),
                 current_user: dict = Depends(require_moderator)):
    """Search users by username, email, or ID (moderator only).
    Proxies to the login service's user directory endpoint and enriches with moderation status."""
    users = []
    
    try:
        # Call the login service to search users
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                f"{LOGIN_SERVICE_URL}/users/directory",
                params={"q": q, "limit": limit}
            )
            if response.status_code == 200:
                users = response.json()
    except Exception as e:
        logging.warning(f"Failed to fetch users from login service: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch users from login service: {e}")
    
    # Enrich users with moderation status
    enriched_users = []
    for user in users:
        user_id = user.get("id")
        mod_status = session.exec(
            select(UserModerationStatus).where(UserModerationStatus.user_id == user_id)
        ).first()
        
        enriched_user = {
            "id": user_id,
            "username": user.get("username"),
            "email": user.get("email", ""),
            "role": user.get("role", "user"),
            "created_at": user.get("created_at"),
            "is_banned": mod_status.is_banned if mod_status else False,
            "is_muted": mod_status.is_muted if mod_status else False,
            "is_suspended": mod_status.is_suspended if mod_status else False,
            "funds_frozen": mod_status.funds_frozen if mod_status else False,
        }
        enriched_users.append(enriched_user)
    
    return enriched_users


@app.get("/users/{user_id}")
def get_user_detail(user_id: int,
                    session: Session = Depends(get_session),
                    current_user: dict = Depends(require_moderator)):
    """Get user details with moderation status (moderator only).
    Fetches user info from login service and enriches with moderation status."""
    user_data = None
    
    try:
        # Call the login service to get user info (public endpoint)
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{LOGIN_SERVICE_URL}/users/{user_id}/public")
            if response.status_code == 200:
                user_data = response.json()
    except Exception as e:
        logging.warning(f"Failed to fetch user from login service: {e}")
    
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get moderation status
    mod_status = session.exec(
        select(UserModerationStatus).where(UserModerationStatus.user_id == user_id)
    ).first()
    
    return {
        "id": user_id,
        "username": user_data.get("username"),
        "email": user_data.get("email", ""),
        "role": user_data.get("role", "user"),
        "created_at": user_data.get("created_at"),
        "is_banned": mod_status.is_banned if mod_status else False,
        "is_muted": mod_status.is_muted if mod_status else False,
        "is_suspended": mod_status.is_suspended if mod_status else False,
        "funds_frozen": mod_status.funds_frozen if mod_status else False,
        "warning_count": mod_status.warning_count if mod_status else 0,
        "ban_reason": mod_status.ban_reason if mod_status else None,
        "mute_reason": mod_status.mute_reason if mod_status else None,
        "suspend_reason": mod_status.suspend_reason if mod_status else None,
        "mute_expires_at": str(mod_status.mute_expires_at) if mod_status and mod_status.mute_expires_at else None,
        "suspend_expires_at": str(mod_status.suspend_expires_at) if mod_status and mod_status.suspend_expires_at else None,
    }


@app.get("/users/{user_id}/actions")
def get_user_actions_short(user_id: int, page: int = 1, per_page: int = 20,
                           session: Session = Depends(get_session),
                           current_user: dict = Depends(require_moderator)):
    """Get all moderation actions for a user (short URL without /api/v1 prefix)"""
    query = select(ModerationAction).where(ModerationAction.target_user_id == user_id)
    query = query.order_by(ModerationAction.created_at.desc())
    actions = session.exec(query.offset((page-1)*per_page).limit(per_page)).all()
    # Return as list of dicts for simpler frontend consumption
    return [
        {
            "id": a.id,
            "action_type": a.action_type.value if a.action_type else None,
            "reason": a.reason,
            "moderator_user_id": a.moderator_user_id,
            "created_at": str(a.created_at) if a.created_at else None,
            "duration_minutes": a.duration_minutes,
            "expires_at": str(a.expires_at) if a.expires_at else None,
            "is_active": a.is_active,
        }
        for a in actions
    ]


# ============== DISPUTE ENDPOINTS ==============

@app.post("/api/v1/disputes", response_model=DisputeRead)
def create_dispute(dispute_in: DisputeCreate, request: Request, session: Session = Depends(get_session),
                   current_user: dict = Depends(get_current_user)):
    """Create a new trade dispute (buyer or seller)"""
    # Verify user is part of the transaction (would need to call transactions service)
    user_id = current_user["user_id"]
    
    dispute = Dispute(
        transaction_id=dispute_in.transaction_id,
        offer_id=dispute_in.offer_id,
        buyer_id=user_id,  # Will be updated when we verify transaction
        seller_id=0,  # Will be updated when we verify transaction
        initiated_by_user_id=user_id,
        category=dispute_in.category,
        description=dispute_in.description
    )
    
    # Set initial statement based on who initiated
    if dispute_in.evidence:
        dispute.evidence_buyer = dispute_in.evidence  # Assume initiator is buyer for now
    
    session.add(dispute)
    session.commit()
    session.refresh(dispute)
    
    add_to_queue(session, "dispute", dispute.id, priority=2)  # Higher priority than reports
    
    # Notify the other party
    create_notification(session, dispute.seller_id, NotificationType.DISPUTE_OPENED,
                       "Dispute Opened", f"A dispute has been opened for transaction #{dispute_in.transaction_id}",
                       related_dispute_id=dispute.id)
    
    create_audit_log(session, user_id, current_user["role"], "create_dispute",
                     "dispute", str(dispute.id), {"transaction_id": dispute_in.transaction_id}, request)
    
    return dispute


@app.get("/api/v1/disputes", response_model=DisputeList)
def list_disputes(status: Optional[DisputeStatus] = None, page: int = 1, per_page: int = 20,
                  session: Session = Depends(get_session), current_user: dict = Depends(require_moderator)):
    """List all disputes (moderator only)"""
    query = select(Dispute)
    if status:
        query = query.where(Dispute.status == status)
    query = query.order_by(Dispute.created_at.desc())
    total = session.exec(select(func.count()).select_from(Dispute)).one()
    disputes = session.exec(query.offset((page-1)*per_page).limit(per_page)).all()
    return DisputeList(disputes=disputes, total=total, page=page, per_page=per_page)


@app.get("/api/v1/disputes/{dispute_id}", response_model=DisputeRead)
def get_dispute(dispute_id: int, session: Session = Depends(get_session),
                current_user: dict = Depends(get_current_user)):
    """Get a specific dispute (parties involved or moderator)"""
    dispute = session.get(Dispute, dispute_id)
    if not dispute:
        raise HTTPException(status_code=404, detail="Dispute not found")
    
    user_id = current_user["user_id"]
    is_party = user_id in [dispute.buyer_id, dispute.seller_id]
    is_mod = current_user["role"] in ["moderator", "admin"]
    
    if not is_party and not is_mod:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return dispute


@app.patch("/api/v1/disputes/{dispute_id}", response_model=DisputeRead)
def update_dispute(dispute_id: int, update_in: DisputeUpdate, request: Request,
                   session: Session = Depends(get_session), current_user: dict = Depends(get_current_user)):
    """Update a dispute (add statement/evidence or moderator update)"""
    dispute = session.get(Dispute, dispute_id)
    if not dispute:
        raise HTTPException(status_code=404, detail="Dispute not found")
    
    user_id = current_user["user_id"]
    is_buyer = user_id == dispute.buyer_id
    is_seller = user_id == dispute.seller_id
    is_mod = current_user["role"] in ["moderator", "admin"]
    
    if not is_buyer and not is_seller and not is_mod:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Users can only update their own statements
    if is_buyer and not is_mod:
        if update_in.buyer_statement:
            dispute.buyer_statement = update_in.buyer_statement
        if update_in.evidence_buyer:
            dispute.evidence_buyer = update_in.evidence_buyer
    elif is_seller and not is_mod:
        if update_in.seller_statement:
            dispute.seller_statement = update_in.seller_statement
        if update_in.evidence_seller:
            dispute.evidence_seller = update_in.evidence_seller
    elif is_mod:
        update_data = update_in.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(dispute, key, value)
    
    session.commit()
    session.refresh(dispute)
    
    # Notify parties of update
    other_party = dispute.seller_id if is_buyer else dispute.buyer_id
    if not is_mod:
        create_notification(session, other_party, NotificationType.DISPUTE_UPDATE,
                           "Dispute Updated", "The other party has updated the dispute",
                           related_dispute_id=dispute.id)
    
    create_audit_log(session, user_id, current_user["role"], "update_dispute",
                     "dispute", str(dispute_id), {"updated_by": "moderator" if is_mod else "party"}, request)
    
    return dispute


@app.post("/api/v1/disputes/{dispute_id}/resolve", response_model=DisputeRead)
def resolve_dispute(dispute_id: int, resolve_in: DisputeResolve, request: Request,
                    session: Session = Depends(get_session), current_user: dict = Depends(require_moderator)):
    """Resolve a dispute (moderator only)"""
    dispute = session.get(Dispute, dispute_id)
    if not dispute:
        raise HTTPException(status_code=404, detail="Dispute not found")
    
    if dispute.status in [DisputeStatus.RESOLVED_BUYER_FAVOR, DisputeStatus.RESOLVED_SELLER_FAVOR, 
                          DisputeStatus.RESOLVED_SPLIT, DisputeStatus.CANCELLED]:
        raise HTTPException(status_code=400, detail="Dispute already resolved")
    
    dispute.status = resolve_in.status
    dispute.resolution_notes = resolve_in.resolution_notes
    dispute.resolution_amount_to_buyer = resolve_in.resolution_amount_to_buyer
    dispute.resolution_amount_to_seller = resolve_in.resolution_amount_to_seller
    dispute.resolved_at = datetime.utcnow()
    
    session.commit()
    session.refresh(dispute)
    
    # Notify both parties
    create_notification(session, dispute.buyer_id, NotificationType.DISPUTE_RESOLVED,
                       "Dispute Resolved", f"Your dispute has been resolved: {resolve_in.status.value}",
                       related_dispute_id=dispute.id)
    create_notification(session, dispute.seller_id, NotificationType.DISPUTE_RESOLVED,
                       "Dispute Resolved", f"The dispute has been resolved: {resolve_in.status.value}",
                       related_dispute_id=dispute.id)
    
    create_audit_log(session, current_user["user_id"], current_user["role"], "resolve_dispute",
                     "dispute", str(dispute_id), 
                     {"status": resolve_in.status.value, "buyer_amount": resolve_in.resolution_amount_to_buyer,
                      "seller_amount": resolve_in.resolution_amount_to_seller}, request)
    
    return dispute


@app.get("/api/v1/users/{user_id}/disputes", response_model=DisputeList)
def get_user_disputes(user_id: int, page: int = 1, per_page: int = 20,
                      session: Session = Depends(get_session), current_user: dict = Depends(get_current_user)):
    """Get disputes for a user (own disputes or moderator)"""
    if current_user["user_id"] != user_id and current_user["role"] not in ["moderator", "admin"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    query = select(Dispute).where((Dispute.buyer_id == user_id) | (Dispute.seller_id == user_id))
    query = query.order_by(Dispute.created_at.desc())
    total = session.exec(select(func.count()).select_from(Dispute).where(
        (Dispute.buyer_id == user_id) | (Dispute.seller_id == user_id))).one()
    disputes = session.exec(query.offset((page-1)*per_page).limit(per_page)).all()
    return DisputeList(disputes=disputes, total=total, page=page, per_page=per_page)


# ============== DISPUTE MESSAGE ENDPOINTS ==============

@app.post("/api/v1/disputes/{dispute_id}/messages", response_model=DisputeMessageRead)
def create_dispute_message(dispute_id: int, message_in: DisputeMessageCreate,
                           session: Session = Depends(get_session), current_user: dict = Depends(get_current_user)):
    """Add a message to a dispute thread"""
    dispute = session.get(Dispute, dispute_id)
    if not dispute:
        raise HTTPException(status_code=404, detail="Dispute not found")
    
    user_id = current_user["user_id"]
    is_party = user_id in [dispute.buyer_id, dispute.seller_id]
    is_mod = current_user["role"] in ["moderator", "admin"]
    
    if not is_party and not is_mod:
        raise HTTPException(status_code=403, detail="Access denied")
    
    message = DisputeMessage(
        dispute_id=dispute_id,
        sender_user_id=user_id,
        is_moderator=is_mod,
        message=message_in.message,
        attachments=message_in.attachments
    )
    session.add(message)
    session.commit()
    session.refresh(message)
    
    # Notify other parties
    recipients = [dispute.buyer_id, dispute.seller_id]
    if is_mod and dispute.assigned_moderator_id:
        recipients.append(dispute.assigned_moderator_id)
    recipients = [r for r in recipients if r != user_id]
    
    for recipient in recipients:
        create_notification(session, recipient, NotificationType.DISPUTE_UPDATE,
                           "New Dispute Message", "A new message has been added to the dispute",
                           related_dispute_id=dispute_id)
    
    return message


@app.get("/api/v1/disputes/{dispute_id}/messages", response_model=list[DisputeMessageRead])
def get_dispute_messages(dispute_id: int, session: Session = Depends(get_session),
                         current_user: dict = Depends(get_current_user)):
    """Get all messages for a dispute"""
    dispute = session.get(Dispute, dispute_id)
    if not dispute:
        raise HTTPException(status_code=404, detail="Dispute not found")
    
    user_id = current_user["user_id"]
    is_party = user_id in [dispute.buyer_id, dispute.seller_id]
    is_mod = current_user["role"] in ["moderator", "admin"]
    
    if not is_party and not is_mod:
        raise HTTPException(status_code=403, detail="Access denied")
    
    messages = session.exec(select(DisputeMessage).where(DisputeMessage.dispute_id == dispute_id)
                           .order_by(DisputeMessage.created_at)).all()
    return messages


# ============== APPEAL ENDPOINTS ==============

@app.post("/api/v1/appeals", response_model=AppealRead)
def create_appeal(appeal_in: AppealCreate, request: Request, session: Session = Depends(get_session),
                  current_user: dict = Depends(get_current_user)):
    """Create an appeal against a moderation action"""
    action = session.get(ModerationAction, appeal_in.moderation_action_id)
    if not action:
        raise HTTPException(status_code=404, detail="Moderation action not found")
    
    if action.target_user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Can only appeal actions against yourself")
    
    # Check if already appealed
    existing = session.exec(select(Appeal).where(
        Appeal.moderation_action_id == appeal_in.moderation_action_id)).first()
    if existing:
        raise HTTPException(status_code=400, detail="This action has already been appealed")
    
    appeal = Appeal(
        user_id=current_user["user_id"],
        moderation_action_id=appeal_in.moderation_action_id,
        reason=appeal_in.reason,
        evidence=appeal_in.evidence
    )
    session.add(appeal)
    session.commit()
    session.refresh(appeal)
    
    add_to_queue(session, "appeal", appeal.id, priority=1)
    create_audit_log(session, current_user["user_id"], current_user["role"], "create_appeal",
                     "appeal", str(appeal.id), {"action_id": appeal_in.moderation_action_id}, request)
    
    return appeal


@app.get("/api/v1/appeals", response_model=AppealList)
def list_appeals(status: Optional[AppealStatus] = None, page: int = 1, per_page: int = 20,
                 session: Session = Depends(get_session), current_user: dict = Depends(require_moderator)):
    """List appeals (moderator only)"""
    query = select(Appeal)
    if status:
        query = query.where(Appeal.status == status)
    query = query.order_by(Appeal.created_at.desc())
    total = session.exec(select(func.count()).select_from(Appeal)).one()
    appeals = session.exec(query.offset((page-1)*per_page).limit(per_page)).all()
    return AppealList(appeals=appeals, total=total, page=page, per_page=per_page)


@app.get("/api/v1/appeals/{appeal_id}", response_model=AppealRead)
def get_appeal(appeal_id: int, session: Session = Depends(get_session),
               current_user: dict = Depends(get_current_user)):
    """Get a specific appeal"""
    appeal = session.get(Appeal, appeal_id)
    if not appeal:
        raise HTTPException(status_code=404, detail="Appeal not found")
    
    if appeal.user_id != current_user["user_id"] and current_user["role"] not in ["moderator", "admin"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return appeal


@app.patch("/api/v1/appeals/{appeal_id}", response_model=AppealRead)
def update_appeal(appeal_id: int, update_in: AppealUpdate, request: Request,
                  session: Session = Depends(get_session), current_user: dict = Depends(require_moderator)):
    """Update/resolve an appeal (moderator only)"""
    appeal = session.get(Appeal, appeal_id)
    if not appeal:
        raise HTTPException(status_code=404, detail="Appeal not found")
    
    update_data = update_in.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(appeal, key, value)
    
    if update_in.status in [AppealStatus.APPROVED, AppealStatus.DENIED]:
        appeal.resolved_at = datetime.utcnow()
        
        # If approved, reverse the original action
        if update_in.status == AppealStatus.APPROVED:
            action = session.get(ModerationAction, appeal.moderation_action_id)
            if action:
                action.is_active = False
                # Reverse the effect on user status
                if action.target_user_id:
                    status = get_or_create_user_status(session, action.target_user_id)
                    _reverse_action_on_status(status, action.action_type)
        
        create_notification(session, appeal.user_id, NotificationType.APPEAL_RESOLVED,
                           "Appeal Resolved", f"Your appeal has been {update_in.status.value}",
                           related_appeal_id=appeal.id)
    
    session.commit()
    session.refresh(appeal)
    create_audit_log(session, current_user["user_id"], current_user["role"], "update_appeal",
                     "appeal", str(appeal_id), update_data, request)
    
    return appeal


def _reverse_action_on_status(status: UserModerationStatus, action_type: ModerationActionType):
    """Reverse a moderation action on user status"""
    if action_type == ModerationActionType.MUTE:
        status.is_muted = False
        status.mute_reason = None
    elif action_type == ModerationActionType.BAN:
        status.is_banned = False
        status.ban_reason = None
    elif action_type == ModerationActionType.SUSPEND:
        status.is_suspended = False
        status.suspend_reason = None
    elif action_type == ModerationActionType.FREEZE_FUNDS:
        status.funds_frozen = False
        status.frozen_amount = 0.0


# ============== NOTIFICATION ENDPOINTS ==============

@app.get("/api/v1/notifications", response_model=NotificationList)
def get_notifications(unread_only: bool = False, page: int = 1, per_page: int = 20,
                      session: Session = Depends(get_session), current_user: dict = Depends(get_current_user)):
    """Get notifications for current user"""
    query = select(Notification).where(Notification.user_id == current_user["user_id"])
    if unread_only:
        query = query.where(Notification.is_read == False)
    query = query.order_by(Notification.created_at.desc())
    
    total = session.exec(select(func.count()).select_from(Notification).where(
        Notification.user_id == current_user["user_id"])).one()
    unread_count = session.exec(select(func.count()).select_from(Notification).where(
        (Notification.user_id == current_user["user_id"]) & (Notification.is_read == False))).one()
    
    notifications = session.exec(query.offset((page-1)*per_page).limit(per_page)).all()
    return NotificationList(notifications=notifications, total=total, unread_count=unread_count)


@app.post("/api/v1/notifications/{notification_id}/read")
def mark_notification_read(notification_id: int, session: Session = Depends(get_session),
                           current_user: dict = Depends(get_current_user)):
    """Mark a notification as read"""
    notif = session.get(Notification, notification_id)
    if not notif:
        raise HTTPException(status_code=404, detail="Notification not found")
    if notif.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    notif.is_read = True
    notif.read_at = datetime.utcnow()
    session.commit()
    return {"status": "ok"}


@app.post("/api/v1/notifications/read-all")
def mark_all_notifications_read(session: Session = Depends(get_session),
                                 current_user: dict = Depends(get_current_user)):
    """Mark all notifications as read"""
    notifications = session.exec(select(Notification).where(
        (Notification.user_id == current_user["user_id"]) & (Notification.is_read == False))).all()
    
    for notif in notifications:
        notif.is_read = True
        notif.read_at = datetime.utcnow()
    
    session.commit()
    return {"status": "ok", "marked_count": len(notifications)}


# ============== MODERATION QUEUE ENDPOINTS ==============

@app.get("/api/v1/queue", response_model=QueueList)
def get_moderation_queue(item_type: Optional[str] = None, status: str = "pending",
                         session: Session = Depends(get_session), current_user: dict = Depends(require_moderator)):
    """Get moderation queue (moderator only)"""
    query = select(ModerationQueue).where(ModerationQueue.status == status)
    if item_type:
        query = query.where(ModerationQueue.item_type == item_type)
    query = query.order_by(ModerationQueue.priority.desc(), ModerationQueue.created_at)
    
    items = session.exec(query).all()
    pending_count = session.exec(select(func.count()).select_from(ModerationQueue).where(
        ModerationQueue.status == "pending")).one()
    in_progress_count = session.exec(select(func.count()).select_from(ModerationQueue).where(
        ModerationQueue.status == "in_progress")).one()
    
    return QueueList(items=items, total=len(items), pending_count=pending_count, in_progress_count=in_progress_count)


@app.post("/api/v1/queue/{queue_id}/claim")
def claim_queue_item(queue_id: int, session: Session = Depends(get_session),
                     current_user: dict = Depends(require_moderator)):
    """Claim a queue item for review"""
    item = session.get(ModerationQueue, queue_id)
    if not item:
        raise HTTPException(status_code=404, detail="Queue item not found")
    if item.status != "pending":
        raise HTTPException(status_code=400, detail="Item already claimed or completed")
    
    item.status = "in_progress"
    item.assigned_moderator_id = current_user["user_id"]
    item.claimed_at = datetime.utcnow()
    session.commit()
    return {"status": "ok", "item_type": item.item_type, "item_id": item.item_id}


@app.post("/api/v1/queue/{queue_id}/complete")
def complete_queue_item(queue_id: int, session: Session = Depends(get_session),
                        current_user: dict = Depends(require_moderator)):
    """Mark a queue item as completed"""
    item = session.get(ModerationQueue, queue_id)
    if not item:
        raise HTTPException(status_code=404, detail="Queue item not found")
    
    item.status = "completed"
    item.completed_at = datetime.utcnow()
    session.commit()
    return {"status": "ok"}


# ============== DASHBOARD & STATISTICS ==============

@app.get("/api/v1/dashboard", response_model=DashboardStats)
def get_dashboard_stats(session: Session = Depends(get_session), current_user: dict = Depends(require_moderator)):
    """Get dashboard statistics (moderator only)"""
    pending_reports = session.exec(select(func.count()).select_from(Report).where(
        Report.status == ReportStatus.PENDING)).one()
    pending_disputes = session.exec(select(func.count()).select_from(Dispute).where(
        Dispute.status == DisputeStatus.OPEN)).one()
    pending_appeals = session.exec(select(func.count()).select_from(Appeal).where(
        Appeal.status == AppealStatus.PENDING)).one()
    queue_size = session.exec(select(func.count()).select_from(ModerationQueue).where(
        ModerationQueue.status == "pending")).one()
    
    active_bans = session.exec(select(func.count()).select_from(UserModerationStatus).where(
        UserModerationStatus.is_banned == True)).one()
    active_mutes = session.exec(select(func.count()).select_from(UserModerationStatus).where(
        UserModerationStatus.is_muted == True)).one()
    active_suspensions = session.exec(select(func.count()).select_from(UserModerationStatus).where(
        UserModerationStatus.is_suspended == True)).one()
    
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_actions = session.exec(select(func.count()).select_from(ModerationAction).where(
        ModerationAction.created_at >= today)).one()
    today_resolved = session.exec(select(func.count()).select_from(Report).where(
        (Report.resolved_at >= today) & (Report.status == ReportStatus.RESOLVED))).one()
    
    # Most reported users (top 5)
    # This is a simplified version - in production you'd want a proper aggregation
    most_reported = []
    
    # Recent actions (last 10)
    recent_actions = session.exec(select(ModerationAction).order_by(
        ModerationAction.created_at.desc()).limit(10)).all()
    
    return DashboardStats(
        pending_reports=pending_reports,
        pending_disputes=pending_disputes,
        pending_appeals=pending_appeals,
        queue_size=queue_size,
        active_bans=active_bans,
        active_mutes=active_mutes,
        active_suspensions=active_suspensions,
        today_actions=today_actions,
        today_resolved=today_resolved,
        most_reported_users=most_reported,
        recent_actions=recent_actions
    )


@app.get("/api/v1/audit-logs", response_model=AuditLogList)
def get_audit_logs(actor_user_id: Optional[int] = None, action: Optional[str] = None,
                   page: int = 1, per_page: int = 50, session: Session = Depends(get_session),
                   current_user: dict = Depends(require_admin)):
    """Get audit logs (admin only)"""
    query = select(AuditLog)
    if actor_user_id:
        query = query.where(AuditLog.actor_user_id == actor_user_id)
    if action:
        query = query.where(AuditLog.action == action)
    query = query.order_by(AuditLog.created_at.desc())
    
    total = session.exec(select(func.count()).select_from(AuditLog)).one()
    logs = session.exec(query.offset((page-1)*per_page).limit(per_page)).all()
    return AuditLogList(logs=logs, total=total, page=page, per_page=per_page)


# ============== MATRIX INTEGRATION ==============

async def _matrix_admin_request(method: str, endpoint: str, data: dict = None):
    """Make an admin request to Matrix/Synapse"""
    url = f"{MATRIX_HS_URL}/_synapse/admin/v1{endpoint}"
    headers = {"Authorization": f"Bearer {MATRIX_ADMIN_SECRET}"}
    
    async with httpx.AsyncClient() as client:
        if method == "GET":
            response = await client.get(url, headers=headers)
        elif method == "POST":
            response = await client.post(url, headers=headers, json=data)
        elif method == "PUT":
            response = await client.put(url, headers=headers, json=data)
        elif method == "DELETE":
            response = await client.delete(url, headers=headers)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return response.status_code, response.json() if response.content else {}


@app.post("/api/v1/matrix/ban")
async def matrix_ban_user(ban_request: MatrixBanRequest, request: Request,
                          session: Session = Depends(get_session),
                          current_user: dict = Depends(require_moderator)):
    """Ban a user from Matrix rooms"""
    # Get user's Matrix ID
    user_status = session.exec(select(UserModerationStatus).where(
        UserModerationStatus.user_id == ban_request.user_id)).first()
    
    # Deactivate user in Synapse
    matrix_user_id = f"@u{ban_request.user_id}:{MATRIX_SERVER_NAME}"
    status_code, response = await _matrix_admin_request(
        "POST", f"/deactivate/{matrix_user_id}", {"erase": False}
    )
    
    create_audit_log(session, current_user["user_id"], current_user["role"], "matrix_ban",
                     "matrix_user", str(ban_request.user_id),
                     {"reason": ban_request.reason, "response": response}, request)
    
    return {"status": "ok" if status_code == 200 else "error", "matrix_response": response}


@app.post("/api/v1/matrix/message")
async def send_matrix_message(msg_request: MatrixMessageRequest, request: Request,
                              session: Session = Depends(get_session),
                              current_user: dict = Depends(require_moderator)):
    """Send a direct message to a user via Matrix (for notifications)"""
    # This would create a DM room and send a message
    # Implementation depends on your Matrix setup
    matrix_user_id = f"@u{msg_request.user_id}:{MATRIX_SERVER_NAME}"
    
    create_audit_log(session, current_user["user_id"], current_user["role"], "matrix_message",
                     "matrix_user", str(msg_request.user_id),
                     {"message_preview": msg_request.message[:100]}, request)
    
    return {"status": "ok", "message": "Message queued for delivery"}


# ============== FUND MANAGEMENT (Admin only) ==============

@app.post("/api/v1/funds/increase")
async def increase_user_funds(fund_req: FundAdjustment, request: Request,
                              session: Session = Depends(get_session),
                              current_user: dict = Depends(require_admin)):
    """Increase a user's funds (admin only)"""
    # Call wallet service
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{WALLET_SERVICE_URL}/balance/{fund_req.user_id}/increase",
            json={"amount": fund_req.amount}
        )
    
    if response.status_code == 200:
        # Log the action
        action = ModerationAction(
            moderator_user_id=current_user["user_id"],
            target_user_id=fund_req.user_id,
            action_type=ModerationActionType.INCREASE_FUNDS,
            reason=fund_req.reason,
            details=json.dumps({"amount": fund_req.amount}),
            related_dispute_id=fund_req.related_dispute_id
        )
        session.add(action)
        session.commit()
        
        create_notification(session, fund_req.user_id, NotificationType.FUNDS_ADJUSTED,
                           "Funds Added", f"{fund_req.amount} XMR has been added to your account",
                           related_action_id=action.id)
        
        create_audit_log(session, current_user["user_id"], current_user["role"], "increase_funds",
                         "user", str(fund_req.user_id),
                         {"amount": fund_req.amount, "reason": fund_req.reason}, request)
    
    return {"status": "ok" if response.status_code == 200 else "error", 
            "wallet_response": response.json() if response.content else {}}


@app.post("/api/v1/funds/freeze")
async def freeze_user_funds(freeze_req: FundFreeze, request: Request,
                            session: Session = Depends(get_session),
                            current_user: dict = Depends(require_moderator)):
    """Freeze a user's funds"""
    status = get_or_create_user_status(session, freeze_req.user_id)
    status.funds_frozen = True
    if freeze_req.amount:
        status.frozen_amount = freeze_req.amount
    
    action = ModerationAction(
        moderator_user_id=current_user["user_id"],
        target_user_id=freeze_req.user_id,
        action_type=ModerationActionType.FREEZE_FUNDS,
        reason=freeze_req.reason,
        details=json.dumps({"amount": freeze_req.amount}),
        related_dispute_id=freeze_req.related_dispute_id
    )
    session.add(action)
    session.commit()
    
    create_notification(session, freeze_req.user_id, NotificationType.FUNDS_FROZEN,
                       "Funds Frozen", f"Your funds have been frozen. Reason: {freeze_req.reason}",
                       related_action_id=action.id)
    
    create_audit_log(session, current_user["user_id"], current_user["role"], "freeze_funds",
                     "user", str(freeze_req.user_id),
                     {"amount": freeze_req.amount, "reason": freeze_req.reason}, request)
    
    return {"status": "ok"}


@app.post("/api/v1/funds/unfreeze")
async def unfreeze_user_funds(user_id: int = Body(..., embed=True), reason: str = Body(..., embed=True),
                              request: Request = None, session: Session = Depends(get_session),
                              current_user: dict = Depends(require_moderator)):
    """Unfreeze a user's funds"""
    status = get_or_create_user_status(session, user_id)
    status.funds_frozen = False
    status.frozen_amount = 0.0
    
    action = ModerationAction(
        moderator_user_id=current_user["user_id"],
        target_user_id=user_id,
        action_type=ModerationActionType.UNFREEZE_FUNDS,
        reason=reason
    )
    session.add(action)
    session.commit()
    
    create_notification(session, user_id, NotificationType.FUNDS_UNFROZEN,
                       "Funds Unfrozen", "Your funds have been unfrozen",
                       related_action_id=action.id)
    
    create_audit_log(session, current_user["user_id"], current_user["role"], "unfreeze_funds",
                     "user", str(user_id), {"reason": reason}, request)
    
    return {"status": "ok"}
