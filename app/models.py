from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field
from sqlalchemy.sql import func
from enum import Enum


# ============== ENUMS ==============

class ReportCategory(str, Enum):
    SCAM = "scam"
    HARASSMENT = "harassment"
    SPAM = "spam"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    FRAUD = "fraud"
    IMPERSONATION = "impersonation"
    ILLEGAL_ACTIVITY = "illegal_activity"
    OTHER = "other"


class ReportStatus(str, Enum):
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class ContentType(str, Enum):
    USER = "user"
    OFFER = "offer"
    REVIEW = "review"
    CHAT_MESSAGE = "chat_message"
    PROFILE = "profile"
    TRANSACTION = "transaction"


class ModerationActionType(str, Enum):
    WARN = "warn"
    MUTE = "mute"
    UNMUTE = "unmute"
    BAN = "ban"
    UNBAN = "unban"
    SUSPEND = "suspend"
    UNSUSPEND = "unsuspend"
    DELETE_CONTENT = "delete_content"
    HIDE_CONTENT = "hide_content"
    RESTORE_CONTENT = "restore_content"
    EDIT_CONTENT = "edit_content"
    FREEZE_FUNDS = "freeze_funds"
    UNFREEZE_FUNDS = "unfreeze_funds"
    INCREASE_FUNDS = "increase_funds"
    DECREASE_FUNDS = "decrease_funds"
    KICK_FROM_MATRIX = "kick_from_matrix"
    BAN_FROM_MATRIX = "ban_from_matrix"


class DisputeStatus(str, Enum):
    OPEN = "open"
    UNDER_REVIEW = "under_review"
    AWAITING_RESPONSE = "awaiting_response"
    RESOLVED_BUYER_FAVOR = "resolved_buyer_favor"
    RESOLVED_SELLER_FAVOR = "resolved_seller_favor"
    RESOLVED_SPLIT = "resolved_split"
    CANCELLED = "cancelled"


class AppealStatus(str, Enum):
    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    DENIED = "denied"


class NotificationType(str, Enum):
    WARNING_RECEIVED = "warning_received"
    MUTED = "muted"
    UNMUTED = "unmuted"
    BANNED = "banned"
    UNBANNED = "unbanned"
    SUSPENDED = "suspended"
    UNSUSPENDED = "unsuspended"
    CONTENT_REMOVED = "content_removed"
    CONTENT_RESTORED = "content_restored"
    REPORT_RESOLVED = "report_resolved"
    DISPUTE_OPENED = "dispute_opened"
    DISPUTE_UPDATE = "dispute_update"
    DISPUTE_RESOLVED = "dispute_resolved"
    APPEAL_SUBMITTED = "appeal_submitted"
    APPEAL_RESOLVED = "appeal_resolved"
    FUNDS_FROZEN = "funds_frozen"
    FUNDS_UNFROZEN = "funds_unfrozen"
    FUNDS_ADJUSTED = "funds_adjusted"


class NotificationChannel(str, Enum):
    IN_APP = "in_app"
    MATRIX = "matrix"
    BOTH = "both"


# ============== DATABASE MODELS ==============

class Report(SQLModel, table=True):
    """User reports against other users or content"""
    __tablename__ = "report"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    reporter_user_id: int = Field(index=True)  # User who submitted the report
    reported_user_id: Optional[int] = Field(default=None, index=True)  # User being reported (if applicable)
    content_type: ContentType = Field(index=True)  # Type of content being reported
    content_id: Optional[str] = Field(default=None, max_length=64, index=True)  # ID of the content (offer_id, review_id, etc.)
    category: ReportCategory = Field(index=True)
    description: str = Field(max_length=2048)  # Reporter's description of the issue
    evidence: Optional[str] = Field(default=None, max_length=2048)  # URLs or references to evidence
    status: ReportStatus = Field(default=ReportStatus.PENDING, index=True)
    assigned_moderator_id: Optional[int] = Field(default=None, index=True)
    resolution_notes: Optional[str] = Field(default=None, max_length=2048)
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.current_timestamp()})
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.current_timestamp(), "onupdate": func.current_timestamp()})
    resolved_at: Optional[datetime] = Field(default=None)


class ModerationAction(SQLModel, table=True):
    """Log of all moderation actions taken"""
    __tablename__ = "moderation_action"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    moderator_user_id: int = Field(index=True)  # Moderator who took the action
    target_user_id: Optional[int] = Field(default=None, index=True)  # User affected by the action
    action_type: ModerationActionType = Field(index=True)
    content_type: Optional[ContentType] = Field(default=None)  # Type of content affected
    content_id: Optional[str] = Field(default=None, max_length=64)  # ID of content affected
    reason: str = Field(max_length=1024)  # Moderator's reason for the action
    details: Optional[str] = Field(default=None, max_length=4096)  # Additional details (JSON string for complex data)
    related_report_id: Optional[int] = Field(default=None, index=True)  # Related report if any
    related_dispute_id: Optional[int] = Field(default=None, index=True)  # Related dispute if any
    duration_minutes: Optional[int] = Field(default=None)  # For temporary actions (mute, suspend)
    expires_at: Optional[datetime] = Field(default=None, index=True)  # When temporary action expires
    is_active: bool = Field(default=True, index=True)  # Whether action is currently active
    reversed_by_action_id: Optional[int] = Field(default=None)  # If this action was reversed
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.current_timestamp()})


class UserModerationStatus(SQLModel, table=True):
    """Current moderation status for each user"""
    __tablename__ = "user_moderation_status"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(unique=True, index=True)
    is_banned: bool = Field(default=False, index=True)
    is_muted: bool = Field(default=False, index=True)
    is_suspended: bool = Field(default=False, index=True)
    funds_frozen: bool = Field(default=False, index=True)
    frozen_amount: float = Field(default=0.0)
    warning_count: int = Field(default=0)
    ban_reason: Optional[str] = Field(default=None, max_length=1024)
    mute_reason: Optional[str] = Field(default=None, max_length=1024)
    suspend_reason: Optional[str] = Field(default=None, max_length=1024)
    mute_expires_at: Optional[datetime] = Field(default=None, index=True)
    suspend_expires_at: Optional[datetime] = Field(default=None, index=True)
    last_action_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.current_timestamp()})
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.current_timestamp(), "onupdate": func.current_timestamp()})


class Dispute(SQLModel, table=True):
    """Trade disputes between buyers and sellers"""
    __tablename__ = "dispute"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    transaction_id: int = Field(index=True)  # Related transaction
    offer_id: int = Field(index=True)  # Related offer
    buyer_id: int = Field(index=True)
    seller_id: int = Field(index=True)
    initiated_by_user_id: int = Field(index=True)  # Who opened the dispute
    status: DisputeStatus = Field(default=DisputeStatus.OPEN, index=True)
    category: ReportCategory = Field(index=True)
    description: str = Field(max_length=2048)  # Initial dispute description
    buyer_statement: Optional[str] = Field(default=None, max_length=2048)
    seller_statement: Optional[str] = Field(default=None, max_length=2048)
    evidence_buyer: Optional[str] = Field(default=None, max_length=2048)  # Evidence provided by buyer
    evidence_seller: Optional[str] = Field(default=None, max_length=2048)  # Evidence provided by seller
    assigned_moderator_id: Optional[int] = Field(default=None, index=True)
    resolution_notes: Optional[str] = Field(default=None, max_length=2048)
    resolution_amount_to_buyer: Optional[float] = Field(default=None)  # Amount refunded to buyer
    resolution_amount_to_seller: Optional[float] = Field(default=None)  # Amount released to seller
    funds_frozen_amount: float = Field(default=0.0)  # Amount frozen during dispute
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.current_timestamp()})
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.current_timestamp(), "onupdate": func.current_timestamp()})
    resolved_at: Optional[datetime] = Field(default=None)


class DisputeMessage(SQLModel, table=True):
    """Messages within a dispute thread"""
    __tablename__ = "dispute_message"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    dispute_id: int = Field(index=True)
    sender_user_id: int = Field(index=True)
    is_moderator: bool = Field(default=False)
    message: str = Field(max_length=4096)
    attachments: Optional[str] = Field(default=None, max_length=2048)  # JSON array of attachment URLs
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.current_timestamp()})


class Appeal(SQLModel, table=True):
    """Appeals against moderation decisions"""
    __tablename__ = "appeal"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)  # User appealing
    moderation_action_id: int = Field(index=True)  # Action being appealed
    status: AppealStatus = Field(default=AppealStatus.PENDING, index=True)
    reason: str = Field(max_length=2048)  # User's reason for appeal
    evidence: Optional[str] = Field(default=None, max_length=2048)  # Supporting evidence
    assigned_moderator_id: Optional[int] = Field(default=None, index=True)
    resolution_notes: Optional[str] = Field(default=None, max_length=2048)
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.current_timestamp()})
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.current_timestamp(), "onupdate": func.current_timestamp()})
    resolved_at: Optional[datetime] = Field(default=None)


class AuditLog(SQLModel, table=True):
    """Comprehensive audit log for all moderation activities"""
    __tablename__ = "audit_log"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    actor_user_id: int = Field(index=True)  # Who performed the action
    actor_role: str = Field(max_length=32, index=True)  # Role at time of action (admin, moderator)
    action: str = Field(max_length=64, index=True)  # Action performed
    target_type: str = Field(max_length=32, index=True)  # Type of target (user, report, dispute, etc.)
    target_id: Optional[str] = Field(default=None, max_length=64, index=True)  # ID of target
    details: str = Field(max_length=4096)  # JSON string with full details
    ip_address: Optional[str] = Field(default=None, max_length=45)
    user_agent: Optional[str] = Field(default=None, max_length=512)
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.current_timestamp()})


class Notification(SQLModel, table=True):
    """Notifications sent to users about moderation actions"""
    __tablename__ = "notification"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    notification_type: NotificationType = Field(index=True)
    title: str = Field(max_length=255)
    message: str = Field(max_length=2048)
    channel: NotificationChannel = Field(default=NotificationChannel.BOTH)
    related_action_id: Optional[int] = Field(default=None)
    related_report_id: Optional[int] = Field(default=None)
    related_dispute_id: Optional[int] = Field(default=None)
    related_appeal_id: Optional[int] = Field(default=None)
    is_read: bool = Field(default=False, index=True)
    sent_in_app: bool = Field(default=False)
    sent_matrix: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.current_timestamp()})
    read_at: Optional[datetime] = Field(default=None)


class ModerationQueue(SQLModel, table=True):
    """Queue of items pending moderation review"""
    __tablename__ = "moderation_queue"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    item_type: str = Field(max_length=32, index=True)  # report, dispute, appeal
    item_id: int = Field(index=True)
    priority: int = Field(default=0, index=True)  # Higher = more urgent
    assigned_moderator_id: Optional[int] = Field(default=None, index=True)
    status: str = Field(default="pending", max_length=32, index=True)  # pending, in_progress, completed
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.current_timestamp()})
    claimed_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)


class ModerationStats(SQLModel, table=True):
    """Daily statistics for moderation dashboard"""
    __tablename__ = "moderation_stats"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    date: datetime = Field(index=True)
    total_reports: int = Field(default=0)
    resolved_reports: int = Field(default=0)
    total_disputes: int = Field(default=0)
    resolved_disputes: int = Field(default=0)
    total_appeals: int = Field(default=0)
    approved_appeals: int = Field(default=0)
    denied_appeals: int = Field(default=0)
    warnings_issued: int = Field(default=0)
    mutes_issued: int = Field(default=0)
    bans_issued: int = Field(default=0)
    suspensions_issued: int = Field(default=0)
    content_removed: int = Field(default=0)
    funds_frozen_total: float = Field(default=0.0)
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.current_timestamp()})
