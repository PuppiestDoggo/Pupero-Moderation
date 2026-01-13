from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from app.models import (
    ReportCategory, ReportStatus, ContentType, ModerationActionType,
    DisputeStatus, AppealStatus, NotificationType, NotificationChannel
)


# ============== REPORT SCHEMAS ==============

class ReportCreate(BaseModel):
    reported_user_id: Optional[int] = None
    content_type: ContentType
    content_id: Optional[str] = None
    category: ReportCategory
    description: str = Field(min_length=10, max_length=2000)
    evidence: Optional[str] = None


class ReportUpdate(BaseModel):
    status: Optional[ReportStatus] = None
    assigned_moderator_id: Optional[int] = None
    resolution_notes: Optional[str] = None


class ReportRead(BaseModel):
    id: int
    reporter_user_id: int
    reported_user_id: Optional[int]
    content_type: ContentType
    content_id: Optional[str]
    category: ReportCategory
    description: str
    evidence: Optional[str]
    status: ReportStatus
    assigned_moderator_id: Optional[int]
    resolution_notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime]

    class Config:
        from_attributes = True


class ReportList(BaseModel):
    reports: List[ReportRead]
    total: int
    page: int
    per_page: int


# ============== MODERATION ACTION SCHEMAS ==============

class ModerationActionCreate(BaseModel):
    target_user_id: Optional[int] = None
    action_type: ModerationActionType
    content_type: Optional[ContentType] = None
    content_id: Optional[str] = None
    reason: str = Field(min_length=1, max_length=2000)
    details: Optional[str] = None
    related_report_id: Optional[int] = None
    related_dispute_id: Optional[int] = None
    duration_minutes: Optional[int] = None  # For mute/suspend
    amount: Optional[float] = None  # For fund adjustments


class ModerationActionRead(BaseModel):
    id: int
    moderator_user_id: int
    target_user_id: Optional[int]
    action_type: ModerationActionType
    content_type: Optional[ContentType]
    content_id: Optional[str]
    reason: str
    details: Optional[str]
    related_report_id: Optional[int]
    related_dispute_id: Optional[int]
    duration_minutes: Optional[int]
    expires_at: Optional[datetime]
    is_active: bool
    reversed_by_action_id: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


class ModerationActionList(BaseModel):
    actions: List[ModerationActionRead]
    total: int
    page: int
    per_page: int


# ============== USER MODERATION STATUS SCHEMAS ==============

class UserModerationStatusRead(BaseModel):
    id: int
    user_id: int
    is_banned: bool
    is_muted: bool
    is_suspended: bool
    funds_frozen: bool
    frozen_amount: float
    warning_count: int
    ban_reason: Optional[str]
    mute_reason: Optional[str]
    suspend_reason: Optional[str]
    mute_expires_at: Optional[datetime]
    suspend_expires_at: Optional[datetime]
    last_action_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UserModerationCheck(BaseModel):
    """Quick check response for other services"""
    user_id: int
    can_login: bool
    can_trade: bool
    can_chat: bool
    is_banned: bool
    is_muted: bool
    is_suspended: bool
    funds_frozen: bool
    message: Optional[str] = None


# ============== DISPUTE SCHEMAS ==============

class DisputeCreate(BaseModel):
    transaction_id: int
    offer_id: int
    category: ReportCategory
    description: str = Field(min_length=10, max_length=5000)
    evidence: Optional[str] = None


class DisputeUpdate(BaseModel):
    status: Optional[DisputeStatus] = None
    assigned_moderator_id: Optional[int] = None
    buyer_statement: Optional[str] = None
    seller_statement: Optional[str] = None
    evidence_buyer: Optional[str] = None
    evidence_seller: Optional[str] = None
    resolution_notes: Optional[str] = None


class DisputeResolve(BaseModel):
    status: DisputeStatus  # Must be one of resolved statuses
    resolution_notes: str = Field(min_length=10, max_length=5000)
    resolution_amount_to_buyer: Optional[float] = None
    resolution_amount_to_seller: Optional[float] = None


class DisputeRead(BaseModel):
    id: int
    transaction_id: int
    offer_id: int
    buyer_id: int
    seller_id: int
    initiated_by_user_id: int
    status: DisputeStatus
    category: ReportCategory
    description: str
    buyer_statement: Optional[str]
    seller_statement: Optional[str]
    evidence_buyer: Optional[str]
    evidence_seller: Optional[str]
    assigned_moderator_id: Optional[int]
    resolution_notes: Optional[str]
    resolution_amount_to_buyer: Optional[float]
    resolution_amount_to_seller: Optional[float]
    funds_frozen_amount: float
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime]

    class Config:
        from_attributes = True


class DisputeList(BaseModel):
    disputes: List[DisputeRead]
    total: int
    page: int
    per_page: int


# ============== DISPUTE MESSAGE SCHEMAS ==============

class DisputeMessageCreate(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    attachments: Optional[str] = None


class DisputeMessageRead(BaseModel):
    id: int
    dispute_id: int
    sender_user_id: int
    is_moderator: bool
    message: str
    attachments: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ============== APPEAL SCHEMAS ==============

class AppealCreate(BaseModel):
    moderation_action_id: int
    reason: str = Field(min_length=10, max_length=5000)
    evidence: Optional[str] = None


class AppealUpdate(BaseModel):
    status: Optional[AppealStatus] = None
    assigned_moderator_id: Optional[int] = None
    resolution_notes: Optional[str] = None


class AppealRead(BaseModel):
    id: int
    user_id: int
    moderation_action_id: int
    status: AppealStatus
    reason: str
    evidence: Optional[str]
    assigned_moderator_id: Optional[int]
    resolution_notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime]

    class Config:
        from_attributes = True


class AppealList(BaseModel):
    appeals: List[AppealRead]
    total: int
    page: int
    per_page: int


# ============== AUDIT LOG SCHEMAS ==============

class AuditLogRead(BaseModel):
    id: int
    actor_user_id: int
    actor_role: str
    action: str
    target_type: str
    target_id: Optional[str]
    details: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class AuditLogList(BaseModel):
    logs: List[AuditLogRead]
    total: int
    page: int
    per_page: int


# ============== NOTIFICATION SCHEMAS ==============

class NotificationRead(BaseModel):
    id: int
    user_id: int
    notification_type: NotificationType
    title: str
    message: str
    channel: NotificationChannel
    related_action_id: Optional[int]
    related_report_id: Optional[int]
    related_dispute_id: Optional[int]
    related_appeal_id: Optional[int]
    is_read: bool
    sent_in_app: bool
    sent_matrix: bool
    created_at: datetime
    read_at: Optional[datetime]

    class Config:
        from_attributes = True


class NotificationList(BaseModel):
    notifications: List[NotificationRead]
    total: int
    unread_count: int


# ============== MODERATION QUEUE SCHEMAS ==============

class QueueItemRead(BaseModel):
    id: int
    item_type: str
    item_id: int
    priority: int
    assigned_moderator_id: Optional[int]
    status: str
    created_at: datetime
    claimed_at: Optional[datetime]
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class QueueList(BaseModel):
    items: List[QueueItemRead]
    total: int
    pending_count: int
    in_progress_count: int


# ============== STATISTICS SCHEMAS ==============

class ModerationStatsRead(BaseModel):
    id: int
    date: datetime
    total_reports: int
    resolved_reports: int
    total_disputes: int
    resolved_disputes: int
    total_appeals: int
    approved_appeals: int
    denied_appeals: int
    warnings_issued: int
    mutes_issued: int
    bans_issued: int
    suspensions_issued: int
    content_removed: int
    funds_frozen_total: float
    created_at: datetime

    class Config:
        from_attributes = True


class DashboardStats(BaseModel):
    """Aggregated stats for dashboard"""
    pending_reports: int
    pending_disputes: int
    pending_appeals: int
    queue_size: int
    active_bans: int
    active_mutes: int
    active_suspensions: int
    today_actions: int
    today_resolved: int
    most_reported_users: List[dict]
    recent_actions: List[ModerationActionRead]


# ============== FUND MANAGEMENT SCHEMAS ==============

class FundAdjustment(BaseModel):
    user_id: int
    amount: float = Field(gt=0)
    reason: str = Field(min_length=1, max_length=2000)
    related_dispute_id: Optional[int] = None


class FundFreeze(BaseModel):
    user_id: int
    amount: Optional[float] = None  # None = freeze all
    reason: str = Field(min_length=1, max_length=2000)
    related_dispute_id: Optional[int] = None


# ============== MATRIX INTEGRATION SCHEMAS ==============

class MatrixBanRequest(BaseModel):
    user_id: int
    reason: str
    room_ids: Optional[List[str]] = None  # None = all rooms


class MatrixKickRequest(BaseModel):
    user_id: int
    reason: str
    room_ids: Optional[List[str]] = None


class MatrixMessageRequest(BaseModel):
    user_id: int
    message: str


# ============== BULK OPERATIONS ==============

class BulkActionRequest(BaseModel):
    user_ids: List[int]
    action_type: ModerationActionType
    reason: str = Field(min_length=1, max_length=2000)
    duration_minutes: Optional[int] = None


class BulkActionResponse(BaseModel):
    success_count: int
    failed_count: int
    failed_user_ids: List[int]
    errors: List[str]
