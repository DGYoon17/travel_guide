from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class RequesterLevel(str, Enum):
    STAFF = "staff"
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"


class PurchaseCategory(str, Enum):
    OFFICE_SUPPLIES = "office_supplies"
    SOFTWARE = "software"
    EQUIPMENT = "equipment"
    MARKETING_SERVICE = "marketing_service"
    PROFESSIONAL_SERVICE = "professional_service"
    TRAINING = "training"
    OTHER = "other"


class SupplierType(str, Enum):
    EXISTING = "existing_vendor"
    NEW = "new_vendor"


class UrgencyLevel(str, Enum):
    NORMAL = "normal"
    URGENT = "urgent"
    CRITICAL = "critical"


class ProcurementRequest(BaseModel):
    request_id: str
    requester_id: str
    requester_level: RequesterLevel
    department: str
    category: PurchaseCategory
    item_name: str
    quantity: int = Field(1, ge=1, le=9999)
    unit_price_cny: Decimal = Field(default=Decimal("0"), ge=0)
    needed_by: date
    reason: str
    budget_code: Optional[str] = None
    budget_ceiling_cny: Optional[Decimal] = None
    supplier_name: Optional[str] = None
    supplier_type: SupplierType = SupplierType.EXISTING
    single_source: bool = False
    urgency: UrgencyLevel = UrgencyLevel.NORMAL
    contract_months: int = Field(1, ge=1, le=60)
    extra_notes: Optional[str] = None


class ProcurementLine(BaseModel):
    line_id: str
    item_name: str
    category: PurchaseCategory
    quantity: int
    unit_price_cny: Decimal
    subtotal_cny: Decimal
    supplier_name: Optional[str] = None


class ApprovalStep(BaseModel):
    role: str
    reason: str
    required: bool = True


class PurchaseRequisition(BaseModel):
    requisition_id: str
    request_id: str
    title: str
    department: str
    line_items: list[ProcurementLine] = Field(default_factory=list)
    estimated_total_cny: Decimal = Decimal("0")
    approval_path: list[ApprovalStep] = Field(default_factory=list)
    compliance_warnings: list[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None


class ApprovalThresholdRule(BaseModel):
    max_auto_approve_cny: Decimal
    finance_review_threshold_cny: Decimal
    executive_review_threshold_cny: Decimal


class CategoryRule(BaseModel):
    need_multi_quote_above_cny: Decimal
    requires_security_review: bool = False
    requires_legal_review: bool = False


class ProcurementPolicy(BaseModel):
    policy_id: str
    company_id: str
    effective_from: date
    approval_rules: dict[RequesterLevel, ApprovalThresholdRule] = Field(default_factory=dict)
    category_rules: dict[PurchaseCategory, CategoryRule] = Field(default_factory=dict)
    urgent_lead_days_min: int = 3
    budget_owner_review_threshold_cny: Decimal = Decimal("3000")
    new_vendor_review_required: bool = True
    sole_source_reason_required_above_cny: Decimal = Decimal("8000")


class ApprovalCaseStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalCase(BaseModel):
    approval_id: str
    request_id: str
    requisition_id: str
    status: ApprovalCaseStatus
    total_amount_cny: Decimal
    current_stage: str
    submitted_at: Optional[datetime] = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
