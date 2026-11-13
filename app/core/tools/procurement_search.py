"""Procurement catalog / supplier search tools."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class SearchMode(str, Enum):
    CATALOG = "catalog"
    SUPPLIER = "supplier"


class CatalogSearchRequest(BaseModel):
    keyword: str
    category: str | None = None
    max_budget_cny: int | None = None


class SupplierSearchRequest(BaseModel):
    keyword: str
    category: str | None = None
    city: str | None = None
    preferred_existing_vendor: bool = True


async def search_catalog(req: CatalogSearchRequest) -> dict[str, Any]:
    results = [
        {
            "sku": "NB-14-001",
            "item_name": "14寸商务笔记本",
            "category": req.category or "equipment",
            "price_cny": 6299,
            "supplier": "华北集采供应商A",
        },
        {
            "sku": "SAAS-001",
            "item_name": "协同办公 SaaS 年度订阅",
            "category": req.category or "software",
            "price_cny": 18800,
            "supplier": "云服务合作商B",
        },
    ]
    if req.max_budget_cny is not None:
        results = [r for r in results if r["price_cny"] <= req.max_budget_cny]
    return {
        "mode": SearchMode.CATALOG.value,
        "query": req.model_dump(mode="json"),
        "results": results,
    }


async def search_suppliers(req: SupplierSearchRequest) -> dict[str, Any]:
    return {
        "mode": SearchMode.SUPPLIER.value,
        "query": req.model_dump(mode="json"),
        "results": [
            {
                "supplier_id": "VEN-001",
                "supplier_name": "华北集采供应商A",
                "category": req.category or "equipment",
                "city": req.city or "北京",
                "cooperation_status": "existing" if req.preferred_existing_vendor else "candidate",
                "rating": 4.7,
            },
            {
                "supplier_id": "VEN-002",
                "supplier_name": "云服务合作商B",
                "category": req.category or "software",
                "city": req.city or "上海",
                "cooperation_status": "existing",
                "rating": 4.8,
            }
        ],
    }
