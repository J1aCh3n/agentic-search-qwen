from __future__ import annotations

import json
from datetime import date
from typing import Any


CURRENT_YEAR = date.today().year


def _y(month_day: str) -> str:
    _, month, day = month_day.split("-")
    return f"{CURRENT_YEAR}-{month}-{day}"


class EnterpriseSDK:
    """Small in-memory simulator for enterprise systems used by the demo agent."""

    def __init__(self) -> None:
        self.systems = self._build_systems()

    def _build_systems(self) -> dict[str, Any]:
        employees = {
            "E001": {
                "name": "Alex Chen",
                "department": "Engineering",
                "position": "Senior Backend Engineer",
                "manager": "Maya Patel",
                "email": "alex.chen@example.com",
            },
            "E002": {
                "name": "Nina Li",
                "department": "Product",
                "position": "Product Manager",
                "manager": "Owen Wright",
                "email": "nina.li@example.com",
            },
            "E003": {
                "name": "Maya Patel",
                "department": "Engineering",
                "position": "Engineering Manager",
                "manager": "Grace Kim",
                "email": "maya.patel@example.com",
            },
            "E004": {
                "name": "Jordan Smith",
                "department": "Marketing",
                "position": "Marketing Lead",
                "manager": "Owen Wright",
                "email": "jordan.smith@example.com",
            },
        }

        transactions = [
            {
                "id": f"TXN-{CURRENT_YEAR}-001",
                "type": "income",
                "amount": 580000,
                "date": _y("2024-01-15"),
                "counterpart": "Northstar Analytics",
                "account": "Primary Operating Account",
                "category": "subscription",
                "description": "Annual platform subscription",
                "status": "settled",
            },
            {
                "id": f"TXN-{CURRENT_YEAR}-002",
                "type": "expense",
                "amount": 260000,
                "date": _y("2024-02-06"),
                "counterpart": "CloudHub",
                "account": "Primary Operating Account",
                "category": "cloud services",
                "description": "Compute and storage services",
                "status": "settled",
            },
            {
                "id": f"TXN-{CURRENT_YEAR}-003",
                "type": "expense",
                "amount": 180000,
                "date": _y("2024-05-15"),
                "counterpart": "AdBridge",
                "account": "Primary Operating Account",
                "category": "marketing",
                "description": "May advertising campaign",
                "status": "settled",
            },
            {
                "id": f"TXN-{CURRENT_YEAR}-004",
                "type": "income",
                "amount": 620000,
                "date": _y("2024-05-18"),
                "counterpart": "BlueBridge Networks",
                "account": "Primary Operating Account",
                "category": "professional services",
                "description": "Implementation milestone payment",
                "status": "settled",
            },
            {
                "id": f"TXN-{CURRENT_YEAR}-005",
                "type": "expense",
                "amount": 290000,
                "date": _y("2024-05-22"),
                "counterpart": "CloudHub",
                "account": "Primary Operating Account",
                "category": "cloud services",
                "description": "May cloud service fee",
                "status": "pending",
            },
        ]

        projects = {
            "P001": {
                "name": "Smart Support V2",
                "status": "in progress",
                "lead": "E003",
                "department": "Engineering",
                "start": _y("2024-01-01"),
                "deadline": _y("2024-06-30"),
                "progress": 68,
                "priority": "high",
            },
            "P002": {
                "name": "Data Platform Refresh",
                "status": "in progress",
                "lead": "E001",
                "department": "Engineering",
                "start": _y("2024-02-15"),
                "deadline": _y("2024-09-30"),
                "progress": 34,
                "priority": "high",
            },
            "P003": {
                "name": "Mobile App Redesign",
                "status": "planned",
                "lead": "E002",
                "department": "Product",
                "start": _y("2024-04-01"),
                "deadline": _y("2024-08-31"),
                "progress": 8,
                "priority": "medium",
            },
        }

        wiki = [
            {
                "title": "Remote Work Policy",
                "category": "HR policy",
                "content": "Employees may work remotely up to two days per week after manager approval.",
                "updated": _y("2024-01-10"),
            },
            {
                "title": "Expense Reimbursement Policy",
                "category": "Finance policy",
                "content": "Expenses must be submitted within 30 days. Items above 5000 require department manager approval.",
                "updated": _y("2024-02-05"),
            },
            {
                "title": "Code Review Standard",
                "category": "Engineering policy",
                "content": "Every pull request requires two reviewers. Critical modules require architecture review.",
                "updated": _y("2024-03-12"),
            },
            {
                "title": "API Design Guide",
                "category": "Engineering documentation",
                "content": "REST APIs use lowercase resource names, HTTP verbs, standard status codes, and cursor pagination.",
                "updated": _y("2024-01-25"),
            },
        ]

        return {
            "hr": {"employees": employees},
            "finance": {"transactions": transactions},
            "project": {"projects": projects},
            "wiki": {"documents": wiki},
        }

    def call(self, system: str, action: str, params: dict[str, Any] | None = None) -> str:
        params = params or {}
        if system == "hr":
            return self._call_hr(action, params)
        if system == "finance":
            return self._call_finance(action, params)
        if system == "project":
            return self._call_project(action, params)
        if system == "wiki":
            return self._call_wiki(action, params)
        return json.dumps({"error": f"Unknown system: {system}"})

    def _call_hr(self, action: str, params: dict[str, Any]) -> str:
        employees = self.systems["hr"]["employees"]
        if action == "get_employee":
            employee_id = params.get("employee_id")
            return json.dumps({"data": employees.get(employee_id)}, ensure_ascii=False, indent=2)
        if action == "search_employees":
            keyword = str(params.get("keyword", "")).lower()
            data = [
                {"id": employee_id, **employee}
                for employee_id, employee in employees.items()
                if not keyword or keyword in json.dumps(employee).lower()
            ]
            return json.dumps({"data": data}, ensure_ascii=False, indent=2)
        if action == "get_org_chart":
            chart = [
                {"employee_id": employee_id, "name": employee["name"], "manager": employee["manager"]}
                for employee_id, employee in employees.items()
            ]
            return json.dumps({"data": chart}, ensure_ascii=False, indent=2)
        return json.dumps({"error": f"Unknown HR action: {action}"})

    def _call_finance(self, action: str, params: dict[str, Any]) -> str:
        if action not in {"get_transactions", "get_payments"}:
            return json.dumps({"error": f"Unknown finance action: {action}"})

        records = list(self.systems["finance"]["transactions"])
        transaction_type = params.get("type")
        category = params.get("category")
        status = params.get("status")
        month = params.get("month")
        limit = int(params.get("limit", len(records)))

        if transaction_type:
            records = [item for item in records if item["type"] == transaction_type]
        if category:
            records = [item for item in records if category.lower() in item["category"].lower()]
        if status:
            records = [item for item in records if item["status"] == status]
        if month:
            records = [item for item in records if item["date"][5:7] == str(month).zfill(2)]

        total_income = sum(item["amount"] for item in records if item["type"] == "income")
        total_expense = sum(item["amount"] for item in records if item["type"] == "expense")
        payload = {
            "data": records[:limit],
            "summary": {
                "total_income": total_income,
                "total_expense": total_expense,
                "net_amount": total_income - total_expense,
                "count": len(records),
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _call_project(self, action: str, params: dict[str, Any]) -> str:
        projects = self.systems["project"]["projects"]
        if action == "get_project":
            project_id = params.get("project_id")
            return json.dumps({"data": projects.get(project_id)}, ensure_ascii=False, indent=2)
        if action == "search_projects":
            keyword = str(params.get("keyword", "")).lower()
            status = params.get("status")
            data = []
            for project_id, project in projects.items():
                if keyword and keyword not in json.dumps(project).lower():
                    continue
                if status and project["status"] != status:
                    continue
                data.append({"id": project_id, **project})
            return json.dumps({"data": data}, ensure_ascii=False, indent=2)
        if action == "get_milestones":
            milestones = [
                {"project_id": "P001", "name": "Requirements signed off", "date": _y("2024-02-01"), "status": "done"},
                {"project_id": "P001", "name": "Core workflow build", "date": _y("2024-04-15"), "status": "in progress"},
                {"project_id": "P002", "name": "Architecture review", "date": _y("2024-03-15"), "status": "done"},
                {"project_id": "P002", "name": "Data pipeline build", "date": _y("2024-05-30"), "status": "in progress"},
            ]
            project_id = params.get("project_id")
            if project_id:
                milestones = [item for item in milestones if item["project_id"] == project_id]
            return json.dumps({"data": milestones}, ensure_ascii=False, indent=2)
        return json.dumps({"error": f"Unknown project action: {action}"})

    def _call_wiki(self, action: str, params: dict[str, Any]) -> str:
        documents = self.systems["wiki"]["documents"]
        if action in {"search_policies", "search_tech_docs"}:
            keyword = str(params.get("keyword", "")).lower()
            data = [doc for doc in documents if not keyword or keyword in json.dumps(doc).lower()]
            return json.dumps({"data": data}, ensure_ascii=False, indent=2)
        if action == "get_all_categories":
            categories = sorted({doc["category"] for doc in documents})
            return json.dumps({"data": categories}, ensure_ascii=False, indent=2)
        return json.dumps({"error": f"Unknown wiki action: {action}"})

    def list_systems(self) -> str:
        systems = {
            "hr": {
                "description": "Human resources system",
                "actions": ["get_employee", "search_employees", "get_org_chart"],
            },
            "finance": {
                "description": "Finance system",
                "actions": ["get_transactions", "get_payments"],
            },
            "project": {
                "description": "Project management system",
                "actions": ["get_project", "search_projects", "get_milestones"],
            },
            "wiki": {
                "description": "Internal knowledge base",
                "actions": ["search_policies", "search_tech_docs", "get_all_categories"],
            },
        }
        return json.dumps(systems, ensure_ascii=False, indent=2)
