from typing import Any, List, Optional

from pydantic import BaseModel


class NotionTemplateItem(BaseModel):
    key: str
    parentKey: Optional[str]
    title: str
    content: Any


class NotionTemplatePayload(BaseModel):
    projectId: int
    templates: List[NotionTemplateItem]
