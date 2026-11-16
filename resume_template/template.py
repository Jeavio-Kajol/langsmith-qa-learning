"""The single target template every resume must be converted into."""
import re
from typing import List, Optional

from pydantic import BaseModel, Field

DATE_PATTERN = re.compile(r"^(\d{4}(-(0[1-9]|1[0-2]))?|Present)$")

# Fixed section order of the rendered template.
HEADINGS = ["Summary", "Skills", "Experience", "Education"]


class Job(BaseModel):
    company: str
    title: str
    start: str = Field(description="YYYY-MM, YYYY, or Present")
    end: str = Field(description="YYYY-MM, YYYY, or Present")
    bullets: List[str] = Field(default_factory=list)


class Education(BaseModel):
    institution: str
    degree: str
    year: Optional[str] = None


class ResumeTemplate(BaseModel):
    name: str
    email: str
    phone: str = ""
    location: str = ""
    summary: str = ""
    skills: List[str] = Field(default_factory=list)
    experience: List[Job] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)


def render_markdown(t: ResumeTemplate) -> str:
    """Deterministic rendering: same headings, same order, for every resume."""
    contact = " | ".join(x for x in (t.email, t.phone, t.location) if x)
    lines = [f"# {t.name}", contact, "", "## Summary", t.summary, "", "## Skills"]
    lines += [f"- {s}" for s in t.skills]
    lines += ["", "## Experience"]
    for j in t.experience:
        lines += [f"### {j.title} - {j.company} ({j.start} to {j.end})"]
        lines += [f"- {b}" for b in j.bullets]
    lines += ["", "## Education"]
    for e in t.education:
        lines.append(f"- {e.degree}, {e.institution}" + (f" ({e.year})" if e.year else ""))
    return "\n".join(lines) + "\n"
