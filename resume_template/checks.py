"""QA checks that decide whether a conversion into the template is correct.

Two kinds of check:
- Universal (need only the source resume + converted output): structure, formats,
  grounding (nothing invented), completeness (nothing dropped), ordering.
- Golden (need a hand-written expected result): exact field comparison.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

from resume_template.template import DATE_PATTERN, HEADINGS, ResumeTemplate, render_markdown

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
SKILL_RECALL_MIN = 0.8


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _in_source(value: str, source: str) -> bool:
    return _norm(value) in _norm(source)


def _sort_key(date: str) -> str:
    return "9999" if date == "Present" else date


def _check(name: str, problems: List[str]) -> Check:
    return Check(name, not problems, "; ".join(problems))


def run_checks(source: str, t: ResumeTemplate, golden: Optional[dict] = None) -> List[Check]:
    checks: List[Check] = []

    # 1. Template structure: fixed headings, once each, in order.
    rendered = render_markdown(t)
    found = re.findall(r"^## (.+)$", rendered, re.MULTILINE)
    checks.append(_check("structure", [] if found == HEADINGS else [f"headings {found} != {HEADINGS}"]))

    # 2. Required fields.
    problems = [f"missing {f}" for f in ("name", "email") if not getattr(t, f).strip()]
    if not t.experience:
        problems.append("no experience entries")
    checks.append(_check("required_fields", problems))

    # 3. Value formats (email, dates).
    problems = []
    if t.email and not EMAIL_RE.fullmatch(t.email):
        problems.append(f"bad email {t.email!r}")
    for j in t.experience:
        for label, d in (("start", j.start), ("end", j.end)):
            if not DATE_PATTERN.match(d):
                problems.append(f"{j.company}: bad {label} date {d!r}")
        if j.start != "Present" and j.end != "Present" and DATE_PATTERN.match(j.start) and DATE_PATTERN.match(j.end) and j.start > j.end:
            problems.append(f"{j.company}: start {j.start} after end {j.end}")
    checks.append(_check("formats", problems))

    # 4. Grounding: every extracted fact must appear in the source (no hallucination).
    problems = []
    if t.name and not _in_source(t.name, source):
        problems.append(f"name {t.name!r} not in source")
    if t.email and t.email.lower() not in source.lower():
        problems.append(f"email {t.email!r} not in source")
    if t.phone and _digits(t.phone) not in _digits(source):
        problems.append(f"phone {t.phone!r} not in source")
    for j in t.experience:
        for label, v in (("company", j.company), ("title", j.title)):
            if not _in_source(v, source):
                problems.append(f"{label} {v!r} not in source")
    for e in t.education:
        if not _in_source(e.institution, source):
            problems.append(f"institution {e.institution!r} not in source")
    for s in t.skills:
        if not _in_source(s, source):
            problems.append(f"skill {s!r} not in source")
    checks.append(_check("grounding", problems))

    # 5. Completeness: contact details present in the source must not be dropped.
    problems = []
    for email in EMAIL_RE.findall(source):
        if email.lower() not in rendered.lower():
            problems.append(f"email {email!r} dropped")
    source_phones = [p for p in PHONE_RE.findall(source) if len(_digits(p)) >= 10]
    if source_phones and not t.phone:
        problems.append("phone present in source but empty in output")
    checks.append(_check("completeness", problems))

    # 6. Ordering: experience newest first.
    keys = [_sort_key(j.start) for j in t.experience]
    checks.append(_check("experience_order", [] if keys == sorted(keys, reverse=True) else [f"not newest first: {keys}"]))

    if golden is not None:
        checks.extend(_golden_checks(t, golden))
    return checks


def _golden_checks(t: ResumeTemplate, g: dict) -> List[Check]:
    out: List[Check] = []
    problems = []
    if _norm(t.name) != _norm(g["name"]):
        problems.append(f"name {t.name!r} != {g['name']!r}")
    if t.email.lower() != g["email"].lower():
        problems.append(f"email {t.email!r} != {g['email']!r}")
    if _digits(t.phone) != g["phone_digits"]:
        problems.append(f"phone digits {_digits(t.phone)!r} != {g['phone_digits']!r}")
    out.append(_check("golden_contact", problems))

    problems = []
    actual = {_norm(j.company): j for j in t.experience}
    expected = {_norm(j["company"]): j for j in g["experience"]}
    for key in expected.keys() - actual.keys():
        problems.append(f"missing job at {expected[key]['company']!r}")
    for key in actual.keys() - expected.keys():
        problems.append(f"unexpected job at {actual[key].company!r}")
    for key in expected.keys() & actual.keys():
        a, e = actual[key], expected[key]
        for field in ("title", "start", "end"):
            if _norm(getattr(a, field)) != _norm(e[field]):
                problems.append(f"{e['company']}: {field} {getattr(a, field)!r} != {e[field]!r}")
    out.append(_check("golden_experience", problems))

    expected_skills = {_norm(s) for s in g["skills"]}
    got = [_norm(s) for s in t.skills]
    # "Python (pandas)" still counts as "Python": match on whole words, not exact equality.
    missing = {e for e in expected_skills if not any(re.search(rf"(?<!\w){re.escape(e)}(?!\w)", g) for g in got)}
    recall = 1 - len(missing) / len(expected_skills) if expected_skills else 1.0
    out.append(Check("golden_skills", recall >= SKILL_RECALL_MIN, f"recall {recall:.2f}, missing {sorted(missing)}"))

    problems = []
    for e in g["education"]:
        match = [x for x in t.education if _norm(x.institution) == _norm(e["institution"])]
        if not match:
            problems.append(f"missing education {e['institution']!r}")
        elif (match[0].year or "") != e["year"]:
            problems.append(f"{e['institution']}: year {match[0].year!r} != {e['year']!r}")
    out.append(_check("golden_education", problems))
    return out
