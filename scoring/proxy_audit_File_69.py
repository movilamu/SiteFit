"""
proxy_audit.py

Reviews the site-scoring criteria hierarchy (see criteria_hierarchy.py) for
criteria that could function, in practice, as a proxy for a legally or
ethically protected characteristic -- e.g. religion, caste/community, or
ethnicity -- even though no such characteristic is used directly anywhere
in the tree.

WHY THIS EXISTS
----------------
Site-scoring models built from demographic and economic data can end up
reproducing discriminatory outcomes indirectly. In the Indian context this
risk is most acute where a criterion correlates strongly, at the
neighbourhood level, with religious or caste/community composition -- for
example some income, rent, or "affluence" proxies can track historical
patterns of religious or caste segregation in urban areas, so that
screening a site in/out on that basis has a similar effect to screening on
community directly, regardless of intent.

WHAT THIS MODULE DOES AND DOES NOT DO
--------------------------------------
- It flags criteria whose data_field / label / notes suggest this kind of
  correlation risk, using domain heuristics tuned to this scoring model,
  and gives a human-readable rationale and a suggested mitigation for each.
- It does NOT compute any actual statistical correlation with protected
  characteristics (that would require the underlying demographic microdata
  and is a separate, follow-up analysis this module recommends).
- It does NOT remove, reweight, or otherwise modify any criterion. This
  module is read-only and produces a report only.

*** THIS OUTPUT MUST BE READ AND ACTED ON BY A HUMAN BEFORE DEPLOYMENT. ***
`audit_criteria_for_proxies` never mutates `criteria_hierarchy_config` and
never blocks or filters criteria automatically. It is the responsibility of
a qualified human reviewer (ideally with input from legal/compliance and
someone versed in the local social context) to decide, for each flagged
criterion, whether to keep it, drop it, or add the suggested documentation
and safeguards.

LIMITS OF AUTOMATED PROXY DETECTION
------------------------------------
Automated detection of proxy discrimination has real, structural limits:
  - It can only reason from field names, labels, and notes -- it has no
    access to the actual joint distribution of a criterion and protected
    characteristics in the areas being scored, which is what would be
    needed to *confirm* a proxy relationship.
  - Correlation strength between an economic/geographic variable and a
    protected characteristic varies by city, state, and even neighbourhood,
    and can change over time; a heuristic tuned for one metro may
    over- or under-flag in another.
  - Absence of a flag here is not a clearance. Combinations of criteria
    (e.g. several individually-mild proxies combined through AHP weights)
    can jointly reconstruct a strong proxy signal that no single-criterion
    heuristic will catch.
  - This script is a starting aid to focus human review, not a substitute
    for it, and not a compliance sign-off of any kind.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional


class RiskLevel(str, Enum):
    """Coarse triage level for a flagged criterion -- for sorting/reporting
    only. Not a certified risk score; a human reviewer sets the real
    priority."""
    HIGH = "high"        # plausible direct proxy for a protected characteristic
    MEDIUM = "medium"    # correlated in some contexts; depends on usage/weighting
    LOW = "low"          # minor or indirect concern, mainly a documentation gap


@dataclass(frozen=True)
class ProxyFlag:
    """One flagged criterion with the reviewer-facing explanation."""
    key: str
    label: str
    category_key: str
    subcategory_key: str
    data_field: str
    risk_level: RiskLevel
    rationale: str
    mitigation: str


# =============================================================================
# Domain knowledge base
# =============================================================================
#
# This is deliberately NOT a simple keyword match against criterion names.
# Two criteria can share a keyword ("income") and carry very different
# proxy risk depending on *how* the score is used (characterizing demand
# vs. screening sites out) and *what* it's computed over (a ratio/index vs.
# an absolute level, a single snapshot vs. a trend). The entries below
# encode that judgment per criterion key, with heuristic fallbacks for
# criteria this module hasn't been specifically tuned for.
#
# Each entry is (risk_level, rationale, mitigation).

_KNOWN_FLAGS = {
    "disposable_income": (
        RiskLevel.HIGH,
        "Absolute disposable income at the neighbourhood level is one of the "
        "strongest available proxies for religious/caste composition in many "
        "Indian cities, because historical patterns of residential "
        "segregation by community are strongly correlated with local income "
        "levels. Used as a BENEFIT criterion that raises a site's score, "
        "this can systematically favour higher-income, and thus "
        "community-skewed, areas over lower-income ones with equal or "
        "better commercial fundamentals.",
        "Prefer category_relevant_expenditure or spend_index_vs_national "
        "(spend relative to the local basket / national baseline) over raw "
        "disposable income where possible, since relative spend on the "
        "specific retail category is closer to the actual business signal "
        "and less tightly coupled to community composition. If disposable "
        "income is still used, document explicitly why it's needed beyond "
        "the spend criteria, cap its AHP weight, and review its correlation "
        "with local community composition for each city/region before use.",
    ),
    "category_relevant_expenditure": (
        RiskLevel.MEDIUM,
        "Category-level expenditure is a more defensible demand signal than "
        "raw income, but it is still derived from HCES economic data that "
        "can carry the same underlying correlation with community "
        "composition, especially for discretionary-spend categories that "
        "are also affluence-linked.",
        "Use this to characterize demand for the specific retail category, "
        "not as a general affluence screen. Document the expected magnitude "
        "of the effect and consider normalizing per-capita within the unit "
        "rather than using an absolute level, to reduce overlap with "
        "general affluence.",
    ),
    "spend_index_vs_national": (
        RiskLevel.LOW,
        "A ratio against a national baseline is less likely to be a strong "
        "proxy than an absolute income level, but it can still trend with "
        "affluence and thus community composition in areas with strong "
        "residential segregation patterns.",
        "Prefer this over disposable_income where the same demand signal is "
        "needed. Still worth a documented sanity check against community "
        "composition data for the target cities before deployment.",
    ),
    "rent": (
        RiskLevel.MEDIUM,
        "Commercial rent per sqft correlates with the affluence of the "
        "surrounding area, which in turn can correlate with religious/caste "
        "composition in cities with strong residential segregation "
        "patterns. Used as a COST criterion, high rent areas are "
        "penalized, which is a legitimate commercial concern but can have "
        "the secondary effect of favouring lower-cost, and demographically "
        "different, areas independent of actual commercial merit.",
        "Keep this criterion (rent is a legitimate, necessary cost input), "
        "but document that its purpose is unit economics, not area "
        "screening, and check it isn't double-counting with other "
        "affluence-correlated criteria (e.g. disposable_income) in the "
        "aggregate weighting.",
    ),
    "unit_size": (
        RiskLevel.LOW,
        "Unrelated to protected characteristics directly, but larger, "
        "higher-spec retail units cluster in more affluent commercial "
        "corridors, giving this a weak indirect correlation with the same "
        "affluence-composition pattern as rent and income criteria.",
        "No specific action likely needed beyond noting the indirect "
        "correlation in documentation if unit_size is combined with "
        "several other affluence-linked criteria in the same model.",
    ),
    "visibility": (
        RiskLevel.LOW,
        "Visibility scores for property can be indirectly correlated with "
        "investment in an area's built environment, which tracks the same "
        "broad affluence patterns as other property criteria.",
        "Low priority; document the indirect correlation if this criterion "
        "carries significant AHP weight alongside rent/income criteria.",
    ),
    "planning_constraints": (
        RiskLevel.LOW,
        "Zoning and land-use restriction patterns can, in some Indian "
        "cities, reflect historically uneven municipal investment across "
        "areas with different community composition, making this a weak "
        "indirect proxy in aggregate, even though the criterion itself is "
        "purely regulatory.",
        "Document the data source and criteria for the constraint score; "
        "if it's derived from municipal zoning maps, spot-check whether "
        "constraint severity correlates with area community composition "
        "before deployment in a new city.",
    ),
    "pedestrian_footfall": (
        RiskLevel.LOW,
        "Footfall indices are generally a direct commercial signal, but in "
        "areas with strong community-linked commercial specialization "
        "(e.g. specific market areas historically associated with a "
        "community), footfall can partially encode that association.",
        "Generally fine to use as-is; note the indirect risk in "
        "documentation for markets with known community-linked commercial "
        "clustering.",
    ),
}

# Substring heuristics used only for criteria NOT already covered above,
# so a newly added criterion with an income/affluence-flavoured field still
# gets a conservative flag pointing the reviewer at _KNOWN_FLAGS-style
# reasoning, rather than passing through silently.
_FALLBACK_HEURISTICS: tuple = (
    (("income", "affluence", "wealth"), RiskLevel.HIGH,
     "Field name suggests an absolute income/affluence measure, which is "
     "the pattern most likely to proxy religious/caste/community "
     "composition at the neighbourhood level in the Indian context.",
     "Review whether a relative measure (index vs. baseline, or "
     "category-specific spend) can serve the same purpose with less "
     "correlation to community composition; document the decision either "
     "way."),
    (("rent", "property_value", "land_value"), RiskLevel.MEDIUM,
     "Field name suggests a property-value-linked measure, which tends to "
     "correlate with area affluence and, in turn, community composition "
     "in some cities.",
     "Confirm this is used for unit economics rather than as an implicit "
     "area-affluence screen, and document that distinction."),
    (("caste", "religion", "community", "ethnic"), RiskLevel.HIGH,
     "Field name references a protected characteristic, or a term closely "
     "associated with one, directly.",
     "This should not be used as a scoring input at all; escalate for "
     "immediate legal/compliance review rather than routine mitigation."),
)


def _heuristic_flag(data_field: str, label: str) -> Optional[tuple]:
    text = f"{data_field} {label}".lower()
    for keywords, risk, rationale, mitigation in _FALLBACK_HEURISTICS:
        if any(kw in text for kw in keywords):
            return risk, rationale, mitigation
    return None


def audit_criteria_for_proxies(criteria_hierarchy_config: Iterable) -> list:
    """
    Review every leaf criterion in `criteria_hierarchy_config` and flag any
    that could function, directly or indirectly, as a proxy for a
    protected characteristic (religion, caste/community, ethnicity, etc.),
    with an emphasis on the income/affluence correlations most relevant in
    the Indian urban context.

    `criteria_hierarchy_config` is expected to be the CRITERIA_TREE tuple
    from criteria_hierarchy.py (a tuple of Category -> Subcategory ->
    Criterion), or anything else that yields (category, subcategory,
    criterion) triples via that module's `iter_leaves` helper. This
    function accepts the raw tree and walks it itself so it has no import
    dependency on criteria_hierarchy.py beyond the expected shape.

    Returns a list of ProxyFlag objects, one per flagged criterion, sorted
    with HIGH risk first. Criteria not flagged are simply omitted from the
    result -- an empty or short list is NOT a clearance; see the module
    docstring for the limits of this analysis.

    *** IMPORTANT ***
    This function is read-only: it does not modify, remove, or reweight
    any criterion in `criteria_hierarchy_config`, and it has no side
    effects on the scoring pipeline. Its output is a report meant to be
    READ AND ACTED ON BY A HUMAN REVIEWER before the criteria hierarchy is
    used in production scoring. Do not wire this function's output into an
    automated filter that drops or disables criteria without human
    sign-off -- automated proxy detection has real limits (see module
    docstring) and is a starting aid for review, not a substitute for it.
    """
    flags = []

    for category, subcategory, criterion in _iter_leaves(criteria_hierarchy_config):
        entry = _KNOWN_FLAGS.get(criterion.key)
        if entry is None:
            entry = _heuristic_flag(criterion.data_field, criterion.label)
        if entry is None:
            continue

        risk_level, rationale, mitigation = entry
        flags.append(
            ProxyFlag(
                key=criterion.key,
                label=criterion.label,
                category_key=category.key,
                subcategory_key=subcategory.key,
                data_field=criterion.data_field,
                risk_level=risk_level,
                rationale=rationale,
                mitigation=mitigation,
            )
        )

    _risk_order = {RiskLevel.HIGH: 0, RiskLevel.MEDIUM: 1, RiskLevel.LOW: 2}
    flags.sort(key=lambda f: (_risk_order[f.risk_level], f.category_key, f.key))
    return flags


def _iter_leaves(tree: Iterable):
    """Local walk over the tree shape, so this module doesn't require
    importing criteria_hierarchy.py directly (avoids a hard coupling in
    case this audit is run against a config loaded from elsewhere, e.g.
    JSON/dict-based configs in later phases)."""
    for category in tree:
        for subcategory in category.subcategories:
            for criterion in subcategory.criteria:
                yield category, subcategory, criterion


def format_report(flags: list) -> str:
    """Render a list of ProxyFlag objects as a human-readable report.

    This is for a human reviewer's consumption only -- it does not decide
    anything and nothing downstream should parse this string to drive
    automated behaviour.
    """
    if not flags:
        return (
            "No criteria were flagged by the automated heuristics.\n"
            "This is NOT a clearance -- see proxy_audit.py's module "
            "docstring for the limits of automated proxy detection. A "
            "human review of the full criteria set (and its combined "
            "weighting) is still required before deployment."
        )

    lines = [
        "PROXY-CRITERIA AUDIT REPORT",
        "(For human review before deployment. Not an automated filter. "
        "Not a compliance sign-off.)",
        "",
    ]
    for f in flags:
        lines.append(f"[{f.risk_level.value.upper()}] {f.label} ({f.key})")
        lines.append(f"  location:   {f.category_key} / {f.subcategory_key}")
        lines.append(f"  data_field: {f.data_field}")
        lines.append(f"  why flagged: {f.rationale}")
        lines.append(f"  suggested mitigation: {f.mitigation}")
        lines.append("")

    lines.append(
        "Reminder: absence of a flag above does not mean a criterion is "
        "safe -- combinations of criteria can jointly reconstruct a proxy "
        "signal that no single-criterion heuristic catches. See the module "
        "docstring for full limitations."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    # Example manual run against the real tree, if available on the path.
    try:
        from criteria_hierarchy import CRITERIA_TREE
    except ImportError:
        print(
            "criteria_hierarchy.py not found on the import path -- run this "
            "from the project root, or import audit_criteria_for_proxies "
            "and pass your own tree/config."
        )
    else:
        result = audit_criteria_for_proxies(CRITERIA_TREE)
        print(format_report(result))
