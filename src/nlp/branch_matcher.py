"""
Free-text branch matcher.

JEE branch names are formal program titles ("Electronics and Electrical
Communication Engineering") that a first-time applicant usually can't map to
their actual interests ("I like building circuits and gadgets"). A student
who doesn't already know the jargon can't search for it.

This closes that gap with a small rule-based query-expansion layer: every
branch name is tagged against a hand-built topic lexicon (keyword triggers ->
a block of plain-language synonyms/concepts for that discipline), and the
expanded text -- not just the formal name -- is what gets indexed. A free-text
query is then matched against that expanded corpus with TF-IDF + cosine
similarity, so "I like building circuits and gadgets" can surface Electronics
branches even though neither query nor branch name shares those exact words;
the lexicon is the bridge between the two vocabularies.

This is deliberately a transparent, rule-based expansion rather than a
black-box embedding model -- every match is explainable by which topic tags
fired, which matters more for a tool advising 17-year-olds on a real
admission decision than a marginal gain in fuzzy semantic recall.
"""
import re
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# trigger: substrings (lowercased) to look for in the formal branch name
# expansion: plain-language synonyms/concepts a non-expert applicant might type
TOPIC_LEXICON = {
    "computer_science": (
        ["computer science", "cse"],
        "coding programming software apps websites algorithms computer science "
        "data structures backend systems",
    ),
    "ai_data": (
        ["artificial intelligence", "data science", "data analytics", "data engineering", "machine learning"],
        "artificial intelligence machine learning AI models data science analytics "
        "statistics prediction deep learning",
    ),
    "math_computing": (
        ["mathematics and computing", "mathematics & computing", "scientific computing", "computational"],
        "math heavy coding quantitative problem solving algorithms simulations modeling",
    ),
    "electronics": (
        ["electronics", "vlsi", "microelectronics", "integrated circuit", "instrumentation"],
        "circuits chips hardware gadgets embedded systems signal processing sensors VLSI",
    ),
    "electrical_power": (
        ["electrical engineering", "electrical and electronics", "power and automation", "power electronics"],
        "electricity power grids motors generators energy systems automation control",
    ),
    "communication": (
        ["communication engineering", "communication systems"],
        "telecom networks wireless signals antennas communication systems",
    ),
    "mechanical": (
        ["mechanical engineering", "mechatronics"],
        "machines engines vehicles design manufacturing robotics automotive thermodynamics building physical things",
    ),
    "manufacturing_industrial": (
        ["manufacturing", "industrial engineering", "production and industrial", "industrial and systems",
         "quality engineering"],
        "factories production lines manufacturing processes operations efficiency supply chain logistics",
    ),
    "civil_structural": (
        ["civil engineering", "structural engineering", "civil and infrastructure"],
        "buildings bridges construction infrastructure structures roads urban planning",
    ),
    "environmental": (
        ["environmental engineering", "environmental science"],
        "environment pollution sustainability climate water treatment ecology",
    ),
    "transportation": (
        ["transportation engineering"],
        "roads traffic transport systems urban mobility logistics",
    ),
    "chemical": (
        ["chemical engineering", "chemical science", "chemical and biochemical"],
        "chemicals reactions processes plants refineries industrial chemistry",
    ),
    "biotech_bio": (
        ["bio engineering", "biological", "biotechnology", "biochemical", "biomedical", "bioscience",
         "biosciences"],
        "biology medicine genetics biotech health life sciences lab research",
    ),
    "pharma": (
        ["pharmaceutic"],
        "drugs medicine pharmaceuticals formulation healthcare",
    ),
    "materials_metallurgy": (
        ["materials engineering", "materials science", "metallurgical", "ceramic engineering", "polymer"],
        "materials metals alloys ceramics polymers manufacturing processes nanomaterials",
    ),
    "mining": (
        ["mining", "mineral engineering"],
        "mines minerals extraction geology underground resources",
    ),
    "petroleum": (
        ["petroleum"],
        "oil gas drilling refineries energy resources",
    ),
    "aerospace": (
        ["aerospace", "space science"],
        "aircraft rockets satellites flight aerodynamics space exploration",
    ),
    "naval_ocean": (
        ["naval architecture", "ocean engineering"],
        "ships boats marine ocean offshore structures naval design",
    ),
    "physics": (
        ["engineering physics", " physics"],
        "physics theory experiments quantum optics research fundamental science",
    ),
    "chemistry": (
        ["chemistry", "industrial chemistry"],
        "chemistry lab experiments molecules reactions research",
    ),
    "math_stats": (
        ["applied mathematics", "statistics"],
        "math statistics probability theory research pure science",
    ),
    "economics": (
        ["economics"],
        "economics markets finance policy social science",
    ),
    "geology_earth": (
        ["geology", "geophysics", "earth science", "exploration geophysics"],
        "earth sciences rocks geology exploration natural resources fieldwork",
    ),
    "architecture": (
        ["architecture", "engineering design"],
        "architecture buildings design aesthetics urban planning creative structures",
    ),
    "energy": (
        ["energy engineering", "energy systems"],
        "renewable energy power systems sustainability solar wind",
    ),
    "textile": (
        ["textile"],
        "fabrics textiles fashion manufacturing materials",
    ),
    "management_dual": (
        ["mba", "management"],
        "business management leadership strategy entrepreneurship",
    ),
}

DEGREE_BOILERPLATE = re.compile(r"\(\d+ years?,.*?\)", re.IGNORECASE)


def clean_branch_name(name: str) -> str:
    return DEGREE_BOILERPLATE.sub("", name).strip()


def expand_branch_text(branch_name: str) -> str:
    lower = branch_name.lower()
    expansions = [clean_branch_name(branch_name)]
    for _, (triggers, expansion) in TOPIC_LEXICON.items():
        if any(t in lower for t in triggers):
            expansions.append(expansion)
    return " ".join(expansions)


def matched_topics(branch_name: str) -> list:
    lower = branch_name.lower()
    return [topic for topic, (triggers, _) in TOPIC_LEXICON.items() if any(t in lower for t in triggers)]


@dataclass
class Match:
    branch: str
    score: float
    topics: list


class BranchMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.branch_names = []
        self.matrix = None

    def fit(self, branch_names):
        self.branch_names = sorted(set(branch_names))
        corpus = [expand_branch_text(b) for b in self.branch_names]
        self.matrix = self.vectorizer.fit_transform(corpus)
        return self

    def match(self, query: str, top_k: int = 10):
        if not query.strip():
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        ranked = sorted(zip(self.branch_names, sims), key=lambda x: -x[1])
        return [
            Match(branch=b, score=float(s), topics=matched_topics(b))
            for b, s in ranked[:top_k] if s > 0
        ]
