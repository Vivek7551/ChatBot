"""
ontology_validator.py
---------------------
Provides OntologyAnnotator: queries BioPortal's /annotator endpoint against a
curated set of biological ontologies relevant to Prof. Peter Devreotes' research
(chemotaxis, signal transduction, membrane protein biology).

Ontologies queried:
  GO    — Gene Ontology  (processes, functions, components)
  PR    — Protein Ontology (canonical protein names)
  CHEBI — Chemical Entities of Biological Interest (lipids, small molecules)
  CL    — Cell Ontology (cell types)
  NCIT  — NCI Thesaurus (methods, disease, phenotypes)
  GO-BP, GO-MF, GO-CC subsets are all included via GO

Usage:
    from ontology_validator import OntologyAnnotator
    ann = OntologyAnnotator()                       # reads BIOPORTAL_API_KEY from env
    hits = ann.annotate("PI3K phosphorylates PIP2 to produce PIP3 at the leading edge.")
    for h in hits:
        print(h)   # OntologyHit(term='PIP3', pref_label='phosphatidylinositol 3,4,5-trisphosphate', ontology='CHEBI', ont_type='Lipid')

If BIOPORTAL_API_KEY is not set or the API is unreachable, annotate() returns []
without raising an exception, so the extraction pipeline degrades gracefully.
"""

from __future__ import annotations

import os
import re
import logging
import functools
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BioPortal REST configuration
# ---------------------------------------------------------------------------
_BIOPORTAL_BASE = "https://data.bioontology.org"
_REQUEST_TIMEOUT = 10   # seconds per API call
_MAX_TEXT_CHARS  = 4000  # BioPortal annotator cap; we truncate if needed

# Ontologies most relevant to Devreotes' work.
# Full list: https://bioportal.bioontology.org/ontologies
_DEVREOTES_ONTOLOGIES = [
    "GO",     # Gene Ontology — processes / functions / components
    "PR",     # Protein Ontology
    "CHEBI",  # Chemical Entities (lipids, second messengers, drugs)
    "CL",     # Cell Ontology
    "NCIT",   # NCI Thesaurus — methods, assays, phenotypes
]

# Map BioPortal "semantic type" (from UMLS) → graph node type used in triples
# We also map ontology → fallback node type when semantic_type is absent.
_SEMANTIC_TYPE_TO_NODE_TYPE: dict[str, str] = {
    "T116": "Protein",           # Amino Acid, Peptide, or Protein
    "T126": "Protein",           # Enzyme
    "T085": "Protein",           # Molecular Sequence
    "T028": "Gene",              # Gene or Genome
    "T114": "Gene",              # Nucleic Acid, Nucleoside, or Nucleotide
    "T044": "Process",           # Molecular Function
    "T043": "Process",           # Cell Function
    "T045": "Process",           # Genetic Function
    "T042": "Process",           # Organ or Tissue Function
    "T038": "Process",           # Biologic Function
    "T039": "Process",           # Physiologic Function
    "T046": "Phenotype",         # Pathologic Function
    "T047": "Disease",           # Disease or Syndrome
    "T048": "Disease",           # Mental or Behavioral Dysfunction
    "T060": "Method",            # Diagnostic Procedure
    "T061": "Method",            # Therapeutic or Preventive Procedure
    "T059": "Method",            # Laboratory Procedure
    "T130": "SmallMolecule",     # Indicator, Reagent, or Diagnostic Aid
    "T121": "SmallMolecule",     # Pharmacologic Substance
    "T109": "SmallMolecule",     # Organic Chemical
    "T123": "SmallMolecule",     # Biologically Active Substance
    "T026": "Structure",         # Cell Component
    "T025": "CellType",          # Cell
    "T022": "Organism",          # Body System
    "T001": "Organism",          # Organism
    "T010": "Organism",          # Eukaryote
}

_ONTOLOGY_FALLBACK_NODE_TYPE: dict[str, str] = {
    "GO":    "Process",
    "PR":    "Protein",
    "CHEBI": "SmallMolecule",
    "CL":    "CellType",
    "NCIT":  "Method",
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OntologyHit:
    """A single term found in a text chunk that is present in a biological ontology."""
    term:       str   # The matched surface form from the text
    pref_label: str   # Canonical preferred label from the ontology
    ontology:   str   # e.g. "GO", "CHEBI"
    node_type:  str   # The graph node type we map this to
    ont_id:     str   # Ontology class IRI (for provenance)

    def __str__(self) -> str:
        return f"{self.term!r} → {self.pref_label} [{self.node_type}, {self.ontology}]"


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------
class OntologyAnnotator:
    """
    Wraps the BioPortal /annotator endpoint.

    Call annotate(text) to get a list of OntologyHit objects for all biological
    terms found in the text that match the curated ontology set.

    Results are cached in-process (term → hits) to avoid duplicate API calls
    across chunks that share common vocabulary.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        ontologies: list[str] = _DEVREOTES_ONTOLOGIES,
    ):
        self.api_key   = api_key or os.environ.get("BIOPORTAL_API_KEY", "")
        self.ontologies = ontologies
        self._cache: dict[str, list[OntologyHit]] = {}
        self._available = bool(self.api_key)

        if not self._available:
            logger.warning(
                "[OntologyAnnotator] BIOPORTAL_API_KEY not set — "
                "ontology grounding disabled; falling back to pure-LLM extraction."
            )

    # ------------------------------------------------------------------
    def annotate(self, text: str) -> list[OntologyHit]:
        """
        Return OntologyHit objects for every biological term recognised in *text*.
        Returns [] if the API key is missing or the call fails.
        """
        if not self._available:
            return []

        # Truncate to BioPortal limit
        text_for_api = text[:_MAX_TEXT_CHARS]

        # Cache by text hash to avoid re-querying identical passages
        cache_key = text_for_api.strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        hits = self._call_bioportal(text_for_api)
        self._cache[cache_key] = hits
        return hits

    # ------------------------------------------------------------------
    def _call_bioportal(self, text: str) -> list[OntologyHit]:
        """Make the actual HTTP call to BioPortal /annotator."""
        headers = {"Authorization": f"apikey token={self.api_key}"}
        params  = {
            "text":             text,
            "ontologies":       ",".join(self.ontologies),
            "longest_only":     "true",   # avoid nested/redundant matches
            "exclude_numbers":  "true",
            "whole_word_only":  "true",
            "expand_mappings":  "false",
            "include":          "prefLabel,semanticType",
        }

        try:
            resp = requests.get(
                f"{_BIOPORTAL_BASE}/annotator",
                headers=headers,
                params=params,
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning(f"[OntologyAnnotator] BioPortal API error: {exc}")
            return []

        return self._parse_annotations(data)

    # ------------------------------------------------------------------
    def _parse_annotations(self, data: list[dict]) -> list[OntologyHit]:
        """Convert BioPortal annotation JSON → list[OntologyHit]."""
        hits: list[OntologyHit] = []
        seen_terms: set[str] = set()   # deduplicate by (term, ontology)

        for annotation in data:
            # Extract matched surface-form text
            annotations_list = annotation.get("annotations", [])
            if not annotations_list:
                continue
            term = annotations_list[0].get("text", "").strip()
            if not term:
                continue

            # Extract ontology class details
            cls_obj    = annotation.get("annotatedClass", {})
            ont_id     = cls_obj.get("@id", "")
            pref_label = cls_obj.get("prefLabel", term)

            # Derive ontology acronym from class IRI or links
            links      = cls_obj.get("links", {})
            ont_link   = links.get("ontology", "")
            ontology   = _ontology_from_link(ont_link) or _ontology_from_id(ont_id)

            if ontology not in self.ontologies:
                continue  # filter to our curated set

            # Determine node type
            sem_types   = cls_obj.get("semanticType", []) or []
            node_type   = _resolve_node_type(sem_types, ontology)

            key = (term.lower(), ontology)
            if key in seen_terms:
                continue
            seen_terms.add(key)

            hits.append(OntologyHit(
                term       = term,
                pref_label = pref_label,
                ontology   = ontology,
                node_type  = node_type,
                ont_id     = ont_id,
            ))

        return hits

    # ------------------------------------------------------------------
    def group_by_node_type(self, hits: list[OntologyHit]) -> dict[str, list[OntologyHit]]:
        """Return hits grouped by node_type for easy prompt construction."""
        groups: dict[str, list[OntologyHit]] = {}
        for h in hits:
            groups.setdefault(h.node_type, []).append(h)
        return groups

    # ------------------------------------------------------------------
    def hits_to_prompt_section(self, hits: list[OntologyHit]) -> str:
        """
        Format ontology hits as a prompt section for the LLM.
        Returns an empty string if there are no hits.
        """
        if not hits:
            return ""

        groups = self.group_by_node_type(hits)
        lines  = []
        for node_type, group_hits in sorted(groups.items()):
            for h in group_hits:
                canonical = h.pref_label if h.pref_label != h.term else h.term
                lines.append(
                    f"  - {h.term}"
                    + (f" (canonical: {canonical})" if canonical != h.term else "")
                    + f"  [type={node_type}, ontology={h.ontology}]"
                )

        return (
            "ONTOLOGY-VERIFIED ENTITIES\n"
            "(These terms appear in this text AND are confirmed by biological ontologies.\n"
            " Prefer them as nodes. Use their canonical names where shown.)\n"
            + "\n".join(lines)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ontology_from_link(link: str) -> str:
    """Extract acronym like 'GO' or 'CHEBI' from a BioPortal ontology URL."""
    m = re.search(r"/ontologies/([A-Z0-9_\-]+)", link, re.IGNORECASE)
    return m.group(1).upper() if m else ""


def _ontology_from_id(ont_id: str) -> str:
    """Heuristic: infer ontology from the class IRI."""
    if "purl.obolibrary.org/obo/GO" in ont_id or "GO_" in ont_id:
        return "GO"
    if "purl.obolibrary.org/obo/CHEBI" in ont_id or "CHEBI_" in ont_id:
        return "CHEBI"
    if "purl.obolibrary.org/obo/PR" in ont_id or "PR_" in ont_id:
        return "PR"
    if "purl.obolibrary.org/obo/CL" in ont_id or "CL_" in ont_id:
        return "CL"
    if "ncit" in ont_id.lower():
        return "NCIT"
    return ""


def _resolve_node_type(semantic_types: list[str], ontology: str) -> str:
    """Map UMLS semantic type codes → graph node type, with ontology fallback."""
    for st in semantic_types:
        if st in _SEMANTIC_TYPE_TO_NODE_TYPE:
            return _SEMANTIC_TYPE_TO_NODE_TYPE[st]
    return _ONTOLOGY_FALLBACK_NODE_TYPE.get(ontology, "BiologicalEntity")
