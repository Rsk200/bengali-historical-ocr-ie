#!/usr/bin/env python3
"""
Pydantic Ontology & Extraction Schema
Defines strict Pydantic v2 data models for Bengali historical information extraction:
  - 6 Entity types (PERSON, ORGANIZATION, LOCATION, EVENT, DATE/PERIOD, DOCUMENT/TREATY/LAW)
  - 10 Directed Relation types with domain/range type validation
  - Event Frames with atomic and descriptive slots
  - Domain constraints rejecting bare titles/roles as standalone organizations
  - Mandatory non-empty verbatim evidence_span enforcement
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field, field_validator, model_validator


# ==============================================================================
# 1. Enums: Entity Types & Relation Types
# ==============================================================================

class EntityType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    DATE_PERIOD = "DATE/PERIOD"
    DOCUMENT_TREATY_LAW = "DOCUMENT/TREATY/LAW"


class RelationType(str, Enum):
    participated_in = "participated_in"
    led_by = "led_by"
    member_of = "member_of"
    occurred_at = "occurred_at"
    occurred_on = "occurred_on"
    caused = "caused"
    resulted_in = "resulted_in"
    preceded = "preceded"
    signed_issued = "signed_issued"
    located_in = "located_in"


# Domain/Range compatibility constraints for relations
RELATION_ALLOWED_TYPES: Dict[RelationType, Dict[str, Set[EntityType]]] = {
    RelationType.participated_in: {
        "head": {EntityType.PERSON, EntityType.ORGANIZATION},
        "tail": {EntityType.EVENT}
    },
    RelationType.led_by: {
        "head": {EntityType.ORGANIZATION, EntityType.EVENT},
        "tail": {EntityType.PERSON}
    },
    RelationType.member_of: {
        "head": {EntityType.PERSON},
        "tail": {EntityType.ORGANIZATION}
    },
    RelationType.occurred_at: {
        "head": {EntityType.EVENT},
        "tail": {EntityType.LOCATION}
    },
    RelationType.occurred_on: {
        "head": {EntityType.EVENT, EntityType.DOCUMENT_TREATY_LAW},
        "tail": {EntityType.DATE_PERIOD}
    },
    RelationType.caused: {
        "head": {EntityType.EVENT, EntityType.DOCUMENT_TREATY_LAW},
        "tail": {EntityType.EVENT}
    },
    RelationType.resulted_in: {
        "head": {EntityType.EVENT},
        "tail": {EntityType.EVENT, EntityType.DOCUMENT_TREATY_LAW, EntityType.ORGANIZATION}
    },
    RelationType.preceded: {
        "head": {EntityType.EVENT},
        "tail": {EntityType.EVENT}
    },
    RelationType.signed_issued: {
        "head": {EntityType.PERSON, EntityType.ORGANIZATION},
        "tail": {EntityType.DOCUMENT_TREATY_LAW}
    },
    RelationType.located_in: {
        "head": {EntityType.LOCATION, EntityType.ORGANIZATION},
        "tail": {EntityType.LOCATION}
    }
}

# Bare titles/roles that cannot stand alone as ORGANIZATION nodes
DISALLOWED_BARE_ORGANIZATION_TITLES: Set[str] = {
    "প্রধানমন্ত্রী", "রাষ্ট্রপতি", "মুখ্যমন্ত্রী", "গভর্নর জেনারেল", "মন্ত্রী",
    "স্পিকার", "উপদেষ্টা", "সদস্য", "সভাপতি", "সাধারণ সম্পাদক",
    "prime minister", "president", "chief minister", "governor general", "minister"
}


# ==============================================================================
# 2. Entity Model
# ==============================================================================

class ExtractedEntity(BaseModel):
    entity_id: str = Field(description="Unique entity identifier within chunk")
    name: str = Field(min_length=1, description="Entity surface name")
    entity_type: EntityType = Field(description="One of the 6 canonical entity types")
    page_id: str = Field(default="", description="Source page identifier")
    chunk_id: str = Field(default="", description="Source chunk identifier")
    char_start: Optional[int] = Field(default=None, description="Optional page-level character start offset")
    char_end: Optional[int] = Field(default=None, description="Optional page-level character end offset")
    evidence_span: str = Field(min_length=1, description="Mandatory non-empty verbatim source excerpt")

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Entity name cannot be empty or whitespace only.")
        return s

    @field_validator("evidence_span")
    @classmethod
    def validate_evidence_span(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("evidence_span is mandatory and cannot be empty.")
        return s

    @model_validator(mode="after")
    def validate_domain_constraints(self) -> "ExtractedEntity":
        # Disallow bare role/title alone as an ORGANIZATION
        if self.entity_type == EntityType.ORGANIZATION:
            clean_name = self.name.strip().lower()
            if clean_name in DISALLOWED_BARE_ORGANIZATION_TITLES:
                raise ValueError(
                    f"Bare role/title '{self.name}' cannot be extracted as an ORGANIZATION node. "
                    f"It must qualify a PERSON or be linked to an explicit organizational body."
                )
        return self


# ==============================================================================
# 3. Relation Model
# ==============================================================================

class ExtractedRelation(BaseModel):
    relation_id: str = Field(description="Unique relation identifier within chunk")
    head_text: str = Field(min_length=1, description="Subject entity text")
    head_type: EntityType = Field(description="Subject entity type")
    relation_type: RelationType = Field(description="One of the 10 canonical directed relation types")
    tail_text: str = Field(min_length=1, description="Object entity text")
    tail_type: EntityType = Field(description="Object entity type")
    page_id: str = Field(default="", description="Source page identifier")
    chunk_id: str = Field(default="", description="Source chunk identifier")
    evidence_span: str = Field(min_length=1, description="Mandatory non-empty verbatim source excerpt")

    @field_validator("evidence_span")
    @classmethod
    def validate_evidence_span(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("evidence_span is mandatory and cannot be empty.")
        return s

    @model_validator(mode="after")
    def validate_relation_domain_and_range(self) -> "ExtractedRelation":
        allowed = RELATION_ALLOWED_TYPES.get(self.relation_type)
        if allowed:
            if self.head_type not in allowed["head"]:
                allowed_heads = [t.value for t in allowed["head"]]
                raise ValueError(
                    f"Invalid head_type '{self.head_type.value}' for relation '{self.relation_type.value}'. "
                    f"Allowed head types: {allowed_heads}"
                )
            if self.tail_type not in allowed["tail"]:
                allowed_tails = [t.value for t in allowed["tail"]]
                raise ValueError(
                    f"Invalid tail_type '{self.tail_type.value}' for relation '{self.relation_type.value}'. "
                    f"Allowed tail types: {allowed_tails}"
                )
        return self


# ==============================================================================
# 4. Event Frame Model
# ==============================================================================

class EventParticipant(BaseModel):
    name: str = Field(min_length=1, description="Participant name")
    entity_type: EntityType = Field(description="Participant type (must be PERSON or ORGANIZATION)")
    role: str = Field(default="participant", description="Role of participant, e.g., leader, speaker, protester")

    @model_validator(mode="after")
    def validate_participant_type(self) -> "EventParticipant":
        if self.entity_type not in (EntityType.PERSON, EntityType.ORGANIZATION):
            raise ValueError(
                f"EventParticipant entity_type must be PERSON or ORGANIZATION, got '{self.entity_type.value}'"
            )
        return self


class ExtractedEventFrame(BaseModel):
    event_id: str = Field(description="Unique event frame identifier")
    event_type: str = Field(min_length=1, description="Historical event category, e.g. PROTEST, AGREEMENT, ELECTION, VIOLENCE")
    trigger_name: str = Field(min_length=1, description="Event anchor or nominal trigger, e.g. ধর্মঘট, মিছিল, চুক্তি, সম্মেলন")
    participants: List[EventParticipant] = Field(default_factory=list, description="List of participant entities")
    location: Optional[str] = Field(default=None, description="Event location name")
    date: Optional[str] = Field(default=None, description="Event date or period")
    causes: List[str] = Field(default_factory=list, description="Causes or triggering factors")
    consequences: List[str] = Field(default_factory=list, description="Outcomes, resolutions, or consequences")
    page_id: str = Field(default="", description="Source page identifier")
    chunk_id: str = Field(default="", description="Source chunk identifier")
    evidence_span: str = Field(min_length=1, description="Mandatory non-empty verbatim source excerpt")

    @field_validator("evidence_span")
    @classmethod
    def validate_evidence_span(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("evidence_span is mandatory and cannot be empty.")
        return s


# ==============================================================================
# 5. Top-Level Chunk Extraction Container
# ==============================================================================

class ChunkExtractionOutput(BaseModel):
    """
    Top-level validated container for all facts extracted from a single chunk.
    """
    entities: List[ExtractedEntity] = Field(default_factory=list, description="Extracted entities")
    relations: List[ExtractedRelation] = Field(default_factory=list, description="Extracted relations")
    events: List[ExtractedEventFrame] = Field(default_factory=list, description="Extracted event frames")
