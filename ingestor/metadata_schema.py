# ingestor/metadata_schema.py
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class BenefitType(str, Enum):
    CASH_TRANSFER = "cash_transfer"
    SCHOLARSHIP = "scholarship"
    LOAN = "loan"
    SUBSIDY = "subsidy"
    INSURANCE = "insurance"
    PENSION = "pension"
    FOOD_GRAIN = "food_grain"
    HOUSING = "housing"
    SKILL_TRAINING = "skill_training"
    HEALTHCARE = "healthcare"
    OTHER = "other"


class ChunkType(str, Enum):
    SUMMARY = "summary"          # ~200 tokens, high-level
    ELIGIBILITY = "eligibility"  # ~400 tokens, who can apply
    BENEFITS = "benefits"        # ~400 tokens, what you get
    DOCUMENTS = "documents"      # ~400 tokens, what papers needed
    PROCESS = "process"          # ~400 tokens, how to apply
    FACT = "fact"                # ~100 tokens, specific number/date


class SchemeMetadata(BaseModel):
    """
    Metadata schema for a single government scheme.
    Every chunk stored in Qdrant carries this as its payload.
    40% of dev time goes here — this is what makes retrieval work.
    """
    # Identity
    scheme_id: str = Field(description="Unique slug e.g. 'pm-kisan-001'")
    scheme_name: str = Field(description="Full official scheme name")
    scheme_name_hindi: Optional[str] = Field(default=None)

    # Classification — used for metadata pre-filtering
    ministry: str = Field(description="e.g. 'Ministry of Agriculture'")
    category: list[str] = Field(
        description="e.g. ['agriculture', 'income_support']"
    )
    benefit_type: BenefitType
    chunk_type: ChunkType

    # Eligibility filters — hard filters applied BEFORE vector search
    # This cuts search space from 950 to ~50-80 relevant schemes
    eligible_states: list[str] = Field(
        default=["all"],
        description="List of state names or ['all'] for central schemes"
    )
    beneficiary_types: list[str] = Field(
        description="e.g. ['farmer', 'student', 'widow', 'sc', 'st', 'obc']"
    )
    gender: list[str] = Field(
        default=["all"],
        description="['male', 'female', 'all', 'transgender']"
    )
    age_min: Optional[int] = Field(default=None)
    age_max: Optional[int] = Field(default=None)
    income_ceiling_inr: Optional[int] = Field(
        default=None,
        description="Annual income ceiling in INR. None means no limit."
    )
    caste_eligibility: list[str] = Field(
        default=["all"],
        description="['general', 'obc', 'sc', 'st', 'all']"
    )

    # Benefit details
    benefit_amount: Optional[str] = Field(
        default=None,
        description="Human-readable e.g. '6000/year' or '2 lakh coverage'"
    )
    documents_required: list[str] = Field(
        default=[],
        description="e.g. ['aadhaar', 'land_record', 'bank_passbook']"
    )
    application_mode: list[str] = Field(
        default=["online"],
        description="['online', 'offline', 'csc', 'bank']"
    )

    # Chunk hierarchy — for parent-document retrieval
    parent_id: Optional[str] = Field(
        default=None,
        description="scheme_id of the summary chunk, for child chunks"
    )

    # Provenance
    source_url: str = Field(
        default="https://myscheme.gov.in"
    )
    source_dataset: str = Field(
        description="Which dataset this came from"
    )
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="For embedding drift detection"
    )
    last_verified: str = Field(
        description="YYYY-MM format e.g. '2024-01'"
    )
    language: str = Field(default="en")