"""align document_chunks vector contract

Revision ID: 0003_vec_contract
Revises: 0002_add_hitl_reviews
Create Date: 2026-03-09
"""

from alembic import op


revision = '0003_vec_contract'
down_revision = '0002_add_hitl_reviews'
branch_labels = None
depends_on = None

TARGET_VECTOR_DIM = 768
PREVIOUS_VECTOR_DIM = 384


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    op.execute(
        f"""
        ALTER TABLE document_chunks
        ALTER COLUMN embedding TYPE vector({TARGET_VECTOR_DIM})
        USING NULL::vector({TARGET_VECTOR_DIM})
        """
    )


def downgrade() -> None:
    op.execute(
        f"""
        ALTER TABLE document_chunks
        ALTER COLUMN embedding TYPE vector({PREVIOUS_VECTOR_DIM})
        USING NULL::vector({PREVIOUS_VECTOR_DIM})
        """
    )
