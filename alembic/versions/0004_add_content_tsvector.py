"""add content tsvector column and GIN index for BM25-like sparse retrieval

Revision ID: 0004_content_tsv
Revises: 0003_vec_contract
Create Date: 2026-03-12
"""

from alembic import op

revision = '0004_content_tsv'
down_revision = '0003_vec_contract'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector")
    op.execute("UPDATE document_chunks SET content_tsv = to_tsvector('english', content)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_content_tsv "
        "ON document_chunks USING GIN (content_tsv)"
    )
    op.execute("""
        CREATE OR REPLACE FUNCTION document_chunks_tsv_trigger() RETURNS trigger AS $$
        BEGIN
            NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        DROP TRIGGER IF EXISTS trg_document_chunks_tsv ON document_chunks;
        CREATE TRIGGER trg_document_chunks_tsv
            BEFORE INSERT OR UPDATE OF content ON document_chunks
            FOR EACH ROW EXECUTE FUNCTION document_chunks_tsv_trigger();
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_document_chunks_tsv ON document_chunks")
    op.execute("DROP FUNCTION IF EXISTS document_chunks_tsv_trigger()")
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_content_tsv")
    op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS content_tsv")
