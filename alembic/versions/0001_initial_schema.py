"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-02-16
"""

from alembic import op
import sqlalchemy as sa


revision = '0001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    op.create_table(
        'sessions',
        sa.Column('session_id', sa.String(length=64), primary_key=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('last_active', sa.DateTime(), nullable=False),
    )

    op.create_table(
        'messages',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('session_id', sa.String(length=64), sa.ForeignKey('sessions.session_id'), nullable=False),
        sa.Column('role', sa.String(length=16), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('source', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )

    op.create_table(
        'conversation_summaries',
        sa.Column('session_id', sa.String(length=64), sa.ForeignKey('sessions.session_id'), primary_key=True),
        sa.Column('summary', sa.Text(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
    )

    op.create_table(
        'user_facts',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('session_id', sa.String(length=64), sa.ForeignKey('sessions.session_id'), nullable=False),
        sa.Column('fact_key', sa.String(length=128), nullable=False),
        sa.Column('fact_value', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )

    # pgvector column added as raw SQL to keep migration portable for environments without pgvector package.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            chunk_id VARCHAR(128) UNIQUE NOT NULL,
            doc_id VARCHAR(255) NOT NULL,
            page INTEGER,
            section VARCHAR(255),
            content TEXT NOT NULL,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            embedding vector(384),
            created_at TIMESTAMP NOT NULL DEFAULT now()
        )
        """
    )
    op.create_index('ix_document_chunks_doc_id_page', 'document_chunks', ['doc_id', 'page'])


def downgrade() -> None:
    op.drop_index('ix_document_chunks_doc_id_page', table_name='document_chunks')
    op.drop_table('document_chunks')
    op.drop_table('user_facts')
    op.drop_table('conversation_summaries')
    op.drop_table('messages')
    op.drop_table('sessions')
