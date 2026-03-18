"""add long term memory

Revision ID: 0005
Revises: 0004_content_tsv
Create Date: 2026-03-17
"""
from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy

# revision identifiers
revision = '0005'
down_revision = '0004_content_tsv'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('users',
        sa.Column('user_id', sa.String(64), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('last_seen', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('user_id'),
    )
    op.create_table('long_term_facts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(64), nullable=False),
        sa.Column('fact_key', sa.String(128), nullable=False),
        sa.Column('fact_value', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('source_session', sa.String(64), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_long_term_facts_user_key', 'long_term_facts', ['user_id', 'fact_key'])
    op.create_index('ix_long_term_facts_user_id', 'long_term_facts', ['user_id'])
    op.create_index('ix_long_term_facts_fact_key', 'long_term_facts', ['fact_key'])
    op.create_table('conversation_memories',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(64), nullable=False),
        sa.Column('session_id', sa.String(64), nullable=False),
        sa.Column('summary', sa.Text(), nullable=False),
        sa.Column('embedding', pgvector.sqlalchemy.Vector(768), nullable=True),
        sa.Column('medical_entities', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_conversation_memories_user_id_created', 'conversation_memories', ['user_id', 'created_at'])
    op.create_index('ix_conversation_memories_user_id', 'conversation_memories', ['user_id'])
    op.create_index('ix_conversation_memories_session_id', 'conversation_memories', ['session_id'])


def downgrade() -> None:
    op.drop_index('ix_conversation_memories_session_id', table_name='conversation_memories')
    op.drop_index('ix_conversation_memories_user_id', table_name='conversation_memories')
    op.drop_index('ix_conversation_memories_user_id_created', table_name='conversation_memories')
    op.drop_table('conversation_memories')
    op.drop_index('ix_long_term_facts_fact_key', table_name='long_term_facts')
    op.drop_index('ix_long_term_facts_user_id', table_name='long_term_facts')
    op.drop_index('ix_long_term_facts_user_key', table_name='long_term_facts')
    op.drop_table('long_term_facts')
    op.drop_table('users')
