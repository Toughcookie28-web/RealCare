"""drop hitl_reviews table

Revision ID: 0007_drop_hitl_reviews
Revises: 0006_eval_run_tracking
Create Date: 2026-03-20
"""

from alembic import op
import sqlalchemy as sa


revision = '0007_drop_hitl_reviews'
down_revision = '0006_eval_run_tracking'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index('ix_hitl_reviews_status', table_name='hitl_reviews')
    op.drop_index('ix_hitl_reviews_session_id', table_name='hitl_reviews')
    op.drop_table('hitl_reviews')


def downgrade() -> None:
    op.create_table(
        'hitl_reviews',
        sa.Column('approval_id', sa.String(length=64), primary_key=True),
        sa.Column('session_id', sa.String(length=64), nullable=False),
        sa.Column('trace_id', sa.String(length=64), nullable=False),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False, server_default='pending'),
        sa.Column('reviewer_note', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('reviewed_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_hitl_reviews_session_id', 'hitl_reviews', ['session_id'])
    op.create_index('ix_hitl_reviews_status', 'hitl_reviews', ['status'])
