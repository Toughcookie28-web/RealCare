"""add eval run tracking

Revision ID: 0006_eval_run_tracking
Revises: 0005
Create Date: 2026-03-19
"""
from alembic import op
import sqlalchemy as sa


revision = '0006_eval_run_tracking'
down_revision = '0005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('eval_runs',
        sa.Column('run_id', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('eval_type', sa.String(length=32), nullable=False),
        sa.Column('mode', sa.String(length=64), nullable=False),
        sa.Column('split', sa.String(length=32), nullable=False),
        sa.Column('sample_count', sa.Integer(), nullable=False),
        sa.Column('git_sha', sa.String(length=64), nullable=False),
        sa.Column('dataset_path', sa.Text(), nullable=False),
        sa.Column('snapshot_path', sa.Text(), nullable=False),
        sa.Column('judge_model', sa.String(length=128), nullable=True),
        sa.Column('metadata_json', sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint('run_id'),
    )
    op.create_index('ix_eval_runs_eval_type', 'eval_runs', ['eval_type'])
    op.create_index('ix_eval_runs_mode', 'eval_runs', ['mode'])
    op.create_index('ix_eval_runs_split', 'eval_runs', ['split'])

    op.create_table('eval_run_metrics',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('run_id', sa.String(length=64), nullable=False),
        sa.Column('metric_name', sa.String(length=128), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['eval_runs.run_id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_eval_run_metrics_run_id', 'eval_run_metrics', ['run_id'])
    op.create_index('ix_eval_run_metrics_metric_name', 'eval_run_metrics', ['metric_name'])

    op.create_table('eval_run_slices',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('run_id', sa.String(length=64), nullable=False),
        sa.Column('slice_name', sa.String(length=128), nullable=False),
        sa.Column('metric_name', sa.String(length=128), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('sample_count', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['eval_runs.run_id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_eval_run_slices_run_id', 'eval_run_slices', ['run_id'])
    op.create_index('ix_eval_run_slices_slice_name', 'eval_run_slices', ['slice_name'])
    op.create_index('ix_eval_run_slices_metric_name', 'eval_run_slices', ['metric_name'])


def downgrade() -> None:
    op.drop_index('ix_eval_run_slices_metric_name', table_name='eval_run_slices')
    op.drop_index('ix_eval_run_slices_slice_name', table_name='eval_run_slices')
    op.drop_index('ix_eval_run_slices_run_id', table_name='eval_run_slices')
    op.drop_table('eval_run_slices')

    op.drop_index('ix_eval_run_metrics_metric_name', table_name='eval_run_metrics')
    op.drop_index('ix_eval_run_metrics_run_id', table_name='eval_run_metrics')
    op.drop_table('eval_run_metrics')

    op.drop_index('ix_eval_runs_split', table_name='eval_runs')
    op.drop_index('ix_eval_runs_mode', table_name='eval_runs')
    op.drop_index('ix_eval_runs_eval_type', table_name='eval_runs')
    op.drop_table('eval_runs')
