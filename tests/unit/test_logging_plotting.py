"""Tests for logging and plotting utilities."""

import pytest
import tempfile
import os
import json
from typing import List, Dict

from zeroproof.core import TRTag
from zeroproof.utils.logging import (
    LogEntry,
    TrainingSession,
    StructuredLogger,
    MetricsAggregator,
    ExperimentTracker,
    log_training_step
)


class TestStructuredLogger:
    """Test structured logging functionality."""
    
    def test_initialization(self):
        """Test logger initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = StructuredLogger(temp_dir, "test_session")
            
            assert logger.session_id == "test_session"
            assert logger.run_dir == temp_dir
            assert len(logger.session.logs) == 0
    
    def test_log_metrics(self):
        """Test logging metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = StructuredLogger(temp_dir)
            
            metrics = {
                'loss': 0.5,
                'coverage': 0.8,
                'lambda_rej': 0.1
            }
            
            logger.log_metrics(metrics, epoch=1, step=10)
            
            assert len(logger.session.logs) == 1
            assert logger.session.logs[0].epoch == 1
            assert logger.session.logs[0].step == 10
            assert logger.session.logs[0].metrics == metrics
    
    def test_tag_distribution(self):
        """Test tag distribution logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = StructuredLogger(temp_dir)
            
            tags = [TRTag.REAL, TRTag.REAL, TRTag.PINF, TRTag.PHI]
            tag_metrics = logger.log_tag_distribution(tags)
            
            assert tag_metrics['REAL_count'] == 2
            assert tag_metrics['PINF_count'] == 1
            assert tag_metrics['PHI_count'] == 1
            assert tag_metrics['REAL_ratio'] == 0.5
            assert tag_metrics['coverage'] == 0.5
            assert tag_metrics['total_samples'] == 4
    
    def test_save_and_load(self):
        """Test saving and loading logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = StructuredLogger(temp_dir, "test_save")
            
            # Log some data
            for i in range(5):
                metrics = {'loss': 1.0 / (i + 1), 'epoch': i}
                logger.log_metrics(metrics, epoch=i)
            
            # Save
            log_file = logger.save()
            csv_file = logger.save_csv()
            
            assert os.path.exists(log_file)
            assert os.path.exists(csv_file)
            
            # Check JSON content
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            assert data['session_id'] == "test_save"
            assert len(data['logs']) == 5
            assert data['logs'][0]['metrics']['loss'] == 1.0
    
    def test_training_summary(self):
        """Test training summary generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = StructuredLogger(temp_dir)
            
            # Log training progression
            for epoch in range(10):
                metrics = {
                    'loss': 1.0 - epoch * 0.1,
                    'coverage': 0.5 + epoch * 0.05,
                    'lambda_rej': 0.1
                }
                logger.log_metrics(metrics, epoch=epoch)
            
            summary = logger.get_training_summary()
            
            assert 'session_info' in summary
            assert 'metrics' in summary
            assert summary['session_info']['total_logs'] == 10
            
            # Check metric summaries
            assert 'loss' in summary['metrics']
            assert summary['metrics']['loss']['trend'] == 'improving'  # Loss decreased
            assert summary['metrics']['coverage']['trend'] == 'declining'  # Coverage increased


class TestExperimentTracker:
    """Test experiment tracking functionality."""
    
    def test_experiment_lifecycle(self):
        """Test complete experiment lifecycle."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = ExperimentTracker(temp_dir)
            
            # Start experiment
            config = {'model': 'test', 'lr': 0.01}
            model_info = {'params': 100}
            
            logger = tracker.start_experiment("test_exp", config, model_info)
            
            assert tracker.current_experiment is not None
            assert tracker.current_experiment['name'] == "test_exp"
            
            # Log some data
            logger.log_metrics({'loss': 0.5}, epoch=1)
            
            # Finish experiment
            summary_file = tracker.finish_experiment()
            
            assert tracker.current_experiment is None
            assert len(tracker.experiment_history) == 1
            assert os.path.exists(summary_file)
    
    def test_multiple_experiments(self):
        """Test tracking multiple experiments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = ExperimentTracker(temp_dir)
            
            # Run multiple experiments
            for i in range(3):
                config = {'exp_id': i}
                model_info = {'version': i}
                
                logger = tracker.start_experiment(f"exp_{i}", config, model_info)
                logger.log_metrics({'metric': i * 0.1})
                tracker.finish_experiment()
            
            assert len(tracker.experiment_history) == 3
            
            summary = tracker.get_experiment_summary()
            assert summary['total_experiments'] == 3


class TestMetricsAggregator:
    """Test metrics aggregation functionality."""
    
    def test_collect_and_aggregate(self):
        """Test collecting and aggregating multiple runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple log files
            for i in range(3):
                session_data = {
                    'session_id': f'session_{i}',
                    'config': {'lr': 0.01 * (i + 1)},
                    'logs': [
                        {
                            'epoch': 0,
                            'step': 0,
                            'metrics': {'loss': 1.0 - i * 0.1, 'accuracy': 0.5 + i * 0.1}
                        }
                    ]
                }
                
                log_file = os.path.join(temp_dir, f"session_{i}_logs.json")
                with open(log_file, 'w') as f:
                    json.dump(session_data, f)
            
            # Aggregate
            aggregator = MetricsAggregator(temp_dir)
            log_files = aggregator.collect_runs()
            
            assert len(log_files) == 3
            
            aggregated = aggregator.aggregate_metrics(log_files)
            
            # Should group by configuration
            assert len(aggregated) == 3  # Different learning rates
            
            # Check aggregation structure
            for group_name, group_data in aggregated.items():
                assert 'n_runs' in group_data
                assert 'config' in group_data
                assert 'metrics_stats' in group_data


class TestConvenienceFunctions:
    """Test convenience logging functions."""
    
    def test_log_training_step(self):
        """Test the convenience function for logging training steps."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = StructuredLogger(temp_dir)
            
            tags = [TRTag.REAL, TRTag.REAL, TRTag.PINF]
            
            log_training_step(
                logger=logger,
                epoch=5,
                step=50,
                loss=0.25,
                tags=tags,
                coverage=0.67,
                lambda_rej=0.05,
                gradient_mode="HYBRID",
                delta=0.01,
                additional_metrics={'custom_metric': 1.23}
            )
            
            assert len(logger.session.logs) == 1
            log = logger.session.logs[0]
            
            assert log.epoch == 5
            assert log.step == 50
            assert log.metrics['loss'] == 0.25
            assert log.metrics['coverage'] == 0.67
            assert log.metrics['REAL_count'] == 2
            assert log.metrics['PINF_count'] == 1
            assert log.metrics['gradient_mode'] == "HYBRID"
            assert log.metrics['delta'] == 0.01
            assert log.metrics['custom_metric'] == 1.23


# Note: Plotting tests would require matplotlib and are more complex
# They are omitted here but would test:
# - TrainingCurvePlotter functionality
# - PoleVisualizationPlotter with mock models
# - ResidualAnalysisPlotter with sample data
# - ComparisonPlotter with mock results
# - Paper-ready figure generation
