"""
Performance monitoring and logging for the RAG Research Assistant.

Tracks performance metrics, identifies bottlenecks, and provides
optimization recommendations.
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from functools import wraps
import psutil
import json
from pathlib import Path


class PerformanceMetrics:
    """Container for performance metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.end_time = None
        self.duration = None
        self.memory_usage = None
        self.cpu_usage = None
        self.success = True
        self.error_message = None
        self.metadata = {}
    
    def finish(self, success: bool = True, error_message: Optional[str] = None):
        """Mark the operation as finished."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error_message = error_message
        
        # Capture system metrics
        try:
            process = psutil.Process()
            self.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            self.cpu_usage = process.cpu_percent()
        except Exception:
            pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'memory_usage_mb': self.memory_usage,
            'cpu_usage_percent': self.cpu_usage,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


class PerformanceMonitor:
    """Performance monitoring system."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize performance monitor."""
        self.config = config
        self.enabled = config.get('enabled', True)
        self.log_file = config.get('log_file', 'logs/performance.log')
        self.max_history = config.get('max_history', 1000)
        self.alert_thresholds = config.get('alert_thresholds', {
            'duration': 30.0,  # seconds
            'memory': 500.0,   # MB
            'error_rate': 0.1  # 10%
        })
        
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Metrics storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_history))
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0.0,
            'success_count': 0,
            'error_count': 0,
            'avg_duration': 0.0,
            'min_duration': float('inf'),
            'max_duration': 0.0,
            'last_execution': None
        })
        
        # Setup logging
        if self.enabled:
            self._setup_logging()
    
    def _setup_logging(self):
        """Set up performance logging."""
        try:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create performance logger
            perf_logger = logging.getLogger('performance')
            perf_logger.setLevel(logging.INFO)
            
            # File handler for performance logs
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            perf_logger.addHandler(handler)
            
        except Exception as e:
            self.logger.error(f"Failed to setup performance logging: {e}")
            self.enabled = False
    
    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> PerformanceMetrics:
        """Start monitoring an operation."""
        if not self.enabled:
            return PerformanceMetrics()
        
        metrics = PerformanceMetrics()
        if metadata:
            metrics.metadata = metadata.copy()
        
        return metrics
    
    def finish_operation(self, operation_name: str, metrics: PerformanceMetrics, 
                        success: bool = True, error_message: Optional[str] = None):
        """Finish monitoring an operation."""
        if not self.enabled:
            return
        
        metrics.finish(success, error_message)
        
        with self._lock:
            # Update operation statistics
            stats = self.operation_stats[operation_name]
            stats['count'] += 1
            stats['total_duration'] += metrics.duration
            stats['last_execution'] = datetime.now().isoformat()
            
            if success:
                stats['success_count'] += 1
            else:
                stats['error_count'] += 1
            
            # Update duration statistics
            if metrics.duration < stats['min_duration']:
                stats['min_duration'] = metrics.duration
            if metrics.duration > stats['max_duration']:
                stats['max_duration'] = metrics.duration
            
            stats['avg_duration'] = stats['total_duration'] / stats['count']
            
            # Store metrics in history
            self.metrics_history[operation_name].append(metrics.to_dict())
            
            # Log performance data
            self._log_performance(operation_name, metrics)
            
            # Check for performance alerts
            self._check_alerts(operation_name, metrics, stats)
    
    def _log_performance(self, operation_name: str, metrics: PerformanceMetrics):
        """Log performance metrics."""
        try:
            perf_logger = logging.getLogger('performance')
            
            log_data = {
                'operation': operation_name,
                'duration': round(metrics.duration, 3),
                'success': metrics.success,
                'memory_mb': round(metrics.memory_usage, 2) if metrics.memory_usage else None,
                'cpu_percent': round(metrics.cpu_usage, 2) if metrics.cpu_usage else None,
                'metadata': metrics.metadata
            }
            
            if not metrics.success:
                log_data['error'] = metrics.error_message
            
            perf_logger.info(json.dumps(log_data))
            
        except Exception as e:
            self.logger.warning(f"Failed to log performance data: {e}")
    
    def _check_alerts(self, operation_name: str, metrics: PerformanceMetrics, stats: Dict[str, Any]):
        """Check for performance alerts."""
        alerts = []
        
        # Duration alert
        if metrics.duration > self.alert_thresholds['duration']:
            alerts.append(f"Slow operation: {operation_name} took {metrics.duration:.2f}s")
        
        # Memory alert
        if metrics.memory_usage and metrics.memory_usage > self.alert_thresholds['memory']:
            alerts.append(f"High memory usage: {operation_name} used {metrics.memory_usage:.2f}MB")
        
        # Error rate alert
        error_rate = stats['error_count'] / stats['count']
        if error_rate > self.alert_thresholds['error_rate']:
            alerts.append(f"High error rate: {operation_name} has {error_rate:.1%} error rate")
        
        # Log alerts
        for alert in alerts:
            self.logger.warning(f"Performance Alert: {alert}")
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for operations."""
        with self._lock:
            if operation_name:
                return self.operation_stats.get(operation_name, {}).copy()
            else:
                return {name: stats.copy() for name, stats in self.operation_stats.items()}
    
    def get_recent_metrics(self, operation_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent metrics for an operation."""
        with self._lock:
            history = self.metrics_history.get(operation_name, deque())
            return list(history)[-limit:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        with self._lock:
            summary = {
                'total_operations': sum(stats['count'] for stats in self.operation_stats.values()),
                'total_errors': sum(stats['error_count'] for stats in self.operation_stats.values()),
                'operations': {}
            }
            
            for operation_name, stats in self.operation_stats.items():
                if stats['count'] > 0:
                    summary['operations'][operation_name] = {
                        'count': stats['count'],
                        'avg_duration': round(stats['avg_duration'], 3),
                        'min_duration': round(stats['min_duration'], 3),
                        'max_duration': round(stats['max_duration'], 3),
                        'success_rate': stats['success_count'] / stats['count'],
                        'error_rate': stats['error_count'] / stats['count'],
                        'last_execution': stats['last_execution']
                    }
            
            return summary
    
    def get_bottlenecks(self, min_operations: int = 5) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        with self._lock:
            for operation_name, stats in self.operation_stats.items():
                if stats['count'] >= min_operations:
                    # High average duration
                    if stats['avg_duration'] > 5.0:
                        bottlenecks.append({
                            'type': 'slow_operation',
                            'operation': operation_name,
                            'avg_duration': stats['avg_duration'],
                            'count': stats['count'],
                            'severity': 'high' if stats['avg_duration'] > 15.0 else 'medium'
                        })
                    
                    # High error rate
                    error_rate = stats['error_count'] / stats['count']
                    if error_rate > 0.05:  # 5% error rate
                        bottlenecks.append({
                            'type': 'high_error_rate',
                            'operation': operation_name,
                            'error_rate': error_rate,
                            'error_count': stats['error_count'],
                            'total_count': stats['count'],
                            'severity': 'high' if error_rate > 0.2 else 'medium'
                        })
        
        # Sort by severity
        bottlenecks.sort(key=lambda x: (x['severity'] == 'high', x.get('avg_duration', 0)), reverse=True)
        return bottlenecks
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on performance data."""
        recommendations = []
        bottlenecks = self.get_bottlenecks()
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'slow_operation':
                operation = bottleneck['operation']
                duration = bottleneck['avg_duration']
                
                if 'search' in operation.lower():
                    recommendations.append(
                        f"Consider implementing caching for {operation} (avg: {duration:.2f}s)"
                    )
                elif 'api' in operation.lower():
                    recommendations.append(
                        f"Consider parallel API calls or rate limit optimization for {operation}"
                    )
                elif 'embedding' in operation.lower():
                    recommendations.append(
                        f"Consider batch processing or model optimization for {operation}"
                    )
                else:
                    recommendations.append(
                        f"Investigate performance bottleneck in {operation} (avg: {duration:.2f}s)"
                    )
            
            elif bottleneck['type'] == 'high_error_rate':
                operation = bottleneck['operation']
                error_rate = bottleneck['error_rate']
                
                recommendations.append(
                    f"Improve error handling for {operation} (error rate: {error_rate:.1%})"
                )
        
        # General recommendations based on overall stats
        summary = self.get_performance_summary()
        total_errors = summary['total_errors']
        total_operations = summary['total_operations']
        
        if total_operations > 0:
            overall_error_rate = total_errors / total_operations
            if overall_error_rate > 0.1:
                recommendations.append(
                    f"Overall error rate is high ({overall_error_rate:.1%}). "
                    "Consider implementing better error handling and retry mechanisms."
                )
        
        return recommendations
    
    def reset_stats(self, operation_name: Optional[str] = None):
        """Reset performance statistics."""
        with self._lock:
            if operation_name:
                if operation_name in self.operation_stats:
                    del self.operation_stats[operation_name]
                if operation_name in self.metrics_history:
                    self.metrics_history[operation_name].clear()
            else:
                self.operation_stats.clear()
                self.metrics_history.clear()
        
        self.logger.info(f"Reset performance stats for {operation_name or 'all operations'}")


def monitor_performance(operation_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Decorator for monitoring function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get performance monitor from config
            try:
                from ..utils.config import get_config
                config = get_config()
                monitor_config = config.get('performance_monitoring', {'enabled': True})
                monitor = PerformanceMonitor(monitor_config)
            except Exception:
                # Fallback to basic monitoring
                monitor = PerformanceMonitor({'enabled': True})
            
            # Start monitoring
            metrics = monitor.start_operation(operation_name, metadata)
            
            try:
                result = func(*args, **kwargs)
                monitor.finish_operation(operation_name, metrics, success=True)
                return result
            except Exception as e:
                monitor.finish_operation(operation_name, metrics, success=False, error_message=str(e))
                raise
        
        return wrapper
    return decorator


# Global performance monitor instance
_performance_monitor = None


def get_performance_monitor(config: Optional[Dict[str, Any]] = None) -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    
    if _performance_monitor is None:
        from ..utils.config import get_config
        monitor_config = config or get_config().get('performance_monitoring', {'enabled': True})
        _performance_monitor = PerformanceMonitor(monitor_config)
    
    return _performance_monitor