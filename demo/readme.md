## List of Classes and Functions

### 1. `WeibullEstimator`
- **Methods**:
  - `__init__(self, default_k=0.3, default_lam=10000.0)`
  - `record_failure(self, ttf)`
  - `_sort_samples(self)`
  - `update_params(self)`
  - `get_params(self)`
  - `get_mean_ttf(self)`

### 2. `FixedCheckpointBaseline`
- **Methods**:
  - `__init__(self, fixed_interval=3600)`
  - `decide_allocation(self, simulation, task)`
  - `get_checkpoint_interval(self, task)`
  - `record_device_failure(self, ttf)`
  - `update_policy(self, current_time)`

### 3. `FixedReserveBaseline`
- **Methods**:
  - `__init__(self, fixed_reserve=2, checkpoint_interval=3600)`
  - `decide_allocation(self, simulation, task)`
  - `get_checkpoint_interval(self, task)`
  - `record_device_failure(self, ttf)`
  - `update_policy(self, current_time)`

### 4. `FixedRatioReserveBaseline`
- **Methods**:
  - `__init__(self, ratio=0.2, checkpoint_interval=3600, cluster_size=5)`
  - `decide_allocation(self, simulation, task)`
  - `get_checkpoint_interval(self, task)`
  - `record_device_failure(self, ttf)`
  - `update_policy(self, current_time)`

### 5. `AdaptiveWeibullScheduler`
- **Methods**:
  - `__init__(self, checkpoint_time=100.0, switching_cost=10.0, local_device_cost=0.001)`
  - `attach_simulation(self, sim)`
  - `record_device_failure(self, ttf)`
  - `update_policy(self, current_time)`
  - `compute_optimal_checkpoint(self, mttf, C)`
  - `decide_allocation(self, simulation, task)`
  - `get_checkpoint_interval(self, task)`

### 6. `Event`
- **Methods**:
  - `__init__(self, t, etype, data=None)`
  - `__lt__(self, other)`

### 7. `Task`
- **Methods**:
  - `__init__(self, arrival_time, duration, task_id=None, size=1)`
  - `reset_after_failure(self, current_time)`
  - `update_checkpoint(self, current_time)`

### 8. `Device`
- **Methods**:
  - `__init__(self, device_id, failure_times)`
  - `fail(self)`
  - `recover(self, time_now)`

### 9. `Simulation`
- **Methods**:
  - `__init__(
      self,
      n_devices=5,
      k_device=0.3,
      lam_device=10000,
      healing_time=2*24*3600,
      local_cost_rate=0.001,
      arrival_rate=0.0001,
      time_horizon=24*3600,
      schedulers=None,
      seed=42
    )`
  - `default_cloud_price(self, t)`
  - `sample_weibull(self, k, lam)`
  - `run(self)`
  - `initialize_events(self)`
  - `schedule_event(self, t, etype, data=None)`
  - `charge_cost_for_interval(self, dt)`
  - `process_event(self, evt)`
  - `pick_task_size(self)`
  - `pick_task_duration(self, size)`
  - `handle_task_arrive(self, evt)`
  - `handle_task_complete(self, evt)`
  - `handle_task_failure(self, evt)`
  - `handle_device_failure(self, evt)`
  - `handle_device_recovery(self, evt)`
  - `handle_checkpoint(self, evt)`
  - `start_task_on_device(self, sched_name, tsk, dev_id)`
  - `start_task_on_cloud(self, sched_name, tsk)`
  - `dispatch_waiting_tasks(self, sched_name)`
  - `get_active_local_tasks(self)`
  - `get_active_device_count(self)`
  - `print_stats(self)`

### 10. **Helper / Utility Functions**
- `fetch_csv_column(url, col_index=0)`
- `load_external_data()`
- `generate_placeholder_duration(size)`
