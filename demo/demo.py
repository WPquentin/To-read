"""
Simulation Framework with Online Weibull Estimation, Adaptive Scheduling, and Baseline Schedulers
that loads:
  - Task sizes from job_data.csv (first column)
  - Task durations from seconds1.csv, seconds2.csv, seconds4.csv, seconds8.csv
Provides fallback behavior if the files are unavailable.
"""

import math
import random
import csv
import io
import requests
import heapq
from collections import deque
from math import gamma, log, exp, factorial


# =============================================================================
# HELPER FUNCTIONS FOR DATA LOADING
# =============================================================================

def fetch_csv_column(url, col_index=0):
    """
    Fetch a CSV file from 'url' using requests, return a list of values in column col_index.
    Returns None if there's any error fetching or parsing the file.
    """
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        lines = resp.text.strip().splitlines()
        reader = csv.reader(lines)
        data = []
        for row in reader:
            if row:  # not empty
                val = row[col_index].strip()
                # Try converting to float
                try:
                    num = float(val)
                except ValueError:
                    continue
                data.append(num)
        return data if len(data) > 0 else None
    except requests.exceptions.RequestException:
        return None

def load_external_data():
    """
    Attempt to load:
      1) job_data.csv for the task sizes (first column)
      2) seconds1.csv, seconds2.csv, seconds4.csv, seconds8.csv for durations
    Fallback to default if files unavailable.
    Returns:
      (task_size_list, duration_dict)
        task_size_list: list of integers (1,2,4,8,...)
        duration_dict: dict with keys 1,2,4,8 => list of floats
                      or None if not loaded
    """
    # 1) load job sizes
    job_data_url = "https://raw.githubusercontent.com/WPquentin/To-read/refs/heads/main/job_data.csv"
    sizes_list = fetch_csv_column(job_data_url, col_index=0)
    if (not sizes_list) or len(sizes_list) < 1:
        # fallback: uniform choice among [1,2,4,8]
        sizes_list = None

    # 2) load durations for each GPU size
    duration_dict = {}
    fallback = False
    for s in [1,2,4,8]:
        url = f"https://raw.githubusercontent.com/WPquentin/To-read/refs/heads/main/seconds{s}.csv"
        col = fetch_csv_column(url, col_index=0)
        if (not col) or len(col)<1:
            fallback = True
            break
        duration_dict[s] = col

    if fallback:
        duration_dict = None

    return (sizes_list, duration_dict)

def generate_placeholder_duration(size):
    """
    Fallback if we can't load CSV durations.
    Example: random uniform scaled by 'size'.
    """
    return random.uniform(300, 600) * size


# =============================================================================
# WEIBULL ESTIMATOR
# =============================================================================

class WeibullEstimator:
    """
    Online Weibull estimation using 25% & 75% quantiles of TTF samples.
    """

    def __init__(self, default_k=0.3, default_lam=10000.0):
        self.failure_samples = []   # storing TTF (device operational lifetimes)
        self.k = default_k
        self.lam = default_lam
        self._sorted = False
        self._dirty = False

    def record_failure(self, ttf):
        """
        Record a new time-to-failure sample. Mark for re-fit.
        """
        self.failure_samples.append(ttf)
        self._sorted = False
        self._dirty = True

    def _sort_samples(self):
        if not self._sorted:
            self.failure_samples.sort()
            self._sorted = True

    def update_params(self):
        """
        Recompute k, lam from 25% & 75% quantiles:
          F(t25)=0.25 => (t25/lam)^k = -ln(0.75)
          F(t75)=0.75 => (t75/lam)^k = -ln(0.25)
        => ratio => (t75/t25)^k = [-ln(0.25)/ -ln(0.75)]
        => lam    = t25 / ( -ln(0.75) )^(1/k)
        """
        if len(self.failure_samples) < 2:
            self._dirty = False
            return

        self._sort_samples()
        n = len(self.failure_samples)
        i25 = max(0, min(int(0.25*n), n-1))
        i75 = max(0, min(int(0.75*n), n-1))

        t25 = self.failure_samples[i25]
        t75 = self.failure_samples[i75]
        if t25 < 1e-12:
            t25 = 1e-12
        if t75 < t25:
            t75 = t25 + 1e-12

        A = -math.log(0.75)  # ~0.28768
        B = -math.log(0.25)  # ~1.38629
        ratio = (B / A)

        denom = math.log(t75 / t25) if t75>t25 else 1e-9
        if abs(denom) < 1e-12:
            # fallback
            self.k = 0.3
            self.lam = 10000.0
            self._dirty = False
            return

        self.k = math.log(ratio)/denom
        if A<1e-12:
            A=1e-12
        self.lam = t25 / (A**(1.0/self.k))

        self._dirty = False

    def get_params(self):
        if self._dirty:
            self.update_params()
        return (self.k, self.lam)

    def get_mean_ttf(self):
        k, lam = self.get_params()
        return lam * gamma(1.0 + 1.0/k)


# =============================================================================
# SCHEDULERS
# =============================================================================

class FixedCheckpointBaseline:
    """
    Uses a fixed checkpoint interval, no dynamic reserve management.
    """

    def __init__(self, fixed_interval=3600):
        self.fixed_interval = fixed_interval

    def decide_allocation(self, simulation, task):
        """
        If a local device is free, use it; else go to cloud.
        """
        for dev in simulation.devices:
            if (not dev.is_failed) and (not dev.is_busy):
                return dev.device_id
        return 'cloud'

    def get_checkpoint_interval(self, task):
        return self.fixed_interval

    def record_device_failure(self, ttf):
        pass

    def update_policy(self, current_time):
        pass


class FixedReserveBaseline:
    """
    Maintains a fixed shared reserve_count, never changes. 
    Uses a fixed checkpoint interval.
    """

    def __init__(self, fixed_reserve=2, checkpoint_interval=3600):
        self.shared_reserve_count = fixed_reserve
        self.checkpoint_interval = checkpoint_interval

    def decide_allocation(self, simulation, task):
        normal_capacity = simulation.n_devices - self.shared_reserve_count
        active_local = sum(1 for d in simulation.devices if (not d.is_failed) and d.is_busy)
        if active_local < normal_capacity:
            # free device
            for dev in simulation.devices:
                if (not dev.is_failed) and (not dev.is_busy):
                    return dev.device_id
        # else check reserve
        if self.shared_reserve_count > 0:
            for dev in simulation.devices:
                if (not dev.is_failed) and (not dev.is_busy):
                    return dev.device_id
            return 'cloud'
        else:
            return 'cloud'

    def get_checkpoint_interval(self, task):
        return self.checkpoint_interval

    def record_device_failure(self, ttf):
        pass

    def update_policy(self, current_time):
        pass


class FixedRatioReserveBaseline:
    """
    A ratio-based shared reserve = ratio * n_devices (rounded up).
    Fixed checkpoint interval.
    """

    def __init__(self, ratio=0.2, checkpoint_interval=3600, cluster_size=5):
        self.ratio = ratio
        self.cluster_size = cluster_size
        self.checkpoint_interval = checkpoint_interval
        self.shared_reserve_count = int(ratio * cluster_size + 0.999999)

    def decide_allocation(self, simulation, task):
        normal_capacity = simulation.n_devices - self.shared_reserve_count
        active_local = sum(1 for d in simulation.devices if (not d.is_failed) and d.is_busy)
        if active_local < normal_capacity:
            for dev in simulation.devices:
                if (not dev.is_failed) and (not dev.is_busy):
                    return dev.device_id
        # else check reserve
        if self.shared_reserve_count > 0:
            for dev in simulation.devices:
                if (not dev.is_failed) and (not dev.is_busy):
                    return dev.device_id
            return 'cloud'
        else:
            return 'cloud'

    def get_checkpoint_interval(self, task):
        return self.checkpoint_interval

    def record_device_failure(self, ttf):
        pass

    def update_policy(self, current_time):
        pass


class AdaptiveWeibullScheduler:
    """
    Adaptive approach: 
      - Online Weibull distribution update (25%,75%).
      - Recompute checkpoint intervals by numeric search.
      - Adjust shared_reserve_count with Poisson approach every hour.
    """

    def __init__(self, checkpoint_time=100.0, switching_cost=10.0, local_device_cost=0.001):
        self.checkpoint_time = checkpoint_time
        self.switching_cost = switching_cost
        self.local_device_cost = local_device_cost
        self.weibull_estimator = WeibullEstimator()
        self.shared_reserve_count = 0
        self.sim = None

    def attach_simulation(self, sim):
        self.sim = sim

    def record_device_failure(self, ttf):
        self.weibull_estimator.record_failure(ttf)

    def update_policy(self, current_time):
        """
        1) Recompute (k, lam), then MTTF
        2) Update checkpoint intervals of active tasks
        3) Adjust shared reserve via Poisson
        """
        k, lam = self.weibull_estimator.get_params()
        device_mttf = lam * gamma(1.0 + 1.0/k) if k>0 else 10000.0

        # Update checkpoint for active local tasks
        for t in self.sim.get_active_local_tasks():
            n = max(1, t.num_devices_active)
            eff_mttf = device_mttf / float(n)
            x = self.compute_optimal_checkpoint(eff_mttf, self.checkpoint_time)
            t.checkpoint_interval = x

        # Poisson approach for next hour
        one_hour = 3600.0
        active_devs = self.sim.get_active_device_count()
        if device_mttf < 1e-9:
            mu = 0.0
        else:
            mu = active_devs * (one_hour / device_mttf)

        cloud_price_now = self.sim.cloud_cost_function(current_time)

        def prob_exceed_poisson(mean, r):
            s = 0.0
            for i in range(r+1):
                s += (mean**i)/math.factorial(i)
            return 1.0 - s*math.exp(-mean)

        # Expand if cost suggests
        while True:
            p_ex = prob_exceed_poisson(mu, self.shared_reserve_count)
            if (cloud_price_now * p_ex) > (self.local_device_cost + self.switching_cost):
                self.shared_reserve_count += 1
            else:
                break

        # Shrink if cost suggests
        while self.shared_reserve_count > 0:
            p_ex = prob_exceed_poisson(mu, self.shared_reserve_count)
            if (cloud_price_now * p_ex) < self.local_device_cost:
                self.shared_reserve_count -= 1
            else:
                break

    def compute_optimal_checkpoint(self, mttf, C):
        """
        Minimizes: obj(x) = 0.5*(x/mttf) + [C*exp(-x/mttf)/(C+x)]
        using a simple bracket search.
        """
        if mttf < 1e-6:
            return 100.0

        def obj(x):
            return 0.5*(x/mttf) + (C*math.exp(-x/mttf)/(C+x))

        guess = math.sqrt(2.0*mttf*C)
        a = 1.0
        b = max(2.0, guess*5.0)
        for _ in range(30):
            left = a + 0.382*(b-a)
            right = a + 0.618*(b-a)
            if obj(left) > obj(right):
                a = left
            else:
                b = right
        return (a+b)/2.0

    def decide_allocation(self, simulation, task):
        normal_cap = simulation.n_devices - self.shared_reserve_count
        busy_local = sum(1 for d in simulation.devices if (not d.is_failed) and d.is_busy)
        if busy_local < normal_cap:
            for dev in simulation.devices:
                if (not dev.is_failed) and (not dev.is_busy):
                    return dev.device_id

        if self.shared_reserve_count>0:
            # find a free device
            for dev in simulation.devices:
                if (not dev.is_failed) and (not dev.is_busy):
                    self.shared_reserve_count -= 1
                    return dev.device_id
            return 'cloud'
        else:
            return 'cloud'

    def get_checkpoint_interval(self, task):
        return task.checkpoint_interval


# =============================================================================
# SIMULATION DATA STRUCTURES
# =============================================================================

EVENT_TASK_ARRIVE     = 1
EVENT_TASK_COMPLETE   = 2
EVENT_TASK_FAILURE    = 3
EVENT_DEVICE_FAILURE  = 4
EVENT_DEVICE_RECOVERY = 5
EVENT_CHECKPOINT      = 6
EVENT_SCHEDULER_UPDATE= 7


class Event:
    def __init__(self, t, etype, data=None):
        self.t = t
        self.etype = etype
        self.data = data

    def __lt__(self, other):
        return self.t < other.t


class Task:
    def __init__(self, arrival_time, duration, task_id=None, size=1):
        self.task_id = task_id
        self.arrival_time = arrival_time
        self.full_duration = duration
        self.remaining_time = duration
        self.last_checkpoint_time = 0.0
        self.checkpoint_interval = 100.0
        self.is_running_on_cloud = False
        self.failure_time = None
        self.start_time = None
        self.finish_time = None
        self.num_failures = 0
        self.num_devices_active = size  # for MTTF scaling

    def reset_after_failure(self, current_time):
        lost = current_time - self.last_checkpoint_time
        self.remaining_time += lost

    def update_checkpoint(self, current_time):
        done_since_start = current_time - self.start_time
        done_since_last = done_since_start - (self.last_checkpoint_time - self.start_time)
        self.remaining_time -= done_since_last
        self.last_checkpoint_time = current_time


class Device:
    def __init__(self, device_id, failure_times):
        self.device_id = device_id
        self.failure_times = failure_times
        self.is_failed = False
        self.is_busy = False
        self.task = None
        self.recovery_time = None
        self.last_repair_time = 0.0

    def fail(self):
        self.is_failed = True
        self.is_busy = False
        self.task = None

    def recover(self, time_now):
        self.is_failed = False
        self.last_repair_time = time_now


# =============================================================================
# SIMULATION
# =============================================================================

class Simulation:
    def __init__(self, 
                 n_devices=5,
                 k_device=0.3, lam_device=10000,
                 healing_time=2*24*3600,
                 local_cost_rate=0.001,
                 arrival_rate=0.0001,
                 time_horizon=24*3600,
                 schedulers=None,
                 seed=42):
        random.seed(seed)
        self.current_time = 0.0
        self.time_horizon = time_horizon
        self.healing_time = healing_time
        self.local_cost_rate = local_cost_rate
        self.cloud_cost_function = self.default_cloud_price
        self.arrival_rate = arrival_rate

        # load external data (sizes + durations)
        self.task_size_data, self.duration_dict = load_external_data()

        # If no schedulers provided, define default
        if schedulers is None:
            schedulers = {
                'Adaptive': AdaptiveWeibullScheduler()
            }
        self.schedulers = schedulers
        for s in self.schedulers.values():
            if hasattr(s, 'attach_simulation'):
                s.attach_simulation(self)

        # Generate device data
        self.n_devices = n_devices
        self.devices = []
        for i in range(n_devices):
            fail_times = []
            t = 0.0
            while True:
                t += self.sample_weibull(k_device, lam_device)
                if t > time_horizon*10:
                    break
                fail_times.append(t)
            dev = Device(device_id=i, failure_times=fail_times)
            self.devices.append(dev)

        # Priority queue of events
        self.event_queue = []
        heapq.heapify(self.event_queue)

        # Each scheduler has its own waiting queue, cost stats, etc.
        self.wait_queue = {name: deque() for name in self.schedulers}
        self.total_local_cost = {name: 0.0 for name in self.schedulers}
        self.total_cloud_cost = {name: 0.0 for name in self.schedulers}
        self.total_tasks_completed = {name: 0 for name in self.schedulers}
        self.num_task_failures = {name: 0 for name in self.schedulers}
        self.cloud_active_tasks = {name: set() for name in self.schedulers}

        self.tasks = []   # track all tasks for final stats
        self.last_task_id = 0

    def default_cloud_price(self, t):
        period = 24*3600
        base_price = 0.02
        amplitude = 0.005
        t_mod = t % period
        return base_price + amplitude * math.sin(2*math.pi*(t_mod/period))

    def sample_weibull(self, k, lam):
        u = random.random()
        return lam * (-math.log(u))**(1.0/k)

    def run(self):
        self.initialize_events()

        # schedule hourly scheduler updates
        t = 3600.0
        while t < self.time_horizon*10:
            self.schedule_event(t, EVENT_SCHEDULER_UPDATE, None)
            t += 3600.0

        while self.event_queue:
            evt = heapq.heappop(self.event_queue)
            if evt.t > self.time_horizon*10:
                break

            dt = evt.t - self.current_time
            if dt>0:
                self.charge_cost_for_interval(dt)
                self.current_time = evt.t

            self.process_event(evt)
            # dispatch tasks for each scheduler
            for name in self.schedulers:
                self.dispatch_waiting_tasks(name)

        self.print_stats()

    def initialize_events(self):
        # Create arrivals
        arr_t = 0.0
        while True:
            delta = random.expovariate(self.arrival_rate)
            arr_t += delta
            if arr_t > self.time_horizon:
                break
            self.schedule_event(arr_t, EVENT_TASK_ARRIVE, None)

        # Create device failures
        for dev in self.devices:
            for ft in dev.failure_times:
                if ft > self.time_horizon*10:
                    break
                self.schedule_event(ft, EVENT_DEVICE_FAILURE, dev.device_id)

    def schedule_event(self, t, etype, data=None):
        heapq.heappush(self.event_queue, Event(t, etype, data))

    def charge_cost_for_interval(self, dt):
        for name in self.schedulers:
            # local
            num_local_busy = sum(1 for d in self.devices if d.is_busy and (not d.is_failed))
            self.total_local_cost[name] += num_local_busy * self.local_cost_rate * dt
            # cloud
            ccount = len(self.cloud_active_tasks[name])
            if ccount>0:
                mid = self.current_time + dt/2.0
                cprice = self.cloud_cost_function(mid)
                self.total_cloud_cost[name] += ccount*cprice*dt

    def process_event(self, evt):
        if evt.etype == EVENT_TASK_ARRIVE:
            self.handle_task_arrive(evt)
        elif evt.etype == EVENT_TASK_COMPLETE:
            self.handle_task_complete(evt)
        elif evt.etype == EVENT_TASK_FAILURE:
            self.handle_task_failure(evt)
        elif evt.etype == EVENT_DEVICE_FAILURE:
            self.handle_device_failure(evt)
        elif evt.etype == EVENT_DEVICE_RECOVERY:
            self.handle_device_recovery(evt)
        elif evt.etype == EVENT_CHECKPOINT:
            self.handle_checkpoint(evt)
        elif evt.etype == EVENT_SCHEDULER_UPDATE:
            for s in self.schedulers.values():
                if hasattr(s, 'update_policy'):
                    s.update_policy(self.current_time)
        else:
            pass

    # -------------------------------------------------------------------------
    # Generating a new task
    # -------------------------------------------------------------------------
    def pick_task_size(self):
        """
        If self.task_size_data is loaded, pick next from the list in sequence
        or wrap around. If not loaded, choose uniformly from [1,2,4,8].
        """
        if self.task_size_data:
            # in sequence or cycle
            size = self.task_size_data[self.last_task_id % len(self.task_size_data)]
            size = int(size)
            # ensure it's among {1,2,4,8}, else fallback
            if size not in [1,2,4,8]:
                size = random.choice([1,2,4,8])
            return size
        else:
            return random.choice([1,2,4,8])

    def pick_task_duration(self, size):
        """
        If we have self.duration_dict loaded, pick a random duration from it.
        Otherwise, fallback to generate_placeholder_duration.
        """
        if self.duration_dict and size in self.duration_dict:
            arr = self.duration_dict[size]
            if arr:
                return random.choice(arr)
        # fallback
        return generate_placeholder_duration(size)

    # -------------------------------------------------------------------------
    # EVENT HANDLERS
    # -------------------------------------------------------------------------
    def handle_task_arrive(self, evt):
        self.last_task_id += 1
        # pick size from job_data or fallback
        size = self.pick_task_size()
        # pick a random duration from CSV or fallback
        dur = self.pick_task_duration(size)

        tsk = Task(
            arrival_time=evt.t,
            duration=dur,
            task_id=self.last_task_id,
            size=size
        )
        # sample an internal failure time if desired
        tsk.failure_time = random.expovariate(1.0/20000.0)

        self.tasks.append(tsk)
        # put into each scheduler's waiting queue
        for name in self.schedulers:
            self.wait_queue[name].append(tsk)

    def handle_task_complete(self, evt):
        info = evt.data
        dev_id = info['device_id']
        tsk = info['task']
        sched_name = info['scheduler']

        if dev_id == -1:
            # cloud
            if tsk in self.cloud_active_tasks[sched_name]:
                self.cloud_active_tasks[sched_name].remove(tsk)
        else:
            dev = self.devices[dev_id]
            dev.is_busy = False
            dev.task = None

        tsk.finish_time = self.current_time
        self.total_tasks_completed[sched_name] += 1

    def handle_task_failure(self, evt):
        info = evt.data
        dev_id = info['device_id']
        tsk = info['task']
        sched_name = info['scheduler']

        tsk.num_failures += 1
        self.num_task_failures[sched_name] += 1

        if dev_id == -1:
            # cloud
            if tsk in self.cloud_active_tasks[sched_name]:
                self.cloud_active_tasks[sched_name].remove(tsk)
        else:
            dev = self.devices[dev_id]
            dev.is_busy = False
            dev.task = None

        tsk.reset_after_failure(self.current_time)
        self.wait_queue[sched_name].append(tsk)

    def handle_device_failure(self, evt):
        dev_id = evt.data
        dev = self.devices[dev_id]
        if dev.is_failed:
            return
        dev.fail()
        ttf = self.current_time - dev.last_repair_time

        # record in each scheduler
        for nm,sched in self.schedulers.items():
            if hasattr(sched, 'record_device_failure'):
                sched.record_device_failure(ttf)

        if dev.task:
            # whichever scheduler it was running for
            # we do naive approach again
            found_scheduler = None
            for snm in self.schedulers:
                if dev.task in self.cloud_active_tasks[snm]:
                    found_scheduler = snm
                    break
            if not found_scheduler:
                found_scheduler = 'Adaptive'  # fallback

            dev.task.num_failures += 1
            self.num_task_failures[found_scheduler] += 1
            dev.task.reset_after_failure(self.current_time)
            self.wait_queue[found_scheduler].append(dev.task)
            dev.task = None
            dev.is_busy = False

        rec_t = self.current_time + self.healing_time
        self.schedule_event(rec_t, EVENT_DEVICE_RECOVERY, dev_id)

    def handle_device_recovery(self, evt):
        dev_id = evt.data
        dev = self.devices[dev_id]
        dev.recover(self.current_time)

    def handle_checkpoint(self, evt):
        info = evt.data
        dev_id = info['device_id']
        tsk = info['task']
        sched_name = info['scheduler']

        if dev_id == -1:
            # cloud
            if tsk in self.cloud_active_tasks[sched_name]:
                tsk.update_checkpoint(self.current_time)
                nxt = self.current_time + tsk.checkpoint_interval
                fin = self.current_time + tsk.remaining_time
                if nxt < fin:
                    self.schedule_event(nxt, EVENT_CHECKPOINT, {
                        'device_id': -1, 'task': tsk, 'scheduler': sched_name
                    })
        else:
            dev = self.devices[dev_id]
            if dev.task == tsk and not dev.is_failed:
                tsk.update_checkpoint(self.current_time)
                nxt = self.current_time + tsk.checkpoint_interval
                fin = self.current_time + tsk.remaining_time
                if nxt<fin:
                    self.schedule_event(nxt, EVENT_CHECKPOINT, {
                        'device_id': dev_id, 'task': tsk, 'scheduler': sched_name
                    })

    # -------------------------------------------------------------------------
    # START TASK
    # -------------------------------------------------------------------------
    def start_task_on_device(self, sched_name, tsk, dev_id):
        dev = self.devices[dev_id]
        dev.is_busy = True
        dev.task = tsk
        tsk.start_time = self.current_time
        tsk.last_checkpoint_time = self.current_time
        tsk.is_running_on_cloud = False
        fin = self.current_time + tsk.remaining_time
        self.schedule_event(fin, EVENT_TASK_COMPLETE, {
            'device_id': dev_id, 'task': tsk, 'scheduler': sched_name
        })
        cp = tsk.checkpoint_interval
        if (self.current_time+cp)<fin:
            self.schedule_event(self.current_time+cp, EVENT_CHECKPOINT, {
                'device_id': dev_id, 'task': tsk, 'scheduler': sched_name
            })
        if tsk.failure_time:
            tf = self.current_time + tsk.failure_time
            if tf<fin:
                self.schedule_event(tf, EVENT_TASK_FAILURE, {
                    'device_id': dev_id, 'task': tsk, 'scheduler': sched_name
                })

    def start_task_on_cloud(self, sched_name, tsk):
        tsk.start_time = self.current_time
        tsk.last_checkpoint_time = self.current_time
        tsk.is_running_on_cloud = True
        self.cloud_active_tasks[sched_name].add(tsk)
        fin = self.current_time + tsk.remaining_time
        self.schedule_event(fin, EVENT_TASK_COMPLETE, {
            'device_id': -1, 'task': tsk, 'scheduler': sched_name
        })
        cp = tsk.checkpoint_interval
        if (self.current_time+cp)<fin:
            self.schedule_event(self.current_time+cp, EVENT_CHECKPOINT, {
                'device_id': -1, 'task': tsk, 'scheduler': sched_name
            })
        if tsk.failure_time:
            tf = self.current_time + tsk.failure_time
            if tf<fin:
                self.schedule_event(tf, EVENT_TASK_FAILURE, {
                    'device_id': -1, 'task': tsk, 'scheduler': sched_name
                })

    def dispatch_waiting_tasks(self, sched_name):
        sched = self.schedulers[sched_name]
        queue = self.wait_queue[sched_name]
        while queue:
            tsk = queue[0]
            alloc = sched.decide_allocation(self, tsk)
            if alloc is None:
                break
            queue.popleft()
            tsk.checkpoint_interval = sched.get_checkpoint_interval(tsk)
            if alloc=='cloud':
                self.start_task_on_cloud(sched_name, tsk)
            else:
                self.start_task_on_device(sched_name, tsk, alloc)

    # -------------------------------------------------------------------------
    # UTILITY
    # -------------------------------------------------------------------------
    def get_active_local_tasks(self):
        """
        Return tasks that are running on local devices (not cloud).
        """
        active=[]
        for d in self.devices:
            if d.is_busy and (not d.is_failed) and d.task:
                active.append(d.task)
        return active

    def get_active_device_count(self):
        """
        Return how many devices are up (not failed).
        """
        return sum(1 for d in self.devices if not d.is_failed)

    # -------------------------------------------------------------------------
    # STATS
    # -------------------------------------------------------------------------
    def print_stats(self):
        print(f"==== Simulation Complete at time = {self.current_time:.2f} ====")
        for name in self.schedulers:
            print(f"--- Scheduler: {name} ---")
            print(f"   Tasks completed: {self.total_tasks_completed[name]}")
            print(f"   Task failures:   {self.num_task_failures[name]}")
            print(f"   Local cost:      {self.total_local_cost[name]:.4f}")
            print(f"   Cloud cost:      {self.total_cloud_cost[name]:.4f}")
            # average turnaround
            completed_times = []
            for t in self.tasks:
                if t.finish_time is not None:
                    completed_times.append(t.finish_time - t.arrival_time)
            if completed_times:
                avg_turn = sum(completed_times)/len(completed_times)
            else:
                avg_turn = 0
            print(f"   Avg turnaround:  {avg_turn:.2f} s\n")


# =============================================================================
# MAIN EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example set of schedulers for comparison
    schedulers = {
        'FixedCheckpoint': FixedCheckpointBaseline(fixed_interval=600.0),
        'FixedReserve': FixedReserveBaseline(fixed_reserve=1, checkpoint_interval=600.0),
        'FixedRatio': FixedRatioReserveBaseline(ratio=0.3, checkpoint_interval=600.0, cluster_size=5),
        'Adaptive': AdaptiveWeibullScheduler(checkpoint_time=100.0,
                                             switching_cost=10.0,
                                             local_device_cost=0.001)
    }

    sim = Simulation(
        n_devices=5,
        arrival_rate=0.0005,   # ~1 task every 2000s
        time_horizon=3600,     # simulate 1 hour
        schedulers=schedulers,
        seed=123
    )
    sim.run()
