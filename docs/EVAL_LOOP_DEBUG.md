# Eval Loop Debugging Guide

## Overview
The eval loop is controlled by `run_actor_loop()` in `pipelinerl/actor.py`. It manages both training and test/eval loops using Python generators.

## Key Components

### 1. Initialization (lines 584-665)

```python
# Setup wandb, datasets, LLMs
train_dataset = dataset_loader(cfg.train_dataset_names, ...)
test_dataset = dataset_loader(cfg.test_dataset_names, ...)

# Create train and test ActorLoop objects
train_loop = ActorLoop(..., is_training=True)
test_loop = ActorLoop(..., is_training=False)

# Create generators (these don't run yet!)
train_loop_run = train_loop.run(dataset=train_dataset)  # Generator
test_loop_run = None  # Will be set when eval starts
```

**Key Point**: `train_loop.run()` returns a **generator**, not a running loop. It only executes when you call `next()` on it.

### 2. Main Control Loop (lines 669-708)

The main loop orchestrates both train and test loops:

```python
while True:
    # Step 1: Check if we should start test/eval loop
    # Step 2: If test loop exists, advance it one iteration
    # Step 3: Advance train loop one iteration
```

### 3. Eval Trigger Condition (lines 672-690)

**CRITICAL**: This is where eval starts. The condition is:

```python
next_regular_eval = (
    trainer_state.propagated_weight_version  # If first eval
    if last_regular_eval == -1
    else last_regular_eval + cfg.eval_every_n_versions  # Otherwise
)

if (
    cfg.eval_every_n_versions  # ⚠️ BUG: This is False when value is 0!
    and not cfg.debug.mode
    and trainer_state.propagated_weight_version >= next_regular_eval
    and test_dataset
    and test_loop_run is None
):
    logger.info("Create test loop")
    test_loop_run = test_loop.run(dataset=test_dataset)  # Create generator
    train_loop.is_scheduling_paused = True  # Pause training
```

**The Bug**: When `eval_every_n_versions: 0`, the check `cfg.eval_every_n_versions` evaluates to `False` in Python, so eval never starts!

**What Should Happen**:
- `eval_every_n_versions: 0` means "evaluate immediately at version 0"
- `next_regular_eval` should be `trainer_state.propagated_weight_version` (which is 0)
- Condition `trainer_state.propagated_weight_version >= next_regular_eval` should be `0 >= 0` = True
- But `cfg.eval_every_n_versions` is `False`, so the whole condition fails

**Workaround**: Since `eval_every_n_versions: 0` doesn't work (falsy value), use `eval_every_n_versions: 1` instead. This will:
- Trigger eval at version 0 (first eval when `last_regular_eval == -1`)
- Then eval every version after that
- Works correctly because `1` is truthy

### 4. Test Loop Execution (lines 692-701)

Once `test_loop_run` is created, the main loop advances it:

```python
if test_loop_run is not None:
    try:
        _ = next(test_loop_run)  # Advance test loop one iteration
    except StopIteration:
        # Test loop finished
        test_loop_run = None
        train_loop.is_scheduling_paused = False
        logger.info("Test loop finished")
```

### 5. Train Loop Execution (lines 703-708)

```python
try:
    _ = next(train_loop_run)  # Advance train loop one iteration
except StopIteration:
    logger.info("Train loop finished")
    break  # Exit main loop
```

**Important**: If train loop finishes (raises `StopIteration`), the main loop exits immediately, even if test loop is still running!

### 6. ActorLoop.run() Generator (lines 391-546)

This is a **generator function** that yields control back to the caller each iteration:

```python
def run(self, dataset):
    # Setup...
    while True:
        yield  # ⬅️ Yields control back to caller
        
        # Check if training is done
        if self.trainer_state.samples_processed >= samples_target:
            break  # Raises StopIteration
        
        # Submit problems to queue
        # Get results from queue
        # Update stats
        # Publish stats if needed
```

**Key Points**:
- First `next()` call executes up to first `yield` (line 447)
- Each subsequent `next()` executes one full iteration
- When `break` happens, generator raises `StopIteration`

### 7. Stats Publishing (lines 514-541)

Stats are published when:
- **Training**: When a new model version arrives (`trainer_version_to_publish is not None`)
- **Test**: When all rollouts finish (`finished_groups == expected_rollouts`)

```python
time_to_publish_test_stats = finished_groups == expected_rollouts

if time_to_publish_test_stats:
    self.publish_stats(...)  # Logs to wandb with "test_" prefix
```

**Important**: For test loop, stats are only published **once at the very end** when all rollouts complete!

### 8. TrainerState Updates (state.py)

`TrainerState` listens to messages from the trainer:

```python
# When trainer sends WeightUpdateSuccess:
trainer_state.propagated_weight_version = message.version  # e.g., 0

# When trainer sends TrainingDone:
trainer_state.training_done = True
```

## Debugging Checklist

### Check 1: Is eval condition met?
```python
# Add debug logging in run_actor_loop around line 678:
logger.info(f"eval_every_n_versions: {cfg.eval_every_n_versions}")
logger.info(f"propagated_weight_version: {trainer_state.propagated_weight_version}")
logger.info(f"next_regular_eval: {next_regular_eval}")
logger.info(f"Condition check: {cfg.eval_every_n_versions and ...}")
```

### Check 2: Is test_loop_run created?
Look for log message: `"Create test loop"`

### Check 3: Is test loop actually running?
Look for log messages: `"Published X test samples"`

### Check 4: Are stats being published?
Look for log message when `finished_groups == expected_rollouts`

### Check 5: Is train loop exiting too early?
Check if `"Train loop finished"` appears before test loop completes

## Common Issues

1. **eval_every_n_versions: 0 doesn't work**
   - Fix: Change condition to `cfg.eval_every_n_versions is not None and cfg.eval_every_n_versions >= 0`

2. **Train loop exits before test loop finishes**
   - Fix: Don't break main loop if test_loop_run is still active

3. **Stats not published**
   - Check: Did test loop actually finish? (`finished_groups == expected_rollouts`)
   - Check: Is wandb initialized? (`cfg.wandb.use_wandb`)

4. **Test loop never starts**
   - Check: Is `trainer_state.propagated_weight_version` set? (Should be 0)
   - Check: Is `test_dataset` not empty?
   - Check: Is `cfg.debug.mode` False?

## Flow Diagram

```
run_actor_loop()
├── Initialize train_loop.run() → generator (not running yet)
├── Initialize test_loop object (not running yet)
└── Main loop:
    ├── Check eval trigger condition
    │   └── If True: test_loop_run = test_loop.run() → generator
    ├── If test_loop_run exists:
    │   └── next(test_loop_run) → advances test loop one iteration
    │       ├── Submit problems to queue
    │       ├── Get results from queue
    │       ├── Update stats
    │       └── If all done: raise StopIteration
    └── next(train_loop_run) → advances train loop one iteration
        └── If training done: raise StopIteration → main loop breaks
```

## Configuration for Eval-Only or Minimal Training

Since `eval_every_n_versions: 0` doesn't work (falsy value), use `eval_every_n_versions: 1` instead.

### For Eval-Only (0 training iterations):

```yaml
finetune:
  max_train_steps: 0  # No training iterations

eval_every_n_versions: 1  # ⚠️ Use 1, not 0! (0 is falsy and won't trigger eval)
```

**How it works**:
- `max_train_steps: 0` means training completes immediately after version 0 is created
- `eval_every_n_versions: 1` triggers eval at version 0 (first eval when `last_regular_eval == -1`)
- Version 0 is created when the model is loaded, before any training happens
- Eval will run at version 0, then the system exits

### For Minimal Training (1 iteration):

```yaml
finetune:
  max_train_steps: 1  # One training iteration

eval_every_n_versions: 1  # Eval at version 0 and version 1
```

**How it works**:
- Training runs for 1 step, creating version 1
- Eval triggers at version 0 (before training) and version 1 (after training)
- Useful for debugging the eval pipeline with minimal training overhead

### Why `eval_every_n_versions: 1` works:

When `last_regular_eval == -1` (first eval):
- `next_regular_eval = trainer_state.propagated_weight_version` (which is 0)
- Condition: `trainer_state.propagated_weight_version >= next_regular_eval` → `0 >= 0` = True
- Since `eval_every_n_versions: 1` is truthy, the condition passes and eval starts

## Key Variables to Monitor

- `trainer_state.propagated_weight_version`: Should be 0 initially
- `test_loop_run`: Should be None initially, then a generator when eval starts
- `finished_groups`: Counts completed test rollouts
- `expected_rollouts`: Should be `len(test_dataset)` for test loop
- `train_loop.is_scheduling_paused`: Should be True when test loop is active
