# TODO: Algorithmic Trading Portfolio Project

## Project Overview
Hybrid deep learning and graph visibility model for algorithmic trading, trained and backtested for bias using Monte Carlo permutation simulation. Implementation in C++ with CMake, deep learning components in Python.

---

## Phase 1: Project Foundation & Structure

### Repository Setup
- [ ] Create `.env.sample` file with configuration templates
- [ ] Setup `.gitignore` for Python, C++, and data files
- [ ] Create `pyproject.toml` with TensorFlow ≥ 2.16, CPU/GPU extras
- [ ] Generate `requirements.txt` from pyproject.toml
- [ ] Setup directory structure as specified

### Directory Structure Creation
- [ ] Create `data/` directories (raw, processed, synthetic)
- [ ] Create `python/` module structure
- [ ] Create `cpp/` source and include directories
- [ ] Create `cmake/` configuration directory
- [ ] Create `scripts/` directory for automation
- [ ] Create `tests/` directories for Python and C++
- [ ] Create `docker/` directory for containerization
- [ ] Create `reports/` directory for output artifacts

---

## Phase 2: Python Deep Learning Components

### Core ML Models (`python/`)
- [ ] Implement `model.py` with two architectures:
  - [ ] Baseline: stacked LSTM + dense → δ-hedge outputs
  - [ ] SOTA: SigFormer (signature transform + Transformer encoder)
- [ ] Create `config.yaml` for model configuration
- [ ] Implement `data_loader.py` for data preprocessing

### Training & Optimization
- [ ] Implement `train.py` with:
  - [ ] Bootstrap MC paths from simulator
  - [ ] Minibatch policy-gradient loss (Deep Hedging paper)
  - [ ] Early-stopping on rolling Sharpe ratio
- [ ] Create `hyperopt.py` with Optuna/KerasTuner integration
- [ ] Setup SQLite database for trial logging (`trials.db`)
- [ ] Implement `export_model.py` for TensorFlow SavedModel export

### Backtesting Framework
- [ ] Implement `backtest.py` for strategy evaluation
- [ ] Create `walk_forward.py` for time-series cross-validation
- [ ] Implement walk-forward slicing with in-sample/out-sample splits

### Statistical Testing (`stats_tests.py`)
- [ ] Implement White's Reality Check permutation test
- [ ] Implement Deflated Sharpe Monte Carlo test
- [ ] Create p-value calculation functions
- [ ] Implement overfitting score (OS) calculation

### Utilities (`python/utils/`)
- [ ] Create `simulator.py` for Monte Carlo path generation:
  - [ ] Geometric Brownian Motion (GBM)
  - [ ] Heston model
  - [ ] Tick-to-path conversion
- [ ] Implement `metrics.py` for performance calculations (Sharpe, Sortino, max drawdown)
- [ ] Create `plotting.py` with standardized matplotlib helpers:
  - [ ] `plot_equity_curve()`
  - [ ] `plot_perf_dist()`
  - [ ] `plot_hyper_importance()`
- [ ] Implement `graph_viz.py` for trade execution visualization:
  - [ ] NetworkX → GraphViz trade-execution graph
  - [ ] TensorFlow computational graph dumping

---

## Phase 3: C++ Execution Layer

### CMake Configuration
- [ ] Create `CMakeLists.txt` for main project
- [ ] Implement `cmake/FindTensorFlow.cmake` for TF C API discovery
- [ ] Configure build system for TensorFlow C API integration

### Core C++ Components (`cpp/`)
- [ ] Implement `model_runner.hpp/cpp`:
  - [ ] Load SavedModel via TensorFlow C API
  - [ ] Expose `infer(const float* features, double* deltas)` function
  - [ ] Optimize for zero-copy latency
- [ ] Create `risk_guard.hpp/cpp`:
  - [ ] Hard position limits (max Δ)
  - [ ] Soft position limits (Σ exposure)
  - [ ] Risk validation logic
- [ ] Implement `api_client.hpp/cpp`:
  - [ ] REST/FIX wrapper interface
  - [ ] Support for Alpaca API
  - [ ] Support for Interactive Brokers API
- [ ] Create `utils.hpp` with helpers:
  - [ ] Scope timer for performance monitoring
  - [ ] JSON configuration parser
  - [ ] Logging utilities

### Main Execution Loop (`main.cpp`)
- [ ] Implement event-loop with uvloop/boost::asio:
  - [ ] Fetch latest market tick
  - [ ] Normalize and process through model_runner
  - [ ] Pass orders through risk_guard
  - [ ] Submit orders via api_client
  - [ ] Persist fills to Kafka/Parquet

---

## Phase 4: Testing & Validation

### Python Tests
- [ ] Setup pytest framework
- [ ] Create `test_stats_tests.py`:
  - [ ] Test permutation test correctness
  - [ ] Validate Monte Carlo simulations
  - [ ] Test data loader functionality
  - [ ] Verify metrics calculations

### C++ Tests
- [ ] Setup GoogleTest framework
- [ ] Create `risk_guard_test.cpp`:
  - [ ] Test edge cases for position limits
  - [ ] Verify no trades when limits exceeded
  - [ ] Test risk validation logic

---

## Phase 5: Infrastructure & Deployment

### Docker Configuration
- [ ] Create multi-stage `Dockerfile`:
  - [ ] CUDA base image
  - [ ] TensorFlow C API installation
  - [ ] Clang/GCC compiler setup
- [ ] Implement `entrypoint.sh` script
- [ ] Configure environment variables

### Automation Scripts
- [ ] Create `scripts/ingest.sh` for raw → processed ETL
- [ ] Implement `scripts/run_backtest.sh` for automated backtesting
- [ ] Setup CI/CD pipeline configuration

### Reporting & Visualization
- [ ] Setup report generation in `reports/YYYYMMDD-run-XX/` format
- [ ] Implement PNG output for plots
- [ ] Create CSV metrics export
- [ ] Setup TensorBoard scalar logging
- [ ] Implement overfitting detection (red flags on equity plots)

---

## Phase 6: Integration & Final Testing

### End-to-End Testing
- [ ] Run full pipeline from data ingestion to trade execution
- [ ] Validate model training and export process
- [ ] Test C++ inference integration
- [ ] Verify risk management controls

### Performance Optimization
- [ ] Profile model inference latency
- [ ] Optimize memory usage in C++ components
- [ ] Benchmark trade execution speed
- [ ] Validate real-time performance requirements

### Documentation
- [ ] Update README.md with setup instructions
- [ ] Document API interfaces
- [ ] Create user guide for backtesting
- [ ] Add deployment instructions

---

## Dependencies & Requirements

### Python Dependencies
- TensorFlow ≥ 2.16 (CPU/GPU)
- Optuna / KerasTuner
- NumPy, Pandas, Matplotlib
- NetworkX, GraphViz
- Pytest

### C++ Dependencies
- TensorFlow C API
- Boost.Asio / uvloop
- CMake ≥ 3.15
- GoogleTest
- JSON library (nlohmann/json)

### External Services
- Market data APIs (Alpaca, Interactive Brokers)
- Kafka (for trade persistence)
- SQLite (for trial logging)

---

**Current Status**: Repository scaffolding phase
**Next Priority**: Setup project structure and Python deep learning components 