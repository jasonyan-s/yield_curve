# Dynamic Market Simulation System

This repository provides a modular Python-based framework for simulating financial instruments, derivatives, and market curves. The architecture supports pricing, instrument management and dynamic market behavior through object-oriented design.

## Project Structure

```text
.
├── task.py                         # Main script or task runner
├── market_simulation.py           # Core simulation loop and market engine
├── instruments.py                 # Instrument factory and definitions
├── instrument_classes.py          # Base classes for financial instruments
├── derivatives.py                 # Derivatives pricing and classes (e.g., swaps, options)
├── curve_classes_and_functions.py # Yield/discount curves and interpolation methods
````

## Features

* **Instrument Factory**: Dynamic creation of instruments (bonds, swaps, futures, etc.)
* **Derivatives Support**: Includes pricing logic for various derivatives
* **Market Curves**: Term structure modeling and curve interpolation
* **Simulation Engine**: Time-stepped market simulation and PnL tracking
* **Modular Design**: Easy to extend with new instrument types or curve methods

## Getting Started

### Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

*(Note: `requirements.txt` not provided. Add libraries such as NumPy, pandas, etc.)*

### Running the Simulation

Run the main task file:

```bash
streamlit run task.py
```

Make sure to change system setting to light mode.
Make sure any referenced data files or configuration inputs are in place.

## File Descriptions

| File                             | Description                                              |
| -------------------------------- | -------------------------------------------------------- |
| `task.py`                        | Entry point for executing market tasks                   |
| `market_simulation.py`           | Runs the core simulation over time                       |
| `instruments.py`                 | Instrument construction and management                   |
| `instrument_classes.py`          | Contains base and derived classes for financial products |
| `derivatives.py`                 | Defines and prices derivatives instruments               |
| `curve_classes_and_functions.py` | Curve construction, interpolation, and shifting logic    |

## Example Use Case

The simulation framework could be used to:

* Price a portfolio of bonds and swaps
* Run Monte Carlo paths with stochastic interest rate curves
* Assess sensitivity of instruments to market moves
* Simulate a day-by-day evolution of a market environment

## License

MIT License 


