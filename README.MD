# Trading Bot with Technical Indicators

This project is a Python-based trading bot that uses various technical indicators and strategies to automate trading decisions. The bot integrates with Alpaca's trading API to fetch market data, execute trades, and manage positions.

## Features

- **RSI, CCI, and VZO Strategy**: Implements a strategy based on Relative Strength Index (RSI), Commodity Channel Index (CCI), and Volume Zone Oscillator (VZO).
- **Donchian Breakout Strategy**: Detects breakout opportunities using Donchian channels.
- **Automated Trading**: Submits buy orders based on predefined conditions and manages open positions.
- **Notifications**: Sends notifications for executed trades.
- **Customizable Parameters**: Allows users to configure strategy parameters such as RSI, CCI, and VZO thresholds.

## Prerequisites

- Python 3.8 or higher
- An Alpaca account with API keys
- Required Python libraries (see [Installation](#installation))

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/trading-bot.git
   cd trading-bot