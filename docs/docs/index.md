# Pneuma: LLM-Based Data Discovery System for Tabular Data
[![Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TheDataStation/pneuma/blob/main/quickstart-colab.ipynb)
[![PyPI](https://img.shields.io/pypi/v/pneuma)](https://pypi.org/project/pneuma/)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue?logo=github)](https://github.com/TheDataStation/pneuma)

`Pneuma` is an LLM-powered data discovery system for tabular data. Given a natural language query,
`Pneuma` searches an indexed collection and retrieves the most relevant tables for the question. It performs this search by leveraging both **content** (columns and rows) and **context** (metadata) to match tables with questions.

## Getting Started

If you would like to try `Pneuma` without installation, you can use our [Colab notebook](https://colab.research.google.com/github/TheDataStation/pneuma/blob/main/quickstart.ipynb). For local installation, you may use an OpenAI API token or a local GPU **with at least 20 GB of VRAM** (to load and prompt both the LLM and embedding model).
