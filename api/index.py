# -*- coding: utf-8 -*-
"""
Vercel Serverless Function entry point
======================================
This file exposes the Flask app object to Vercel's Serverless Function builder.
"""

import sys
from pathlib import Path

# Add project root and src directories to Python module search path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "src"))

from web.app import app
