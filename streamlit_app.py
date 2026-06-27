"""Deployment entry point for the microbiome-predict web app.

Used by Streamlit Community Cloud, Render, Railway, Hugging Face Spaces, etc.
Point the platform's main module / start command at this file::

    streamlit run streamlit_app.py

It puts the ``src`` layout on the import path and launches the UI, so the
package does not need to be pip-installed separately on the host.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from microbiome_predict.app import main  # noqa: E402

main()
