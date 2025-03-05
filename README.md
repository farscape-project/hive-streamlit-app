# HIVE Streamlit app

This repository contains code for a streamlit-based web app used for demonstrating our HIVE simulation/surrogate system.

The code can be run with 
```bash
# you may wish to optionally create a new environment
pip install -r requirements.txt # maybe I missed some requirements
streamlit run app.py
```

Notes for running in container:
Needed to comment `MFEMLineSamplerAux` (maybe not available in this Apollo version).

If git-lfs not available, need to copy manually.

