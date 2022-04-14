#!/bin/bash
rm nohup.out
nohup streamlit run app.py --logger.level=info 2>streamlit_logs.txt &

