#!/bin/bash
source /home/ubuntu/miniconda3/bin/activate airflow-test  
cd /home/ubuntu/MLOps/MLOps/app/python_schedule
python predict_congestion_schedule.py

