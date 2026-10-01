# Dataset

The StudentLife data (Dartmouth College) is **not included** in this repository because of its size.

## Download

Download the dataset from the official StudentLife page: https://studentlife.cs.dartmouth.edu/dataset.html

The dataset can also be found by searching the StudentLife paper (Wang et al., 2014; citation in the main README) on Google Scholar.

## Expected layout

Place the StudentLife files in this folder with the following structure (the scripts read these paths relative to the repository root):

```
dataset/
├── app_usage/
├── calendar/
├── call_log/
├── dinning/
├── education/
├── EMA/
│   └── response/
├── sensing/
├── sms/
└── survey/
```

Then run:

```bash
python src/01_data_cleaning/build_daily_dataset.py
```

This creates `dataset/final_dataset_complete.csv` (one row per user-day), which all the analysis scripts use.
