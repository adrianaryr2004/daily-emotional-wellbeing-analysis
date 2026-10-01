#Libreries
import pandas as pd
import os
import glob
import json
import numpy as np
from datetime import datetime
#Time zone (New Hampshire, United States)
TZ = 'America/New_York'

#FUNCTION TO CONVERT TIMESTAMP TO LOCAL DATE
def timestamp_to_local_date(timestamp_series):
    ts = pd.to_numeric(timestamp_series, errors='coerce')
    return (
        pd.to_datetime(ts, unit='s', utc=True)
        .dt.tz_convert(TZ)
        .dt.date
    )

# === PART 1: Process app_usage ===
print("\nProcessing app_usage...")
ruta_app = "dataset/app_usage"
columnas_app = ["timestamp", "RUNNING_TASKS_baseActivity_mPackage", "RUNNING_TASKS_numActivities"]
datos_usuarios = []
def most_frequent(series):
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else None #returns the first mode or None
for archivo in sorted(glob.glob(os.path.join(ruta_app, "running_app_u[0-9][0-9].csv"))): #find all files that match that pattern
    try:
        df = pd.read_csv(archivo, usecols=columnas_app) #reads only the necessary columns
        user_id = int(os.path.basename(archivo).split("_")[-1].replace(".csv", "").lstrip('u'))#Extract the ID from the file name
        # clean timestamp
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce") #converts to numeric
        df = df.dropna(subset=["timestamp"]) # drop Nans rows
        # local date
        df["date"] = timestamp_to_local_date(df["timestamp"])
        # creates a column for app
        df["app"] = df["RUNNING_TASKS_baseActivity_mPackage"]
        # numActivities , if there are unusual values, they are left as NaN
        df["RUNNING_TASKS_numActivities"] = pd.to_numeric(df["RUNNING_TASKS_numActivities"], errors="coerce")

        resumen = df.groupby("date").agg( #in each day
            total_apps_used=("app", "nunique"), #how many different apps did he use that day
            most_used_app=("app",most_frequent), #the most repeated app that day 
            num_sessions=("app", "count"), #how many rows(sesions)
            tiempo_uso_total=("RUNNING_TASKS_numActivities","sum")
        ).reset_index()

        resumen["user_id"] = user_id
        resumen["tiempo_uso_total"] = resumen["tiempo_uso_total"].fillna(0) # if no time , 0

        datos_usuarios.append(resumen)

    except Exception as e:
        print(f"Error reading {archivo}: {e}")

if len(datos_usuarios) > 0: #is there is data , concat
    df_app = pd.concat(datos_usuarios, ignore_index=True)
else: #create an empty dataframe
    df_app = pd.DataFrame()
print("App usage processed")


# === PART 2: Process calendar ===
print("\nProcessing calendar...")
ruta_calendar = "dataset/calendar"
datos_calendar = []
for archivo in sorted(glob.glob(os.path.join(ruta_calendar, "calendar_u[0-9][0-9].csv"))):
    try:
        df = pd.read_csv(archivo)
        user_id = int(os.path.basename(archivo).split("_")[-1].replace(".csv", "").lstrip('u'))
        df['date'] = pd.to_datetime(df['DATE'], errors='coerce').dt.date
        df = df.dropna(subset=["date"])
        # Total number of calendar rows/events per day
        resumen = (
            df.groupby("date")
            .size()
            .reset_index(name="calendar_event_count")
        )
        resumen["user_id"] = user_id
        datos_calendar.append(resumen)
    except Exception as e:
        print(f"Error reading {archivo}: {e}")
# Combine all users
if len(datos_calendar) > 0:
    df_calendar = pd.concat(datos_calendar, ignore_index=True)
    print("Calendar processed")
else:
    df_calendar = pd.DataFrame(columns=["date", "calendar_event_count", "user_id"])
    print("No valid files were found in calendar.")

# === PART 3: Processing call_log ===
print("\nProcessing call_log...")
ruta_call = "dataset/call_log"
datos_call = []
for archivo in sorted(glob.glob(os.path.join(ruta_call, "call_log_u[0-9][0-9].csv"))):
    try:
        df = pd.read_csv(archivo)
        user_id = int(os.path.basename(archivo).split("_")[-1].replace(".csv", "").lstrip("u"))
        if 'timestamp' not in df.columns:
            raise ValueError("Columna 'timestamp' no encontrada")
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        # Convertir timestamp a fecha local
        df['date'] = (
            pd.to_datetime(df['timestamp'], unit='s', utc=True)
            .dt.tz_convert(TZ)
            .dt.date
        )
        resumen = df.groupby('date').size().reset_index(name='total_calls')  #Count calls per day
        resumen['user_id'] = user_id
        datos_call.append(resumen)
    except Exception as e:
        print(f"Error reading {archivo}: {e}")

df_call = pd.concat(datos_call, ignore_index=True) if datos_call else pd.DataFrame()
if not df_call.empty:
    print("Call log processed.")
else:
    print("No valid files were found in call_log.")

# === PART 4: Procesar dinning ===
print("\n Processing dinning...")
ruta_dinning = "dataset/dinning"
datos_dinning = []
for archivo in sorted(glob.glob(os.path.join(ruta_dinning, "u[0-9][0-9].txt"))):
    try:
        with open(archivo, "r", encoding="utf-8") as f:
            lineas = f.readlines()
        user_id = int(os.path.basename(archivo).replace(".txt","").replace("u",""))
        df["user_id"] = user_id
        eventos = []
        for linea in lineas:
            partes = linea.strip().split(",")
            if len(partes) >= 3:
                fecha = partes[0][:10]
                tipo_comida = partes[-1]
                eventos.append((fecha, tipo_comida))

        df = pd.DataFrame(eventos, columns=["date", "meal_type"])
        df["date"] = pd.to_datetime(df["date"]).dt.date  
        df["user_id"] = user_id
        #for each row
        df["is_breakfast"] = (df["meal_type"] == "Breakfast").astype(int)
        df["is_lunch"]     = (df["meal_type"] == "Lunch").astype(int)
        df["is_supper"]    = (df["meal_type"] == "Supper").astype(int)

        resumen = df.groupby(["user_id", "date"]).agg(
            total_meals=("meal_type", "count"),
            breakfast_events=("is_breakfast", "sum"),
            lunch_events=("is_lunch", "sum"),
            supper_events=("is_supper", "sum")
        ).reset_index()
        datos_dinning.append(resumen)
    except Exception as e:
        print(f"Error reading {archivo}: {e}")
df_dinning = pd.concat(datos_dinning, ignore_index=True) if datos_dinning else pd.DataFrame()
if not df_dinning.empty:
    print("Dinning processed.")
else:
    print("No valid files were found in dinning.")

#Education
# === PART 5: Process class.csv === 
#Academic workload of each student
print("\nProcessing class data...")
try:
    with open("dataset/education/class.csv", "r", encoding="utf-8") as f:
        lineas = f.readlines()[1:] #remove the first line(no relevant info)
    data = [line.strip().split(",") for line in lineas if line.strip()] #Goes through each line and separates them by commas
    #Convert that list of lists into a DataFrame without column names yet.
    df_class_csv = pd.DataFrame(data)
    #Number of columns
    num_cols = df_class_csv.shape[1]
    #The remaining ones are class1, class2, … depending on how many columns there are.
    df_class_csv.columns = ['uid'] + [f'class{i}' for i in range(1, num_cols)]
    valid_prefixes = (
        "COSC", "ENGS", "MATH", "BIOL", "ANTH", "PSYC", "CHIN", "TUCK",
        "FILM", "EARS", "SPAN", "NAS", "M&SS", "ENGL", "ECON", "MUS",
        "GERM", "LAT", "JAPN"
    )
    #Columns where the subjects are listed
    class_columns = df_class_csv.columns[1:]
    num_classes = []
    #traverse the dataframe row by row
    for _, row in df_class_csv.iterrows():
        count = 0
        for course in row[class_columns]:
            if isinstance(course, str): #is text
                course = course.strip()#clean spaces
                if course.startswith(valid_prefixes):
                    count += 1
        num_classes.append(count)
    df_class_csv["num_classes"] = num_classes #Now each user has their own number of subjects.
    # user_id into numeric
    df_class_csv["user_id"] = pd.to_numeric(df_class_csv["uid"].astype(str).str.lstrip("u"), errors="coerce")
    df_class_csv = df_class_csv.dropna(subset=["user_id"])
    df_class_csv["user_id"] = df_class_csv["user_id"].astype(int)
    df_class_csv = df_class_csv[["user_id", "num_classes"]]
    print("Class data processed.")
except Exception as e:
    print(f"Error processing class.csv: {e}")
    df_class_csv = pd.DataFrame(columns=['user_id', 'num_classes'])

# === PART 6: Process grades.csv ===
print("\nProcessing grades data...")
try:
    df_grades = pd.read_csv("dataset/education/grades.csv")
    #clear column names
    df_grades.columns = df_grades.columns.str.strip()
    df_grades = df_grades.rename(columns={'uid': 'user_id', 'gpa all': 'grade_mean'})
    #converts user_id
    df_grades["user_id"] = pd.to_numeric(df_grades["user_id"].astype(str).str.lstrip("u"), errors="coerce")
    df_grades = df_grades.dropna(subset=["user_id"])
    df_grades["user_id"] = df_grades["user_id"].astype(int)
    #We ensure that the columns are numeric, and if not, NAN.
    for c in ["grade_mean", "gpa 13s", "cs 65"]:
        if c in df_grades.columns:
            df_grades[c] = pd.to_numeric(df_grades[c], errors="coerce")
    #Check if those columns exist; if they do, calculate the standard deviation per row
    if 'gpa 13s' in df_grades.columns and 'cs 65' in df_grades.columns:
        df_grades['grade_std'] = df_grades[['gpa 13s', 'cs 65']].std(axis=1)
    else:
        df_grades['grade_std'] = pd.NA
    df_grades["grade_std"] = df_grades["grade_std"].round(2)
    df_grades = df_grades[['user_id', 'grade_mean', 'grade_std']]
    print("Grades data processed..")
    #The variable grade_std represents the variability between different academic metrics of the same student, and is used as an approximation of the stability of academic performance.
    #Measures how much a student's grades change between different assessments(BIG DIFFERENCE==GREAT VALUE).
except Exception as e:
    print(f"Error processing grades.csv: {e}")
    df_grades = pd.DataFrame(columns=['user_id', 'grade_mean', 'grade_std'])
# === PART 7: Process deadlines.csv ===
print("\nProcessing deadlines data...")
try:
    df_deadlines = pd.read_csv("dataset/education/deadlines.csv")
    df_deadlines.columns = df_deadlines.columns.str.strip()
    #Converts the format from wide to long
    df_deadlines = df_deadlines.melt(id_vars=['uid'], var_name='date', value_name='num_deadlines_today')
    #user_id correct
    df_deadlines["user_id"] = pd.to_numeric(
        df_deadlines["uid"].astype(str).str.lstrip("u"),
        errors="coerce"
    )
    df_deadlines = df_deadlines.dropna(subset=["user_id"])
    df_deadlines["user_id"] = df_deadlines["user_id"].astype(int)
    #Date format 
    df_deadlines['date'] = pd.to_datetime(df_deadlines['date']).dt.date
    #we filter only days with deadlines to make the merge easier, the nans will then be filled with 0s at the end.
    df_deadlines["num_deadlines_today"] = pd.to_numeric(
        df_deadlines["num_deadlines_today"],
        errors="coerce"
    ).fillna(0).astype(int)
    df_deadlines = df_deadlines[df_deadlines["num_deadlines_today"] > 0]
    df_deadlines = df_deadlines[["user_id", "date", "num_deadlines_today"]]
    print("Deadlines data processed.")
except Exception as e:
    print(f"Error processing deadlines.csv: {e}")
    df_deadlines = pd.DataFrame(columns=['user_id', 'date', 'num_deadlines_today'])
# === PART 8: Process piazza.csv ===
print("\nProcessing piazza data...")
try:
    df_piazza = pd.read_csv("dataset/education/piazza.csv")
    #Modify columns names
    df_piazza = df_piazza.rename(columns={
        'uid': 'user_id',
        'questions': 'num_posts',
        'answers': 'num_responses'
    })
    #Piazza is an online platform for students
    #Num_posts :many, is a person with many doubts or initiative
    #Num_responses:answers given to others, may be a collaborative person or someone who is proficient in a subject
    #piazzar_score : How involved is a student in online academic activity.
    #In final, zeros are not added; they are left as nan because otherwise it would represent that the user participated in Piazza but did nothing.
    
    #convert user_id
    df_piazza["user_id"] = pd.to_numeric(
        df_piazza["user_id"].astype(str).str.lstrip("u"),
        errors="coerce"
    )
    df_piazza = df_piazza.dropna(subset=["user_id"])
    df_piazza["user_id"] = df_piazza["user_id"].astype(int)
    #Create a new piazza_score column which is simply the sum of posts and replies made by the user.
    df_piazza['piazza_score'] = df_piazza['num_posts'] + df_piazza['num_responses']
    df_piazza = df_piazza[['user_id', 'num_posts', 'num_responses', 'piazza_score']]
    print("Piazza data processed.")
except Exception as e:
    print(f"Error processing piazza.csv: {e}")
    df_piazza = pd.DataFrame(columns=['user_id', 'num_posts', 'num_responses', 'piazza_score'])

#EMA
# === PART 0: Process Activity Data ===
print("Processing Activity data...")
ruta_activity = "dataset/EMA/response/Activity"
datos_activity = []
columnas_validas = ['working', 'relaxing', 'other_working', 'other_relaxing', 'Social2']
for archivo in sorted(glob.glob(os.path.join(ruta_activity, "Activity_u[0-9][0-9].json"))):
    try:
        with open(archivo) as f:
            activity_data = json.load(f)
        df = pd.DataFrame(activity_data)
        uid = os.path.basename(archivo).split("_")[-1].replace(".json", "")
        user_id = pd.to_numeric(str(uid).lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            print(f"Could not parse user_id in {archivo}")
            continue
        user_id = int(user_id)
        #If I don't know the day of the response, I can't use this information. This way we avoid noise.
        if 'resp_time' not in df.columns:
            print(f" File {archivo} ignored: coes not contain 'resp_time'")
            continue
        df['date'] = timestamp_to_local_date(df['resp_time'])
        #Convert the time to local date and add user
        df['user_id'] = user_id
        #Creates a new list with the important columns that actually exist in the data frame
        cols_presentes = [c for c in columnas_validas if c in df.columns]
        #Convert only existing columns to numeric
        for c in cols_presentes:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        columnas_a_guardar = ["user_id", "date"] + cols_presentes
        datos_activity.append(df[columnas_a_guardar])

    except Exception as e:
        print(f"Error processing {archivo}: {e}")
    #In the end, it's not filled with 0, because it's a scale; it will be filled with nan.

#concatenate all the dataframes into one large one
df_activity = pd.concat(datos_activity, ignore_index=True) if datos_activity else pd.DataFrame()

if not df_activity.empty:
    cols_to_agg = [c for c in columnas_validas if c in df_activity.columns]
    #Createa a list with the columns I want to use that actually exist.
    agg_dict = {c: "mean" for c in cols_to_agg}
    #We created a dictionary that calculates the average for each column.
    #Group and calculate the average of each column within that group
    daily_activity = df_activity.groupby(['user_id', 'date'], as_index=False).agg(agg_dict)
    print("Activity data processed.")
else:
    print("No valid files were found in Activity.")
    daily_activity = pd.DataFrame()
# === PART 0.1: Process Class Data ===
print("\nProcessing data of Class...")
ruta_class = "dataset/EMA/response/Class"
datos_class = []
for archivo in sorted(glob.glob(os.path.join(ruta_class, "Class_u[0-9][0-9].json"))):
    try:
        with open(archivo) as f:
            class_data = json.load(f)

        df = pd.DataFrame(class_data)
        uid = os.path.basename(archivo).split("_")[-1].replace(".json", "")
        user_id = pd.to_numeric(str(uid).lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        if 'resp_time' not in df.columns:
            print(f"File {archivo} ignored: does not contain 'resp_time'")
            continue
        df['date'] = timestamp_to_local_date(df['resp_time']) #create local date
        df['user_id'] = user_id #add the user id
        columnas_class = ['experience', 'hours', 'due']
        for col in columnas_class:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') #converts ir into a number or Nan
        columnas_a_guardar = ['user_id', 'date'] + [col for col in columnas_class if col in df.columns]
        datos_class.append(df[columnas_a_guardar])
    except Exception as e:
        print(f"Error processing {archivo}: {e}")

df_class = pd.concat(datos_class, ignore_index=True) if datos_class else pd.DataFrame() #concatenates all the dataframes
if not df_class.empty:
    agg_dict = {}
    if "experience" in df_class.columns: agg_dict["experience"] = "mean" #if this column exists , add it to the dictionary
    if "hours" in df_class.columns:      agg_dict["hours"] = "sum"
    if "due" in df_class.columns:        agg_dict["due"] = "mean"

    daily_class = df_class.groupby(["user_id", "date"], as_index=False).agg(agg_dict) #for each day and user apply the mean of the experience...
    print("Class data processed.")
else:
    print("No valid files were found in Class.")
    daily_class = pd.DataFrame()


# === PART 0.2: Process Class 2 Data ===
print("\nProcessing data of Class 2...")
ruta_class2 = "dataset/EMA/response/Class 2"
datos_class2 = []
for archivo in sorted(glob.glob(os.path.join(ruta_class2, "Class 2_u[0-9][0-9].json"))):
    try:
        with open(archivo) as f:
            class2_data = json.load(f)
        df = pd.DataFrame(class2_data)
        uid = os.path.basename(archivo).split("_")[-1].replace(".json", "")
        user_id = pd.to_numeric(str(uid).lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        if 'resp_time' not in df.columns:
            print(f"File {archivo} ignored:does not have 'resp_time'")
            continue
        df['date'] = timestamp_to_local_date(df['resp_time'])
        df['user_id'] = user_id
        columnas_class2 = ['challenge', 'effort', 'grade']
        #We select the possible columns and convert them to numeric.
        cols_presentes = [c for c in columnas_class2 if c in df.columns]
        for c in cols_presentes:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        columnas_a_guardar = ["user_id", "date"] + cols_presentes
        datos_class2.append(df[columnas_a_guardar])
    except Exception as e:
        print(f"Error processing {archivo}: {e}")
df_class2 = pd.concat(datos_class2, ignore_index=True) if datos_class2 else pd.DataFrame()
if not df_class2.empty:
    daily_class2 = df_class2.groupby(['user_id', 'date']).agg({
        'challenge': 'mean',
        'effort': 'mean',
        'grade': 'mean'
    }).reset_index()
    print("Class 2 data processed.")
else:
    print("No valid files were found in Class 2.")
    daily_class2 = pd.DataFrame()
#challengue : "Last week's lab was difficult and challenged me" : (1(agree strongly)-6(Disagree Strongly))
#effort : “I put a great deal of effort into the course last week.”
#grade : “I expect to get the following grade for this course.” (Low values ​​- high expected grade)

# === PART 0.3: Process Behavior Data ===
print("\nProcessing data of Behavior...")
ruta_behavior = "dataset/EMA/response/Behavior"
datos_behavior = []
columnas_behavior = ['anxious', 'calm', 'enthusiastic', 'sympathetic', 'disorganized', 'reserved', 'dependable']

for archivo in sorted(glob.glob(os.path.join(ruta_behavior, "Behavior_u[0-9][0-9].json"))):
    try:
        with open(archivo) as f:
            behavior_data = json.load(f) #Convert JSON content to Python
        df = pd.DataFrame(behavior_data)
        uid = os.path.basename(archivo).split("_")[-1].replace(".json", "")  # "u02"
        user_id = pd.to_numeric(str(uid).lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        if 'resp_time' not in df.columns:
            print(f"File {archivo} ignored: does not contain 'resp_time'")
            continue   
        df['date'] = timestamp_to_local_date(df['resp_time']) #we create the date column
        df['user_id'] = user_id 
        for col in columnas_behavior:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        columnas_a_guardar = ['user_id', 'date'] + [col for col in columnas_behavior if col in df.columns]
        datos_behavior.append(df[columnas_a_guardar]) #a new dataframe is created    
    except Exception as e:
        print(f"Error processing {archivo}: {e}")
df_behavior = pd.concat(datos_behavior, ignore_index=True) if datos_behavior else pd.DataFrame()
if not df_behavior.empty:
    daily_behavior = df_behavior.groupby(['user_id', 'date']).agg({
        col: 'mean' for col in columnas_behavior if col in df_behavior.columns
    }).reset_index() #The average is calculated for each column in the dictionary.
#average of all valid answers that day
    print("Behavior data processed.")
else:
    print("No valid files were found in Behavior.")
    daily_behavior = pd.DataFrame()
#Nan means that they did not respond that day to ema question
#Question : “During the last 15 minutes, I felt…”
#Answer : 1(not at all)-5(extremely)

# === PART 0.4: Process Comment Data ===
print("\nProcess data from Comment...")
ruta_comment = "dataset/EMA/response/Comment"
datos_comment = []
for archivo in sorted(glob.glob(os.path.join(ruta_comment, "Comment_u[0-9][0-9].json"))):
    try:
        with open(archivo) as f:
            comment_data = json.load(f)
        df = pd.DataFrame(comment_data)
        uid = os.path.basename(archivo).split("_")[-1].replace(".json", "")  # "u04"
        user_id = pd.to_numeric(str(uid).lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        if 'resp_time' not in df.columns:
            print(f"File {archivo} not found: does not contain 'resp_time'")
            continue
        df['date'] = timestamp_to_local_date(df['resp_time'])
        df['user_id'] = user_id
        if "comment" not in df.columns:
            continue
        df["comment"] = df["comment"].fillna("").astype(str).str.strip() #The comment remains as clean text, without NaN or spaces.
        df = df[df["comment"] != ""] #remove rows where comment is empty
        if df.empty:
            continue
        # Each valid row counts as 1 comment
        df["comment_count"] = 1
        datos_comment.append(df[["user_id", "date", "comment_count"]])
    except Exception as e:
        print(f"Error processing {archivo}: {e}")
df_comment = pd.concat(datos_comment, ignore_index=True) if datos_comment else pd.DataFrame()
if not df_comment.empty:
    daily_comment = (
        df_comment
        .groupby(["user_id", "date"], as_index=False)
        .agg(comment_count=("comment_count", "sum"))
    )
    print("Comment data processed.")
else:
    print("No valid comments were found.")
    daily_comment = pd.DataFrame(columns=["user_id", "date", "comment_count"])
#This variable is used as an indicator of expressiveness and conscious reflection, complementing quantitative measures of emotional state and daily activity.

# === PART 0.5: Process Dimensions Data ===
print("\nProcessing data from Dimensions...")
ruta_dimensions = "dataset/EMA/response/Dimensions"
datos_dimensions = []
for archivo in sorted(glob.glob(os.path.join(ruta_dimensions, "Dimensions_u[0-9][0-9].json"))):
    try:
        with open(archivo, "r", encoding="utf-8") as f:
            dimensions_data = json.load(f)
        df = pd.DataFrame(dimensions_data)
        uid = os.path.basename(archivo).split("_")[-1].replace(".json", "")
        user_id = pd.to_numeric(str(uid).lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        if "resp_time" not in df.columns:
            continue
        df["date"] = timestamp_to_local_date(df["resp_time"])
        df["user_id"] = user_id
        # Each row (answer) counts as 1
        df["dimensions_count"] = 1
        datos_dimensions.append(df[["user_id", "date", "dimensions_count"]])
    except Exception as e:
        print(f"Error processing {archivo}: {e}")
df_dimensions = pd.concat(datos_dimensions, ignore_index=True) if datos_dimensions else pd.DataFrame()
if not df_dimensions.empty:
    daily_dimensions = (
        df_dimensions.groupby(["user_id", "date"], as_index=False)
        .agg(dimensions_count=("dimensions_count", "sum"))
    )
    print(" Dimensions count processed.")
else:
    print("No valid data for Dimensions.")
    daily_dimensions = pd.DataFrame(columns=["user_id", "date", "dimensions_count"])
#The responses from the Dimensions module were used solely as an indicator of the level of daily interaction with the EMA system, without attributing any direct emotional meaning to them. 
# For example, more active or passive days can be detected (more dimensions, comments...).

# === PART 0.6: Process Dining Halls Data ===
print("\nProcessing datos de Dining Halls...")
ruta_dining_halls = "dataset/EMA/response/Dining Halls"
datos_dining = []
dining_mapping = {
    "1": "Collis", "2": "Hop", "3": "Foco", "4": "KAF",
    "5": "Novack", "6": "East Wheelock Snack Bar",
    "7": "Off-Campus", "8": "Other"
}
#We'll take the most frequent value if there are multiple responses per user and date.
def most_frequent(series):
    mode = series.mode() #returns the most frequently occurring values
    return mode.iloc[0] if not mode.empty else None #if there is one or more ties, you take the first one.
for archivo in glob.glob(os.path.join(ruta_dining_halls, "Dining Halls_u*.json")):
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            dining_data = json.load(f)
        df = pd.DataFrame(dining_data)
        uid = os.path.basename(archivo).split("_")[-1].replace(".json", "")
        user_id = int(uid.lstrip("u"))  
        if 'resp_time' not in df.columns:
            continue
        df['date'] = timestamp_to_local_date(df['resp_time'])
        df['user_id'] = user_id
        for meal in ['breakfast', 'lunch', 'dinner']:
            if meal in df.columns:
                df[meal] = df[meal].astype(str).map(dining_mapping).fillna('Unknown')
        
        datos_dining.append(df[['user_id', 'date', 'breakfast', 'lunch', 'dinner']])
    except Exception as e:
        print(f"Error processing {archivo}: {e}")
df_dining = pd.concat(datos_dining, ignore_index=True) if datos_dining else pd.DataFrame(
    columns=["user_id", "date", "breakfast", "lunch", "dinner"]
) #if there is data, you concatenate all of it.
if not df_dining.empty:
    daily_dining = (
        df_dining.groupby(["user_id", "date"], as_index=False)
        .agg(
            breakfast_place=("breakfast", most_frequent),
            lunch_place=("lunch", most_frequent),
            dinner_place=("dinner", ),
        )
    )
else:
    daily_dining = pd.DataFrame(columns=["user_id", "date", "breakfast_place", "lunch_place", "dinner_place"])

# === PART 0.7: Process Events Data ===
print("\nProcessing data from Events...")
ruta_events = "dataset/EMA/response/Events"
datos_events = []
for archivo in sorted(glob.glob(os.path.join(ruta_events, "Events_u[0-9][0-9].json"))):
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            events_data = json.load(f)  
        uid = os.path.basename(archivo).split("_")[-1].replace(".json", "")
        user_id = pd.to_numeric(str(uid).lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        if not events_data:
            continue     
        df = pd.DataFrame(events_data)
        if 'resp_time' not in df.columns:
            continue   
        df['date'] = timestamp_to_local_date(df['resp_time'])
        df['user_id'] = user_id
        for col in ['positive', 'negative']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") #The number is converted to an integer, and if there is no response, it remains as Nan.
        datos_events.append(df[["user_id", "date", "positive", "negative"]])    
    except Exception as e:
        print(f"Error proccessing {archivo}: {e}")
df_events = pd.concat(datos_events, ignore_index=True) if datos_events else pd.DataFrame(
    columns=["user_id", "date", "positive", "negative"]
)
# Daily aggregation
if not df_events.empty:
    daily_events = (
        df_events.groupby(["user_id", "date"], as_index=False)
        .agg(
            events_responses=("positive", "size"), #How many times did I report events that day?
            positive_mean=("positive", "mean"), #average positive score
            negative_mean=("negative", "mean")
        )
    )
#1(little)-7(extremely high)
else:
    daily_events = pd.DataFrame(
        columns=["user_id", "date", "events_responses", "positive_mean", "negative_mean"]
    )
# === PART 0.8: Process Mood Data ===
print("\nProcessing datos de Mood...")
ruta_mood = "dataset/EMA/response/Mood"
datos_mood = []
for archivo in sorted(glob.glob(os.path.join(ruta_mood, "Mood_u*.json"))):  # acepta cualquier número
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            mood_data = json.load(f)
        # When the file contains a single response, it is represented as a JSON object instead of a list. To ensure consistent data processing, the structure is normalized by converting individual responses into lists before processing.
        if isinstance(mood_data, dict): #If mood data is a dictionary instead of a list of dictionaries
            mood_data = [mood_data]
        df = pd.DataFrame(mood_data)
        uid = os.path.basename(archivo).split("_")[-1].replace(".json", "")  # u02
        user_id = pd.to_numeric(str(uid).lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        if 'resp_time' not in df.columns:
            continue
        df['date'] = timestamp_to_local_date(df['resp_time'])
        df['user_id'] = user_id
        columnas_mood = ['happy', 'sad',]
        for col in columnas_mood:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        columnas_a_guardar = ['user_id', 'date'] + [col for col in columnas_mood if col in df.columns]
        datos_mood.append(df[columnas_a_guardar])
    except Exception as e:
        print(f"Error processing {archivo}: {e}")
df_mood = pd.concat(datos_mood, ignore_index=True) if datos_mood else pd.DataFrame()
if not df_mood.empty:
    daily_mood = df_mood.groupby(['user_id', 'date']).agg({
        col: 'mean' for col in ['happy', 'sad',] if col in df_mood.columns
    }).reset_index() #For each scale, you calculate the average of all the responses for that day.
    print(" Mood data processedo.")
else:
    print("No valid files found in Mood.")
    daily_mood = pd.DataFrame(columns=['user_id', 'date', 'happy', 'sad'])
#happy : “How happy do you feel right now?” ---1(zero happy)-5(very happy)
#sad :  “How sad do you feel right now?”

# === PART 0.9: Process Mood 1  ===
print("\nProcessing data from Mood 1...")
ruta_mood1 = "dataset/EMA/response/Mood 1"
datos_mood1 = []
for archivo in sorted(glob.glob(os.path.join(ruta_mood1, "Mood 1_u[0-9][0-9].json"))):
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            mood1_data = json.load(f)
        if isinstance(mood1_data, dict): #If there is only one answer, you convert it into a list.
            #That's how pandas always work.
            mood1_data = [mood1_data]
        df = pd.DataFrame(mood1_data)
        uid = os.path.basename(archivo).split("_")[-1].replace(".json", "")
        user_id = pd.to_numeric(str(uid).lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        df["user_id"] = user_id
        if 'resp_time' not in df.columns or 'tomorrow' not in df.columns:
            print(f"File {archivo} not found")
            continue
        df['date'] = timestamp_to_local_date(df['resp_time'])
        df['user_id'] = user_id
        df['tomorrow'] = pd.to_numeric(df['tomorrow'], errors='coerce') #convert the scale to numerical
        datos_mood1.append(df[['user_id', 'date', 'tomorrow']])
    except Exception as e:
        print(f"Error processing {archivo}: {e}")
df_mood1 = pd.concat(datos_mood1, ignore_index=True) if datos_mood1 else pd.DataFrame()
if not df_mood1.empty:
    daily_mood1 = df_mood1.groupby(['user_id', 'date'])['tomorrow'].mean().round(1).reset_index() #If they responded multiple times, you calculate the average.
    print("Mood 1 data processed.")
else:
    print("No valid files from Mood 1.")
    daily_mood1 = pd.DataFrame(columns=['user_id', 'date', 'tomorrow'])
#“How do you expect to feel tomorrow?”
#1(I hope I feel really bad.)-5(I hope I feel really good.)

# === PART 0.10: Process Mood 2 ===
print("\n💭 Procesando datos de Mood 2...")
ruta_mood2 = "dataset/EMA/response/Mood 2"
datos_mood2 = []
for archivo in sorted(glob.glob(os.path.join(ruta_mood2, "Mood 2_u[0-9][0-9].json"))):
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            mood2_data = json.load(f)
        if isinstance(mood2_data, dict):
            mood2_data = [mood2_data]
        df = pd.DataFrame(mood2_data)
        uid = os.path.basename(archivo).split("_")[-1].replace(".json", "")
        user_id = pd.to_numeric(str(uid).lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        if 'resp_time' not in df.columns or 'how' not in df.columns:
            continue
        df['date'] = timestamp_to_local_date(df['resp_time'])
        df['user_id'] = user_id
        df['how'] = pd.to_numeric(df['how'], errors='coerce')
        datos_mood2.append(df[['user_id', 'date', 'how']])
    except Exception as e:
        print(f" Error processing {archivo}: {e}")
df_mood2 = pd.concat(datos_mood2, ignore_index=True) if datos_mood2 else pd.DataFrame()
if not df_mood2.empty:
    daily_mood2 = df_mood2.groupby(['user_id', 'date'])['how'].mean().round(1).reset_index()
    print("Mood 2 data processed.")
else:
    print("No valid files for Mood 2.")
    daily_mood2 = pd.DataFrame(columns=['user_id', 'date', 'how'])
#“How do you feel right now?” 1(very bad)-5(very good)

# === PART 0.11: Process Sleep Data ===
print("\nProcessing datos of Sleep...")
ruta_sleep = "dataset/EMA/response/Sleep"
datos_sleep = []
for archivo in sorted(glob.glob(os.path.join(ruta_sleep, "Sleep_u[0-9][0-9].json"))):
    try:
        df = pd.read_json(archivo)
        uid = os.path.basename(archivo).split("_")[-1].replace(".json", "")
        user_id = pd.to_numeric(str(uid).lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        if 'resp_time' not in df.columns:
            continue
        df['date'] = pd.to_datetime(df['resp_time'], utc=True, errors='coerce').dt.tz_convert(TZ).dt.date
        #Standardized the format
# In the Sleep dataset, resp_time is already a datetime object, so the previous timestamp-based function cannot be used here.
        df = df.dropna(subset=['date'])
        df['user_id'] = user_id
        for col in ['hour', 'rate']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['hour', 'rate'], how='all') #If there is at least one piece of data, the row remains. But if there is no data, it is deleted.
        if df.empty:
            continue
        resumen = df.groupby(['user_id', 'date']).agg(
            sleep_hours=('hour', 'mean'), #average hours of sleep that day
            sleep_quality=('rate', 'mean'), #average sleep quality
        ).reset_index()
        resumen["sleep_hours"] = resumen["sleep_hours"].round(2)
        resumen["sleep_quality"] = resumen["sleep_quality"].round(2)
        datos_sleep.append(resumen)
    except Exception as e:
        print(f"Error processing {archivo}: {e}")
daily_sleep = pd.concat(datos_sleep, ignore_index=True) if datos_sleep else pd.DataFrame()
if not daily_sleep.empty:
    print("Sleep data processed.")
else:
    print("No valid files in Sleep.")

# === PART 0.12: Process Social Data ===
print("\nProcessing data from Social...")
ruta_social = "dataset/EMA/response/Social"
datos_social = []
for archivo in sorted(glob.glob(os.path.join(ruta_social, "Social_u[0-9][0-9].json"))):
    try:
        with open(archivo, "r", encoding="utf-8") as f:
            social_data = json.load(f)
        if isinstance(social_data, dict): #Sometimes a JSON can come as a single dictionary instead of a list.
            social_data = [social_data]
        df = pd.DataFrame(social_data)
        df.columns = df.columns.str.strip().str.lower()
        uid = os.path.basename(archivo).split("_")[-1].replace(".json", "")
        user_id = pd.to_numeric(str(uid).lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        if "resp_time" not in df.columns or "number" not in df.columns:
            continue
        df["date"] = timestamp_to_local_date(df["resp_time"])
        df = df.dropna(subset=["date"])
        df["user_id"] = user_id
        df["social_people"] = pd.to_numeric(df["number"], errors="coerce")
        df = df.dropna(subset=["social_people"])
        if df.empty:
            continue
        datos_social.append(df[["user_id", "date", "social_people"]])
    except Exception as e:
        print(f"Error processing {archivo}: {e}")

df_social = pd.concat(datos_social, ignore_index=True) if datos_social else pd.DataFrame()
if not df_social.empty:
    daily_social = (
        df_social.groupby(["user_id", "date"], as_index=False)
        .agg(social_people=("social_people", "mean"))
    )
    print("Social data processed.")
else:
    print("No valid files were found in Social.")
    daily_social = pd.DataFrame(columns=["user_id", "date", "social_people"])
#social_people represents the average level of social interaction that day.
#1(0-4 people)
#2(5-9 people)...

# === PART 0.13: Process Stress Data ===
print("\nProcessing data from Stress...")
ruta_stress = "dataset/EMA/response/Stress"
datos_stress = []
#Stress measures the level of stress perceived by the user
#1 : vrey little stress , 5 : lots of stress
for archivo in sorted(glob.glob(os.path.join(ruta_stress, "Stress_u[0-9][0-9].json"))):
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            stress_data = json.load(f)
        if isinstance(stress_data, dict):
            stress_data = [stress_data]
        df = pd.DataFrame(stress_data)
        df.columns = df.columns.str.strip().str.lower()
        uid = os.path.basename(archivo).split("_")[-1].replace(".json", "")
        user_id = pd.to_numeric(str(uid).lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)       
        if "resp_time" not in df.columns or "level" not in df.columns:
            continue
        df["date"] = timestamp_to_local_date(df["resp_time"])
        df = df.dropna(subset=["date"])
        df["user_id"] = user_id
        df["stress_level"] = pd.to_numeric(df["level"], errors="coerce")
        df = df.dropna(subset=["stress_level"])
        if df.empty:
            continue
        datos_stress.append(df[["user_id", "date", "stress_level"]])
    except Exception as e:
        print(f"Error processing {archivo}: {e}")
df_stress = pd.concat(datos_stress, ignore_index=True) if datos_stress else pd.DataFrame()
if not df_stress.empty:
    daily_stress = (
        df_stress.groupby(["user_id", "date"], as_index=False)
        .agg(stress_level=("stress_level", "mean"))
    )
    daily_stress["stress_level"] = daily_stress["stress_level"].round(2)
    print("Stress data processed.")
else:
    print("No valid files were found in Stress.")
    daily_stress = pd.DataFrame(columns=["user_id", "date", "stress_level"])

# === PART 0.14: Process Study Spaces Data ===
print("\nProcessing data from Study Spaces...")
ruta_study = "dataset/EMA/response/Study Spaces"
datos_study = []
for archivo in sorted(glob.glob(os.path.join(ruta_study, "Study Spaces_u[0-9][0-9].json"))):
    try:
        with open(archivo, "r", encoding="utf-8") as f:
            study_data = json.load(f)
        if not study_data:
            continue
        if isinstance(study_data, dict):
            study_data = [study_data]
        df = pd.DataFrame(study_data)
        df.columns = df.columns.str.strip().str.lower()
        uid = os.path.basename(archivo).split("_")[-1].replace(".json", "")
        user_id = pd.to_numeric(str(uid).lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        if "resp_time" not in df.columns or "productivity" not in df.columns:
            continue
        df["date"] = timestamp_to_local_date(df["resp_time"])
        df = df.dropna(subset=["date"])
        df["user_id"] = user_id
        if "place" in df.columns:
            df["study_place"] = (
                df["place"].astype(str).str.strip().str.lower().replace({"": "unknown", "nan": "unknown"})
            )
        else:
            df["study_place"] = "unknown"
        df["study_productivity"] = pd.to_numeric(df["productivity"], errors="coerce")
        df = df.dropna(subset=["study_productivity"])
        if df.empty:
            continue
        datos_study.append(df[["user_id", "date", "study_place", "study_productivity"]])
    except Exception as e:
        print(f"Error processing {archivo}: {e}")
df_study = pd.concat(datos_study, ignore_index=True) if datos_study else pd.DataFrame()
if not df_study.empty:
    daily_study = (
        df_study
        .groupby(["user_id", "date"], as_index=False)
        .mean(numeric_only=True)
    )
    #Representative study location of the day (first record)
    #The first place you study each day usually coincides with your main study location or the dominant context (many consecutive hours)
    # This is a simple approximation, not an absolute truth.
    daily_study["study_place"] = (
        df_study
        .groupby(["user_id", "date"])["study_place"]
        .first()
        .values
    )
    print("Study Spaces data processed.")
else:
    print("No valid files were found in Study Spaces.")
    daily_study = pd.DataFrame(columns=["user_id", "date", "study_place", "study_productivity"])
##end of EMA

##SENSING
# === PART 1: Process Activity data ===
#0 -> Stationary , 1->Walking , 2->Running , 3->Unknown
#prop_stationary → % of the day spent stationary
#prop_active → % of the day spent active (walking + running)
# === SENSING: Activity Inference ===
print("\nProcessing sensing activity inference...")
ruta_activity = "dataset/sensing/activity/"
datos_activity = []
for archivo in sorted(glob.glob(os.path.join(ruta_activity, "activity_u*.csv"))):
    try:
        uid = os.path.basename(archivo).split("_")[-1].replace(".csv", "")
        user_id = pd.to_numeric(uid.lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        df = pd.read_csv(archivo)
        df.columns = df.columns.str.strip().str.lower()
        if "timestamp" not in df.columns or "activity inference" not in df.columns:
            continue
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["activity"] = pd.to_numeric(df["activity inference"], errors="coerce")
        df = df.dropna(subset=["timestamp", "activity"])
        df["date"] = timestamp_to_local_date(df["timestamp"])
        df["user_id"] = user_id
        datos_activity.append(df[["user_id", "date", "activity"]])

    except Exception as e:
        print(f"Error processing {archivo}: {e}")
if datos_activity:
    df_activity = pd.concat(datos_activity, ignore_index=True)
    #create two boolean variables
    df_activity["is_stationary"] = df_activity["activity"] == 0 #will be True when activity == 0.
    df_activity["is_active"] = df_activity["activity"].isin([1, 2]) #it will be True when activity is 1 or 2.
    # Group by user and date
    daily_act_inference = (
        df_activity
        .groupby(["user_id", "date"])
        .agg(
            prop_stationary=("is_stationary", "mean"),
            prop_active=("is_active", "mean")
        )
        .reset_index())
    daily_act_inference["prop_stationary"] = daily_act_inference["prop_stationary"].round(2)
    daily_act_inference["prop_active"] = daily_act_inference["prop_active"].round(2)
    print("Activity inference processed successfully.")
else:
    daily_act_inference = pd.DataFrame(
        columns=["user_id", "date", "prop_stationary", "prop_active"]
    )
    print("No valid activity inference data found.")

#=== SENSING: Audio ===
#0 -> Silence , 1->Voice , 2-> Noise , 3->Unknown
#prop_voice → proportion of the day with human voice
#prop_noise → proportion of the day with noise

# === SENSING: Audio Inference ===
print("\nProcessing sensing audio inference...")
ruta_audio = "dataset/sensing/audio/"
datos_audio = []
for archivo in sorted(glob.glob(os.path.join(ruta_audio, "audio_u*.csv"))):
    try:
        uid = os.path.basename(archivo).split("_")[-1].replace(".csv", "")
        user_id = pd.to_numeric(uid.lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        df = pd.read_csv(archivo)
        df.columns = df.columns.str.strip().str.lower()
        if "timestamp" not in df.columns or "audio inference" not in df.columns:
            continue
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["audio"] = pd.to_numeric(df["audio inference"], errors="coerce")
        df = df.dropna(subset=["timestamp", "audio"])
        df["date"] = timestamp_to_local_date(df["timestamp"])
        df["user_id"] = user_id
        datos_audio.append(df[["user_id", "date", "audio"]])
    except Exception as e:
        print(f"Error processing {archivo}: {e}")
if datos_audio:
    df_audio = pd.concat(datos_audio, ignore_index=True)
    df_audio["is_voice"] = df_audio["audio"] == 1
    df_audio["is_noise"] = df_audio["audio"] == 2
    daily_audio = (
        df_audio
        .groupby(["user_id", "date"])
        .agg(
            prop_voice=("is_voice", "mean"),
            prop_noise=("is_noise", "mean"),
        ).reset_index())
    daily_audio["prop_voice"] = daily_audio["prop_voice"].round(2)
    daily_audio["prop_noise"] = daily_audio["prop_noise"].round(2)
    print("Audio inference processed successfully.")
else:
    daily_audio = pd.DataFrame(columns=["user_id", "date", "prop_voice", "prop_noise"])
    print("No valid audio inference data found.")

# === SENSING: Bluetooth ===
#bt_unique_devices:Number of different Bluetooth devices detected that day. Count of different MAC addresses.(High value: many people/devices around)
#bt_scans : Number of Bluetooth scans performed that day.Number of distinct timestamps (time) in the day. Measures how many times information was collected. It is a column of data quality/coverage.
#I'm concerned about data quality. For example, it helps you interpret 'bt_unique_devices':
#'bt_unique_devices = 2' with 'bt_scans = 2'→ little information, unreliable
#'bt_unique_devices = 2' with 'bt_scans = 40'→ solid information, a really quiet day
print("\nProcessing sensing bluetooth...")
ruta_bt = "dataset/sensing/bluetooth/"
bt_data = []
for archivo in sorted(glob.glob(os.path.join(ruta_bt, "bt_u*.csv"))):
    try:
        uid = os.path.basename(archivo).split("_")[-1].replace(".csv", "")
        user_id = pd.to_numeric(uid.lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        df = pd.read_csv(archivo)
        df.columns = df.columns.str.strip().str.lower()
        if "time" not in df.columns or "mac" not in df.columns:
            continue
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.dropna(subset=["time", "mac"])
        df["date"] = timestamp_to_local_date(df["time"])
        df["user_id"] = user_id
        # 1) Unique devices per day (diferent MAC)
        daily_unique = (
            df.groupby(["user_id", "date"])["mac"]
            .nunique()
            .reset_index(name="bt_unique_devices")
        )
        # 2) Number of scans per day (unique timestamps)
        daily_scans = (
            df.groupby(["user_id", "date"])["time"]
            .nunique()
            .reset_index(name="bt_scans")
        )
        resumen = pd.merge(daily_unique, daily_scans, on=["user_id", "date"], how="outer") # join tables
        bt_data.append(resumen)
    except Exception as e:
        print(f"Error processing {archivo}: {e}")
# Concatenate everthing
if bt_data:
    daily_bt = pd.concat(bt_data, ignore_index=True)
    print("Bluetooth processed successfully.")
else:
    daily_bt = pd.DataFrame(columns=["user_id", "date", "bt_unique_devices", "bt_scans"])
    print("No valid bluetooth files were found.")
#We will put 0 in the nans, since they are counts

# === SENSING: DARK ===
#“Dark Screen data records intervals in which the device remains with the screen off (start–end), allowing us to estimate the total daily duration of mobile disconnection as an objective indicator related to rest and usage habits.”
#dark_minutes:Total time (in minutes) that the user was in a dark environment that day
# === SENSING: Dark Screen ===
print("\nProcessing sensing dark screen...")
ruta_dark = "dataset/sensing/dark/"
dark_data = []
for archivo in sorted(glob.glob(os.path.join(ruta_dark, "dark_u*.csv"))):
    try:
        uid = os.path.basename(archivo).split("_")[-1].replace(".csv", "")
        user_id = pd.to_numeric(uid.lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        df = pd.read_csv(archivo)
        df.columns = df.columns.str.strip().str.lower()
        if "start" not in df.columns or "end" not in df.columns:
            continue
        df["start"] = pd.to_numeric(df["start"], errors="coerce")
        df["end"] = pd.to_numeric(df["end"], errors="coerce")
        df = df.dropna(subset=["start", "end"])
        # Duration in seconds
        df["duration_sec"] = df["end"] - df["start"]
        df = df[df["duration_sec"] >= 0]
        df["date"] = timestamp_to_local_date(df["start"])
        df["user_id"] = user_id
        # Just minutes of darkness
        resumen = (
            df.groupby(["user_id", "date"])
            .agg(dark_minutes=("duration_sec", "sum"))
            .reset_index()
        )
        #Convert to minutes and round up
        resumen["dark_minutes"] = (resumen["dark_minutes"] / 60).round(2)
        dark_data.append(resumen)
    except Exception as e:
        print(f"Error processing {archivo}: {e}")
if dark_data:
    daily_dark = pd.concat(dark_data, ignore_index=True)
    print("Dark screen processed successfully.")
else:
    daily_dark = pd.DataFrame(columns=["user_id", "date", "dark_minutes"])
    print("No valid dark screen files were found.")

# === SENSING: GPS ===
#“Although the dataset includes GPS location information, this sensor was not incorporated into the final analysis due to its high processing complexity and redundancy with other contextual signals already considered, such as physical activity, social proximity and device use, prioritizing a more interpretable and robust set of variables for the objective of the study.”
# === SENSING: Phone Charge ===
#“The variable charge_minutes represents the total daily time during which the device was connected to the power supply, used as an indirect indicator of daily usage and organization routines.”
print("\nProcessing sensing phone charge...")
ruta_phonecharge = "dataset/sensing/phonecharge/"
phonecharge_data = []
for archivo in sorted(glob.glob(os.path.join(ruta_phonecharge, "phonecharge_u*.csv"))):
    try:
        uid = os.path.basename(archivo).split("_")[-1].replace(".csv", "")
        user_id = pd.to_numeric(uid.lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        df = pd.read_csv(archivo)
        df.columns = df.columns.str.strip().str.lower()
        if "start" not in df.columns or "end" not in df.columns:
            continue
        df["start"] = pd.to_numeric(df["start"], errors="coerce")
        df["end"] = pd.to_numeric(df["end"], errors="coerce")
        df = df.dropna(subset=["start", "end"])
        # duration in minutes
        df["duration_min"] = (df["end"] - df["start"]) / 60
        df = df[df["duration_min"] >= 0]
        df["date"] = timestamp_to_local_date(df["start"])
        df["user_id"] = user_id
        resumen = (
            df.groupby(["user_id", "date"])
            .agg(charge_minutes=("duration_min", "sum"))
            .reset_index() )
        resumen["charge_minutes"] = resumen["charge_minutes"].round(2)
        phonecharge_data.append(resumen)
    except Exception as e:
        print(f"Error processing {archivo}: {e}")
if phonecharge_data:
    daily_phonecharge = pd.concat(phonecharge_data, ignore_index=True)
    print("Phonecharge processed successfully.")
else:
    daily_phonecharge = pd.DataFrame(columns=["user_id", "date", "charge_minutes"])
    print("No valid phonecharge files found.")
# === SENSING: Phone Lock ===
#“The variable locked_minutes represents the total daily time the device remained locked, used as an objective indicator of phone disconnection and usage habits.”
print("\nProcessing sensing phonelock...")
ruta_phonelock = "dataset/sensing/phonelock/"
phonelock_data = []
for archivo in sorted(glob.glob(os.path.join(ruta_phonelock, "phonelock_u*.csv"))):
    try:
        uid = os.path.basename(archivo).split("_")[-1].replace(".csv", "")
        user_id = pd.to_numeric(uid.lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        df = pd.read_csv(archivo)
        df.columns = df.columns.str.strip().str.lower()
        if "start" not in df.columns or "end" not in df.columns:
            continue
        df["start"] = pd.to_numeric(df["start"], errors="coerce")
        df["end"] = pd.to_numeric(df["end"], errors="coerce")
        df = df.dropna(subset=["start", "end"])
        df["duration_min"] = (df["end"] - df["start"]) / 60
        df = df[df["duration_min"] >= 0]
        df["date"] = timestamp_to_local_date(df["start"])
        df["user_id"] = user_id
        resumen = (
            df.groupby(["user_id", "date"])
            .agg(locked_minutes=("duration_min", "sum"))
            .reset_index()
        )
        resumen["locked_minutes"] = resumen["locked_minutes"].round(2)
        phonelock_data.append(resumen)
    except Exception as e:
        print(f"Error processing {archivo}: {e}")
if phonelock_data:
    daily_phonelock = pd.concat(phonelock_data, ignore_index=True)
    print("Phonelock processed successfully.")
else:
    daily_phonelock = pd.DataFrame(columns=["user_id", "date", "locked_minutes"])
    print("No valid phonelock files found.")
#If there is no data, we assume 0 minutes blocked
# === Sensing: WiFi =
#The unique_wifi variable represents the number of unique WiFi access point identifiers (BSSIDs) detected by the device throughout the day, and is used as an indirect measure of mobility and exposure to different environments.
print("Processing data from WiFi...")
dfs_wifi = []
for file in sorted(glob.glob("dataset/sensing/wifi/*.csv")):
    try:
        user_id = int(os.path.basename(file).split("_")[1].replace(".csv", "").replace("u", ""))
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower()
        if "time" not in df.columns or "bssid" not in df.columns:
            continue
        df["timestamp"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["date"] = timestamp_to_local_date(df["timestamp"])
        df["user_id"] = user_id
        dfs_wifi.append(df[["user_id", "date", "bssid"]])
    except Exception as e:
        print(f"Error processing  {file}: {e}")
if dfs_wifi:
    wifi_all = pd.concat(dfs_wifi, ignore_index=True)
    daily_wifi = (
        wifi_all
        .groupby(["user_id", "date"])
        .agg(unique_wifi=("bssid", "nunique"))
        .reset_index())
    print("WiFi processed correctly.")
else:
    daily_wifi = pd.DataFrame(columns=["user_id", "date", "unique_wifi"])
    print("No valid files found in WiFi.")

#SMS
#The variable num_sms collects the total number of SMS message events recorded per day, and is used as a passive indicator of digital social interaction, avoiding the use of sensitive content or information.
# num_sms is a daily count
print("\nProcessing SMS data ...")
ruta_sms = "dataset/sms/"
sms_data = []
for archivo in sorted(glob.glob(os.path.join(ruta_sms, "sms_u*.csv"))):
    try:
        uid = os.path.basename(archivo).split("_")[-1].replace(".csv", "")
        user_id = pd.to_numeric(uid.lstrip("u"), errors="coerce")
        if pd.isna(user_id):
            continue
        user_id = int(user_id)
        df = pd.read_csv(archivo)
        df.columns = df.columns.str.strip().str.lower()
        if "timestamp" not in df.columns:
            continue
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["date"] = timestamp_to_local_date(df["timestamp"])
        df["user_id"] = user_id
        resumen = (
            df.groupby(["user_id", "date"])
            .size()
            .reset_index(name="num_sms") )
        sms_data.append(resumen)
    except Exception as e:
        print(f"Error processing {archivo}: {e}")
if sms_data:
    sms_daily = pd.concat(sms_data, ignore_index=True)
    print("SMS processed successfully.")
else:
    sms_daily = pd.DataFrame(columns=["user_id", "date", "num_sms"])
    print("No valid SMS files found.")

##SURVEY
#BigFive
print("Processing Big Five (BFI-44)...")
try:
    file_path = "dataset/survey/BigFive.csv" 
    df = pd.read_csv(file_path)
#It forces the first two columns to be named uid and type. It gives the 44 question columns clean names:item_1, item_2, ..., item_44
    df = df.rename(columns={df.columns[0]: "uid", df.columns[1]: "type"})
    df_items = df.columns[2:] 
    df = df.rename(columns={old: f"item_{i+1}" for i, old in enumerate(df_items)})
#We convert text responses to numbers
    scale = {
        "Disagree Strongly": 1,
        "Disagree a little": 2,
        "Neither agree nor disagree": 3,
        "Agree a little": 4,
        "Agree strongly": 5
    }
    for i in range(1, 45):
        col = f"item_{i}"
        df[col] = df[col].map(scale) #Replace each text with its number
    # We reverse the answer to the question. If the person answered x = 5 → it becomes 1 If they answered x = 2 → it becomes 4
    reverse_items = [6, 21, 31, 2, 12, 27, 37, 8, 18, 23, 43, 9, 24, 34]
    #A high number should ALWAYS mean "a lot of the trait". That's why if they ask "I am a quiet person", a 1 would mean very quiet, and a 5 would mean not very quiet. That's why it needs to be changed.
    for i in reverse_items:
        col = f"item_{i}"
        df[col] = df[col].apply(lambda x: 6 - x if pd.notna(x) else x)
    #What questions form each trait according to the official test
    traits = {
        "extraversion":      [1, 6, 11, 16, 21, 26, 31, 36],
        "agreeableness":     [2, 7, 12, 17, 22, 27, 32, 37, 42],
        "conscientiousness": [3, 8, 13, 18, 23, 28, 33, 38, 43],
        "neuroticism":       [4, 9, 14, 19, 24, 29, 34, 39],
        "openness":          [5, 10, 15, 20, 25, 30, 35, 40, 41, 44],
    }
    for trait, items in traits.items(): #for each trait
        cols = [f"item_{i}" for i in items]
        df[trait] = df[cols].mean(axis=1, skipna=True).round(2)
    #Select only those columns from each trait, and for each row calculate an average of their responses.
    df["user_id"] = pd.to_numeric(
        df["uid"].astype(str).str.lstrip("u"),
        errors="coerce")
    df = df.dropna(subset=["user_id"])
    df["user_id"] = df["user_id"].astype(int) #Previously, I extracted the user from the file name; now I extract it from a column.
    df["type"] = df["type"].astype(str).str.strip().str.lower() #The type attribute only indicates whether the test is "before" or "after," and that line simply cleans it up to prevent problems.
    df_bigfive = df[["user_id", "type"] + list(traits.keys())].copy() #Combine the user_id and the type with the list of traits
    df_bigfive_pre = (
        df_bigfive[df_bigfive["type"] == "pre"]
        .drop(columns=["type"])
        .reset_index(drop=True)) # we keep only PRE , because we want to use the baseline personalities before the study.
    
    print("Big Five processed.")
    print(df_bigfive.head(5))
except Exception as e:
    print(f"Error processing BigFive.csv: {e}")
    df_bigfive = pd.DataFrame()

#FLOURISHING SCALE
#The Flourishing Scale is a psychological questionnaire composed of 8 items,
#each rated on a 7-point Likert scale. The overall flourishing score is computed
#by summing the scores of the 8 items, resulting in a total score ranging from 8 to 56.
# === PARTE XX: Flourishing Scale (PRE only) ===
print("\nProcessing Flourishing Scale...")
try:
    df_flourishing_raw = pd.read_csv("dataset/survey/FlourishingScale.csv")
    # Clean text and columns
    df_flourishing_raw.columns = df_flourishing_raw.columns.str.strip()
    df_flourishing_raw["uid"] = df_flourishing_raw["uid"].astype(str).str.strip()
    df_flourishing_raw["type"] = df_flourishing_raw["type"].astype(str).str.strip().str.lower()
    # Filter only PRE
    df_pre = df_flourishing_raw[df_flourishing_raw["type"] == "pre"].copy()
    # Columns of items (all but uid and type)
    item_cols = [c for c in df_pre.columns if c not in ["uid", "type"]]
    # numeric
    df_pre[item_cols] = df_pre[item_cols].apply(pd.to_numeric, errors="coerce")
    # Total score(sum) 
    df_pre["flourishing_score"] = df_pre[item_cols].sum(axis=1, skipna=True).round(2)
    df_pre["user_id"] = pd.to_numeric(df_pre["uid"].str.lstrip("u"), errors="coerce")
    df_pre = df_pre.dropna(subset=["user_id"])
    df_pre["user_id"] = df_pre["user_id"].astype(int)
    df_flourishing = df_pre[["user_id", "flourishing_score"]].dropna()
    print("Flourishing Scale processed (PRE). EXAMPLE:")
    print(df_flourishing.head(5))
except Exception as e:
    print(f" Error processing FlourishingScale.csv: {e}")
    df_flourishing = pd.DataFrame(columns=["user_id", "flourishing_score"])

##LONELINESS SCALE
#The UCLA Loneliness Scale measures perceived loneliness: it's not "how many friends you have," but how you feel about social connection.
#A high number always means "more loneliness" in all questions.
#TOTAL SCORE IS THE SUM OF THE 20 ITEMS
print("\Processing Loneliness Scale...")
try:
    df_loneliness_raw = pd.read_csv("dataset/survey/LonelinessScale.csv")
    df_loneliness_raw.columns = df_loneliness_raw.columns.str.strip()
    df_loneliness_raw["uid"]  = df_loneliness_raw["uid"].astype(str).str.strip()
    df_loneliness_raw["type"] = df_loneliness_raw["type"].astype(str).str.strip().str.lower()
    # Filter PRE
    df_pre = df_loneliness_raw[df_loneliness_raw["type"] == "pre"].copy()
    df_pre["user_id"] = pd.to_numeric(df_pre["uid"].str.lstrip("u"), errors="coerce")
    df_pre = df_pre.dropna(subset=["user_id"])
    df_pre["user_id"] = df_pre["user_id"].astype(int)
    # Item columns
    item_cols = [c for c in df_pre.columns if c not in ["uid", "type", "user_id"]]
    # Map responses to numbers (1–4)
    mapping = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4}
    for col in item_cols:
        df_pre[col] = df_pre[col].astype(str).str.strip().map(mapping)
# Reverse-scored items (UCLA Loneliness: "positive" items)
# In StudentLife, item 4 is "I do not feel alone" (positive), so it is also reverse-scored.
    reverse_items = [1, 4, 5, 6, 9, 10, 15, 16, 19, 20] #This list indicates which questions are phrased positively and should be reversed, as indicated by the official scoring.
    for i in reverse_items:
        col_name = f"{i}."  # para localizar por prefijo "1.", "2.", etc. en el nombre real
        # encontrar la columna que empieza por "i."
        matches = [c for c in item_cols if c.strip().startswith(col_name)] #Look for the column that starts like this
        if matches:
            c = matches[0]
            df_pre[c] = df_pre[c].apply(lambda x: 5 - x if pd.notna(x) else x)  # 1<->4, 2<->3
            #If there's a number, we reverse it; if it's empty, we leave it as is.
    # sum by row and calculate the total score
    df_pre["loneliness_score"] = df_pre[item_cols].sum(axis=1, skipna=True).round(2)
    df_loneliness = df_pre[["user_id", "loneliness_score"]].dropna()
    print("Loneliness Scale processed (PRE). Example:")
    print(df_loneliness.head(5))
except Exception as e:
    print(f"Error processing Loneliness Scale: {e}")
    df_loneliness = pd.DataFrame(columns=["user_id", "loneliness_score"])
#PANAS
#The original version contains 20 items:10 positive and 10 negative.Each item is answered on a scale of 1 to 5 and two scores are calculated:
#PA = sum of the positive items , NA = sum of the negative items
#Important: The StudentLife panas.csv file does NOT contain all 20 items; it contains 18 (9 positive and 9 negative).
# === PARTE XX: PANAS (PRE only) ===
print("\nProcessing PANAS...")
try:
    df_panas_raw = pd.read_csv("dataset/survey/panas.csv")
    df_panas_raw.columns = df_panas_raw.columns.str.strip().str.lower()
    df_panas_raw["uid"] = df_panas_raw["uid"].astype(str).str.strip()
    df_panas_raw["type"] = df_panas_raw["type"].astype(str).str.strip().str.lower()
    # Filter PRE
    df_pre = df_panas_raw[df_panas_raw["type"] == "pre"].copy()
    df_pre["user_id"] = pd.to_numeric(df_pre["uid"].str.lstrip("u"), errors="coerce")
    df_pre = df_pre.dropna(subset=["user_id"])
    df_pre["user_id"] = df_pre["user_id"].astype(int)
    positive_items = [
        "interested", "strong", "enthusiastic", "proud", "alert",
        "inspired", "determined", "attentive", "active"
    ]
    negative_items = [
        "distressed", "upset", "guilty", "scared", "hostile",
        "irritable", "nervous", "jittery", "afraid"
    ]
# Let's only keep those that actually exist (for security)
    pos_cols = [c for c in positive_items if c in df_pre.columns]
    neg_cols = [c for c in negative_items if c in df_pre.columns]
# Convert to numeric
    df_pre[pos_cols + neg_cols] = df_pre[pos_cols + neg_cols].apply(pd.to_numeric, errors="coerce")
# Scores (sums) with 2 decimal places
    df_pre["panas_positive"] = df_pre[pos_cols].sum(axis=1, skipna=True).round(2)
    df_pre["panas_negative"] = df_pre[neg_cols].sum(axis=1, skipna=True).round(2)
    df_panas_final = df_pre[["user_id", "panas_positive", "panas_negative"]].dropna()
    print("PANAS processed correctly. Example:")
    print(df_panas_final.head(5))
except Exception as e:
    print(f"Error processing PANAS: {e}")
    df_panas_final = pd.DataFrame(columns=["user_id", "panas_positive", "panas_negative"])
#This dataset contains 18 adjectives (9 positive and 9 negative), so the resulting scores are calculated by adding the available items.

#PERCEIVED STRESS SCALE
#These are 10 questions about perceived stress “in the last month”. Never = 0, Almost never = 1, Sometimes = 2, Fairly often = 3, Very often = 4. There are 4 positive questions (4, 5, 7, 8) that are scored in reverse so that:
#a high number always means more stress
print("\nProcessing Perceived Stress Scale (PSS-10)...")
try:
    df_stress_raw = pd.read_csv("dataset/survey/PerceivedStressScale.csv")
    df_stress_raw.columns = df_stress_raw.columns.str.strip()
    df_stress_raw["uid"]  = df_stress_raw["uid"].astype(str).str.strip()
    df_stress_raw["type"] = df_stress_raw["type"].astype(str).str.strip().str.lower()
    df_pre = df_stress_raw[df_stress_raw["type"] == "pre"].copy()
    df_pre["user_id"] = pd.to_numeric(df_pre["uid"].str.lstrip("u"), errors="coerce")
    df_pre = df_pre.dropna(subset=["user_id"])
    df_pre["user_id"] = df_pre["user_id"].astype(int)
    item_cols = [c for c in df_pre.columns if c not in ["uid", "type", "user_id"]]
    # Official PSS mapping: 0–4 (Never=0 ... Very often=4)
    mapping = {
        "never": 0,
        "almost never": 1,
        "sometime": 2,    
        "sometimes": 2,    
        "fairly often": 3,
        "very often": 4
    }
    for col in item_cols:
        df_pre[col] = (
            df_pre[col].astype(str).str.strip().str.lower().map(mapping)
        )
#Reverse scoring
    reverse_items = [4, 5, 7, 8]
    for i in reverse_items:
        # buscar la columna que empieza por "i."
        prefix = f"{i}."
        matches = [c for c in item_cols if c.strip().startswith(prefix)]
        if matches:
            c = matches[0]
            df_pre[c] = df_pre[c].apply(lambda x: 4 - x if pd.notna(x) else x)  #scale 0-4
    # sum of the 10 items
    df_pre["stress_score"] = df_pre[item_cols].sum(axis=1, skipna=True).round(2)
    df_stress_final = df_pre[["user_id", "stress_score"]].dropna()
    print("PSS-10 processed (PRE). Example:")
    print(df_stress_final.head(10))
except Exception as e:
    print(f"Error processing PerceivedStressScale.csv: {e}")
    df_stress_final = pd.DataFrame(columns=["user_id", "stress_score"])

##PHQ-9
#The more symptoms you have and the more frequent they are,the greater the likelihood of depression.” 0–4: Minimal or no depression, 20–27: Severe
print("\nProcessing PHQ-9...")
try:
    df_phq_raw = pd.read_csv("dataset/survey/PHQ-9.csv")
    df_phq_raw.columns = df_phq_raw.columns.str.strip()
    df_phq_raw["uid"] = df_phq_raw["uid"].astype(str).str.strip()
    df_phq_raw["type"] = df_phq_raw["type"].astype(str).str.strip().str.lower()
    df_pre = df_phq_raw[df_phq_raw["type"] == "pre"].copy()
    df_pre["user_id"] = pd.to_numeric(df_pre["uid"].str.lstrip("u"), errors="coerce")
    df_pre = df_pre.dropna(subset=["user_id"])
    df_pre["user_id"] = df_pre["user_id"].astype(int)
    item_cols = [c for c in df_pre.columns if c not in ["uid", "type", "user_id"]]
    # Mapping oficial PHQ-9(0–3)
    mapping = {
        "not at all": 0,
        "several days": 1,
        "more than half the days": 2,
        "nearly every day": 3
    }
    for col in item_cols:
        df_pre[col] = df_pre[col].astype(str).str.strip().str.lower().map(mapping)
    # Total score: sum of the 9 items (range 0–27)
    df_pre["phq_score"] = df_pre[item_cols].sum(axis=1, skipna=True).round(2)
    df_phq_final = df_pre[["user_id", "phq_score"]].dropna()
    print(" PHQ-9 correctly processed (PRE). Example:")
    print(df_phq_final.head(10))
except Exception as e:
    print(f"Error processing PHQ-9: {e}")
    df_phq_final = pd.DataFrame(columns=["user_id", "phq_score"])

#PSQI
#The PSQI questionnaire was explored, but was discarded due to inconsistencies in textual coding and lack of standardization.

#vr_12
#The VR-12 questionnaire measures general aspects of physical and mental health, while this final degree project focuses on daily emotional well-being.
#Since the project's objective is not to assess health, but rather to analyze emotions and daily routines, it was decided not to use this questionnaire.


###### Final part
print("\n Joining datasets...")
def safe_merge(left, right, on_cols): #left is the df already built
    #right is the new df and on_cols the columns used as keys(user_id and date)
    left = left.copy()
    right = right.copy()
    # normalize keys in both
    for col in on_cols:
        if col == "user_id":
            left[col] = pd.to_numeric(left[col].astype(str).str.lstrip("u"), errors="coerce")
            right[col] = pd.to_numeric(right[col].astype(str).str.lstrip("u"), errors="coerce")
        elif col == "date":
            left[col] = pd.to_datetime(left[col], errors="coerce").dt.date
            right[col] = pd.to_datetime(right[col], errors="coerce").dt.date
    # Remove rows without keys in the RIGHT column to avoid unusual merges
    right = right.dropna(subset=on_cols)
    left = left.dropna(subset=on_cols)
    # returns user_id to int because of coerce
    if "user_id" in on_cols:
        left["user_id"] = left["user_id"].astype(int)
        right["user_id"] = right["user_id"].astype(int)
    #It keeps the left columns, and adds right columns only when there is a match
    return pd.merge(left, right, on=on_cols, how="left")
#Extract users and dates from these datasets
datasets_with_date = [
    ("app", df_app),
    ("calendar", df_calendar),
    ("call", df_call),
    ("dinning", df_dinning),
    ("deadlines", df_deadlines),
    ("activity", daily_activity),
    ("class_ema", daily_class),
    ("class2_ema", daily_class2),
    ("behavior_ema", daily_behavior),
    ("comment_ema", daily_comment),
    ("dimensions_ema", daily_dimensions),
    ("dining_halls_ema", daily_dining),
    ("events_ema", daily_events),
    ("mood_ema", daily_mood),
    ("mood1_ema", daily_mood1),
    ("mood2_ema", daily_mood2),
    ("sleep_ema", daily_sleep),
    ("social_ema", daily_social),
    ("stress_ema", daily_stress), 
    ("study_spaces_ema", daily_study),
    ("activity_inference", daily_act_inference),
    ("audio_inference", daily_audio),
    ("bluetooth", daily_bt),
    ("dark_screen", daily_dark),
    ("phonecharge", daily_phonecharge),
    ("phonelock", daily_phonelock),
    ("wifi", daily_wifi),
    ("sms", sms_daily),
]
# Build base using ONLY real (user_id, date) pairs that exist in ANY dataset above
pairs = []
for name, df in datasets_with_date:
    if not df.empty and "user_id" in df.columns and "date" in df.columns:#df is not empty and has the necessary columns
        tmp = df[["user_id", "date"]].copy()#Create a temporary df with only the necessary contents.
        # normalize keys to avoid duplicates due to formats
        tmp["user_id"] = pd.to_numeric(tmp["user_id"].astype(str).str.lstrip("u"), errors="coerce")
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.date
        tmp = tmp.dropna(subset=["user_id", "date"])
        tmp["user_id"] = tmp["user_id"].astype(int)
        pairs.append(tmp)
# create base only with real dates
df_base = pd.concat(pairs, ignore_index=True).drop_duplicates()
df_final = df_base.copy()
#Merge all daily datasets onto the base
for name, df in datasets_with_date:
    if not df.empty:
        print(f"Joining dataset: {name}")
        df_final = safe_merge(df_final, df, ["user_id", "date"])

#Merge datasets that are per-user only (no date)
datasets_user_only = [
    ("class", df_class_csv),
    ("grades", df_grades),
    ("piazza", df_piazza),
    ("bigfive_pre", df_bigfive_pre) ,
    ("flourishing_pre", df_flourishing),
    ("loneliness_pre", df_loneliness),
    ("panas_pre", df_panas_final),
    ("pss10_pre", df_stress_final),
    ("phq9_pre", df_phq_final)

]
for name, df in datasets_user_only:
    if not df.empty:
        print(f"Joining dataset by user: {name}")
        df_final = safe_merge(df_final, df, ["user_id"])

# We fill in the durations, counts, and quantities with 0.
# And we fill in with Nans when it means that it was not measured or there was no response.
for c in ["total_apps_used", "num_sessions", "tiempo_uso_total","calendar_event_count","total_calls",
        "total_meals", "breakfast_events", "lunch_events", "supper_events","num_classes", "num_deadlines_today",
        "comment_count","dimensions_count" ,"events_responses","bt_unique_devices", "bt_scans","dark_minutes",
        "charge_minutes","locked_minutes","unique_wifi","num_sms"]:
    if c in df_final.columns:
        df_final[c] = df_final[c].fillna(0)
if "comment_count" in df_final.columns:
    df_final["comment_count"] = df_final["comment_count"].astype(int) #int instead of float
if "dimensions_count" in df_final.columns:
    df_final["dimensions_count"] = df_final["dimensions_count"].astype(int)
# Save final CSV
output_path = "dataset/final_dataset_complete.csv"
df_final.to_csv(output_path, index=False)
print(f"\nFinal dataset saved at: {output_path}")
