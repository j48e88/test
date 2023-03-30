import pandas as pd
import plotly.express as px
import streamlit as st


# Set page configuration
st.set_page_config(page_title="Employee on Duty Record", page_icon=":bar_chart", layout="wide")
st.title("Crew Data Analysis")

# Upload data
uploaded_file = st.file_uploader("Upload an Excel file", type="xlsx")

# Define a function to read the uploaded Excel file and return a DataFrame
@st.cache_data
def read_data(file):
    df = pd.read_excel(file, engine='openpyxl', usecols='A:AL', nrows=5000)
    blockon = pd.to_datetime(df['Block_On_Time'], format='%H:%M:%S')
    blockoff = pd.to_datetime(df['Block_Off_Time'], format='%H:%M:%S')

    blockon_decimal = blockon.dt.hour + blockon.dt.minute/60 + blockon.dt.second/3600
    blockoff_decimal = blockoff.dt.hour + blockoff.dt.minute/60 + blockoff.dt.second/3600

    blockon = blockon + pd.DateOffset(days=1)
    diff = blockon - blockoff

    diff_str = diff.astype(str)
    diff_str = diff_str.apply(lambda x: x.split()[-1])

    df.insert(3, "Block_Diff", diff_str)

    diff = pd.to_datetime(df['Block_Diff'], format='%H:%M:%S')
    diff_decimal = round(diff.dt.hour + diff.dt.minute/60 + diff.dt.second/3600, 0)

    df.insert(4, "Block_Diff_Round", diff_decimal)
    df['Block_Diff_Round'] = df['Block_Diff_Round'].fillna(0)
    return df

if uploaded_file is not None:
    df = read_data(uploaded_file)

    # Sidebar for user input and there is no default value, also the options are sorted
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; color: black; font-size: 30px;'>Please Filter Here:</h1>", unsafe_allow_html=True)
        departure = st.multiselect("Select the Departure:", options=sorted(df["Departure"].astype(str).unique()), default=sorted(df["Departure"].astype(str).unique()), key="departure")
        destination = st.multiselect("Select the Destination:", options=sorted(df["Destination"].astype(str).unique()), default=sorted(df["Destination"].astype(str).unique()), key="destination")
        ac_type = st.multiselect("Select the Aircraft Type:", options=df["AcType"].unique(), default=df["AcType"].unique(), key="ac_type")
        int1 = st.multiselect("Select the Block_Diff:", options=sorted(df["Block_Diff_Round"].unique()), default=sorted(df["Block_Diff_Round"].unique()), key="Block_Diff")

    # Let the user to sort the data by emp_no, the application is depended on the user demand
    st.markdown("<h1 style='text-align: left; color: black; font-size: 25px;'>Enter Emp_No:</h1>", unsafe_allow_html=True)
    emp_no = st.text_input("(If certain employee is needed)", value="", key="Emp_No")


    if emp_no:
        df_filtered = df[df['Emp_No'].astype(str).str.contains(emp_no)]
        if not df_filtered.empty:
            st.write(df_filtered)
            num_records = len(df_filtered)
            st.write(f"<h2 style='font-size:20px;'>Number of records matching the selected conditions: <b><span style='font-size: 25px; color: red; font-weight:bold;'>{num_records}</span></b></h2>", unsafe_allow_html=True)
        else:
            st.write("No records found matching the entered Emp_No")
    else:
        df_filtered = df.query("Departure == @departure & Destination == @destination & AcType == @ac_type & Block_Diff_Round == @int1").reset_index(drop=True)
        st.write(df_filtered)
        num_records = len(df_filtered)
        st.write(f"<h2 style='font-size:20px;'>Number of records matching the selected conditions: <b><span style='font-size: 25px; color: red; font-weight:bold;'>{num_records}</span></b></h2>", unsafe_allow_html=True)
   
    # Display bar chart of destination counts
    if not df_filtered.empty:  
    # Count the number of occurrences for each destination
        blockdiffround_counts = df_filtered['Block_Diff_Round'].value_counts()
        destination_counts = df_filtered['Destination'].value_counts()
        departure_counts = df_filtered['Departure'].value_counts()

        if not blockdiffround_counts.empty:
            most_common_blockdiff = blockdiffround_counts.index[0]
            most_common_count = blockdiffround_counts.iloc[0]
            st.write(f"<h2 style='font-size:20px;'>Most common flight time: <b><span style='font-size: 25px; color: blue; font-weight:bold;'>{most_common_blockdiff}</span></b> ({most_common_count} flights)</h2>", unsafe_allow_html=True)
        else:
            st.write("No destinations found matching the selected conditions")

        if not destination_counts.empty:
    # Get the most common destination and its count
            most_common_destination = destination_counts.index[0]
            most_common_count = destination_counts.iloc[0]
            st.write(f"<h2 style='font-size:20px;'>Most common destination: <b><span style='font-size: 25px; color: blue; font-weight:bold;'>{most_common_destination}</span></b> ({most_common_count} flights)</h2>", unsafe_allow_html=True)
        else:
            st.write("No destinations found matching the selected conditions")
    
        if not df_filtered.empty:
            destination_counts = df_filtered['Destination'].value_counts().sort_values(ascending=False)
            fig = px.bar(destination_counts, x=destination_counts.index, y=destination_counts.values, 
                         title="Common Destinations", labels={"x": "Destination", "y": "Count"})
            fig.update_traces(text=destination_counts.values, textposition='outside')
            fig.update_layout(xaxis_title="Number of Flights", yaxis_title="Destination", xaxis_tickangle=-45, xaxis_tickfont_size=12, yaxis_tickfont_size=12, title_font_size=20)
            st.plotly_chart(fig)

    # Get the most common departure and its count
            most_common_departure = departure_counts.index[0]
            most_common_count = departure_counts.iloc[0]
            st.write(f"<h2 style='font-size:20px;'>Most common departure: <b><span style='font-size: 25px; color: blue; font-weight:bold;'>{most_common_departure}</span></b> ({most_common_count} flights)</h2>", unsafe_allow_html=True)        
        if not df_filtered.empty:
            destination_counts = df_filtered['Departure'].value_counts().sort_values(ascending=False)
            fig = px.bar(destination_counts, x=destination_counts.index, y=destination_counts.values, 
                         title="Common Departure", labels={"x": "Departure", "y": "Count"})
            fig.update_traces(text=departure_counts.values, textposition='outside')
            fig.update_layout(xaxis_title="Number of Flights", yaxis_title="Departure", xaxis_tickangle=-45, xaxis_tickfont_size=12, yaxis_tickfont_size=12, title_font_size=20)
            st.plotly_chart(fig)
        
        else:
            st.write("No records found matching the selected conditions")

    # Display pie chart of AcType counts
    if st.button("Show commonly used Aircraft Types"):
        ac_type_counts = df_filtered['AcType'].value_counts()
        fig = px.pie(ac_type_counts, values=ac_type_counts.values, names=ac_type_counts.index, title="Percentage of Aircraft Types")
        fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=12, textposition='outside')
        fig.update_layout(title_font_size=20)
        st.plotly_chart(fig)

import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

st.title("Flight Data Analysis")
# Add a file uploader to allow the user to upload an Excel file
uploaded_file2 = st.file_uploader("Upload the Flight Schedule file", type=["xlsx"])


# Define a function to preprocess the data
def preprocess_data(df):
    # Drop all rows that have a 'CNL' status from the original DataFrame
    df.drop(df[df['Status'] == 'CNL'].index, inplace=True)

    # split the "STD/ETD" column into two columns
    df[['STD', 'ETD']] = df['STD/ETD'].str.split('/', expand=True)
    df[['STA', 'ETA']] = df['STA/ETA'].str.split('/', expand=True)
    # drop the original "STD/ETD" column
    df.drop('STD/ETD', axis=1, inplace=True)
    df.drop('STA/ETA', axis=1, inplace=True)

    # Convert the 'Type' column to string data type
    df['Type'] = df['Type'].astype(str)
    # Create a new column 'ac_type' containing the first two characters of the 'Type' column
    df['ac_type'] = df['Type'].apply(lambda x: x[:2] if len(x) >= 2 else None)

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    # Convert the 'Date' column to the desired format
    df['Date'] = df['Date'].dt.strftime('%d%b')

    df['STD'] = df['STD'].str.strip()
    stds = pd.to_datetime(df['STD'], format='%H:%M')
    stds = stds.dropna()
    df['STA'] = df['STA'].str.strip()
    stas = pd.to_datetime(df['STA'], format='%H:%M')
    stas = stas.dropna()
    BlockOff = pd.to_datetime(df['STD'], format='%H:%M')
    BlockOn = pd.to_datetime(df['STA'], format='%H:%M')
    BlockOn = BlockOn + pd.DateOffset(days=1)
    diff = BlockOn - BlockOff
    diff_str = diff.astype(str)
    diff_str = diff_str.apply(lambda x: x.split()[-1])
    df.insert(15, "Diff", diff_str)

    return df

def parse_date(date_str):
    if isinstance(date_str, str):
        return pd.datetime.strptime(date_str, '%d%b')
    else:
        return date_str


if uploaded_file2 is not None:
    # Read the uploaded file and preprocess the data
    df = pd.read_excel(uploaded_file2, skiprows=3, parse_dates=['Date'], date_parser=parse_date)
    df = df.dropna()
    if 'ac_type' in df.columns:
        df['ac_type'] = df['ac_type'].astype(str).tolist()

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    stc = df['STC']
    df = df.rename(columns={'Flight No': 'Flight_No'})
    df = preprocess_data(df)

    # Display the preprocessed data and allow the user to download it as an Excel file
    st.write("Preprocessed Data:")
    st.write(df)

    def calculate_num_layover(count, dep, arr):
        dep, arr = sorted([dep, arr])
        if count % 2 == 1:
            return (count - 1) // 2
        else:
            return 0

    def calculate_num_nonregular(groups, date):
        non_regular_count = 0
        for group, count in groups.items():
            if group[0] == date and group[2] == ac_type:
                dep = group[1][0]
                arr = group[1][1]
                if dep == 'HKG' and arr == 'HKG':
                    non_regular_count += count
        return non_regular_count

    groups = {}
    ac_32 = []
    ac_33 = []

    for index, row in df.iterrows():
        dep = row['DepStn']
        arr = row['ArrStn']
        ac_type = row['ac_type']
        group = (datetime.strptime(row['Date'], '%d%b').strftime('%d%b'), tuple(sorted([dep, arr])), row['ac_type'])
        if group not in groups:
            groups[group] = 1
        else:
            groups[group] += 1
        if row['ac_type'] == '32':
            ac_32.append(row['Flight_No'])
        elif row['ac_type'] == '33':
            ac_33.append(row['Flight_No'])

    # Group flights by aircraft and date
    ac_dates = {}
    for group, count in groups.items():
        ac_type = group[2]
        date = group[0]
        if ac_type not in ac_dates:
            ac_dates[ac_type] = {}
        if date not in ac_dates[ac_type]:
            ac_dates[ac_type][date] = []
        ac_dates[ac_type][date].append(count)

    # Calculate number of turnarounds for each group
    num_turnarounds = {}
    for ac_type, dates in ac_dates.items():
        for date, counts in dates.items():
            num_turnarounds[(date, ac_type)] = sum([count // 2 for count in counts])

    # Calculate number of layovers for each group
    num_layovers = {}
    for group, count in groups.items():
        dep = group[1][0]
        arr = group[1][1]
        ac_type = group[2]
        num_layovers[(group[0], ac_type)] = calculate_num_layover(count, dep, arr)

    # Calculate number of non-regular flights for each date
    num_nonregular = {}
    for group, count in groups.items():
        date = group[0]
        ac_type = group[2]
        num_nonregular[(group[0], ac_type)] = calculate_num_nonregular(groups, date)

    def calculate_crew_num(ac_type, num_turnaround, num_layover, is_nonreg=False, num_nonreg=0):
        if is_nonreg:
            return 4 * num_nonreg + 7 * (num_turnaround + num_layover) if ac_type == '32' else 11 * (num_turnaround + num_layover)
        elif ac_type == '32':
            return 7 * (num_turnaround + num_layover)
        elif ac_type == '33':
            return 11 * (num_turnaround + num_layover)
        else:
            return None

    # Calculate total number of crew
    total_crew_num = 0
    for (date, ac_type), num_turnaround in num_turnarounds.items():
        num_layover = num_layovers[(date, ac_type)]
        num_nonreg = num_nonregular[(date, ac_type)]
        if row['STC'] == 'J':
            if num_nonreg > 0:
                num_crew = calculate_crew_num(ac_type, num_turnaround, num_layover, is_nonreg=True, num_nonreg=num_nonreg)
            else:
                num_crew = calculate_crew_num(ac_type, num_turnaround, num_layover)
            if num_crew is not None:
                total_crew_num += num_crew
        elif row['STC'] != 'J':
            num_crew = 4

    # Calculate number of crew on each day
    daily_crew_nums = {}
    for (date, ac_type), num_turnaround in num_turnarounds.items():
        num_layover = num_layovers[(date, ac_type)]
        num_nonreg = num_nonregular[(date, ac_type)]
        if num_nonreg > 0:
            num_crew = calculate_crew_num(ac_type, num_turnaround, num_layover, is_nonreg=True, num_nonreg=num_nonreg)
        else:
            num_crew = calculate_crew_num(ac_type, num_turnaround, num_layover)
        if num_crew is not None:
            daily_crew_nums[date] = daily_crew_nums.get(date, 0) + num_crew 

    # Print results sorted by date
    if st.button("Show Flights grouped for Dep and Des"):
        st.markdown("<h1 style='text-align: left; color: black; font-size: 30px;'>Flights in groups:</h1>", unsafe_allow_html=True)
        for group, count in sorted(groups.items()):
            st.write(group[0], group[1], ':', count)
    st.write('\n')
    if st.button("Show Flights info"):
        st.markdown("<h1 style='text-align: left; color: black; font-size: 30px;'>Date: Aircraft Type: TurnArounds : Layovers : and Non-Regular:</h1>", unsafe_allow_html=True)
        for (date, ac_type), num_turnaround in sorted(num_turnarounds.items()):
            num_layover = num_layovers[(date, ac_type)]
            num_nonreg = num_nonregular[(date, ac_type)]
            st.write(f"{date} : {ac_type} : {num_turnaround} turnarounds, {num_layover} layovers, {num_nonreg} non-regular flights") 
        # Calculate the total number of turnarounds, layovers, and non-regular flights
        st.write('\n')
        total_turnaround = sum(num_turnarounds.values())
        total_layover = sum(num_layovers.values())
        total_nonreg = sum(num_nonregular.values())
        # Display the total number of turnarounds, layovers, and non-regular flights
        st.markdown("<h1 style='text-align: left; color: black; font-size: 30px;'>Total Number:</h1>", unsafe_allow_html=True)
        st.write(f"Turnaround: {total_turnaround}, Layover: {total_layover}, Non-Regular Flights: {total_nonreg}")

    st.write('\n')
    if st.button("Show Crew info"):
        for date, crew_num in daily_crew_nums.items():
            st.write(f"{date} : Required Crew = {crew_num}")
        st.markdown("<h1 style='text-align: left; color: black; font-size: 30px;'>Total Num of Crew Needed:</h1>", unsafe_allow_html=True)
        st.write(f"{total_crew_num}")
        st.markdown("<h1 style='text-align: left; color: black; font-size: 30px;'>Aircraft Types Counts:</h1>", unsafe_allow_html=True)
        st.write("\nType Series 320: ", len(ac_32))
        st.write("\nType Series 330: ", len(ac_33))
        
    fdp_rules = {
    "0700-0759": {1: 13, 2: 12.25, 3: 11.5, 4: 10.75, 5: 10, 6: 9.25, 7: 9, 8: 9},
    "0800-1259": {1: 14, 2: 13.25, 3: 12.5, 4: 11.75, 5: 11, 6: 10.25, 7: 9.5, 8: 9},
    "1300-1759": {1: 13, 2: 12.25, 3: 11.5, 4: 10.75, 5: 10, 6: 9.25, 7: 9, 8: 9},
    "1800-2159": {1: 12, 2: 11.25, 3: 10.5, 4: 9.75, 5: 9, 6: 9, 7: 9, 8: 9},
    "2200-0659": {1: 11, 2: 10.25, 3: 9.5, 4: 9, 5: 9, 6: 9, 7: 9, 8: 9}
    }

    df = df.dropna()
    departures = df['DepStn'].tolist()
    arrivals = df['ArrStn'].tolist()
    # Remove trailing spaces from the 'STD' column
    df['STD'] = df['STD'].str.strip()
    # Parse the 'STD' column as datetime
    stds = pd.to_datetime(df['STD'], format='%H:%M')
    stds = stds.dropna()

    # Remove trailing spaces from the 'STD' column
    df['STA'] = df['STA'].str.strip()
    # Parse the 'STD' column as datetime
    stas = pd.to_datetime(df['STA'], format='%H:%M')
    stas = stas.dropna()
    aircraft_types = df['Type'].tolist()
    flight_no = df['Flight_No'].tolist()
    reg = df['Reg'].str.upper().astype(str).tolist()
    df['date'] = pd.to_datetime(df['Date'], format='%d%b')
    date = df['date']

    reporting_time = [(std - timedelta(minutes=75)).strftime('%H%M') for std in stds]
    fdt = pd.to_datetime(df['Diff'], format='%H:%M:%S')
    fdt_decimal = fdt.dt.hour + fdt.dt.minute/60 + fdt.dt.second/3600
    df['diff decimal'] = fdt_decimal
    diff_sum = round(df['diff decimal'].sum(), 2)
    n_flights = len(departures)

    valid_flights = set()  # set to store flights with a valid connection
    checked = set()  # set to store checked flights
    valid_count = 0  # counter for valid flights
    invalid_count = 0
    turn = []
    lay = []

    if reporting_time:
        time_ranges = []
        for rt in reporting_time:
            rt_obj = datetime.strptime(rt, '%H%M')
            time_0700 = datetime.strptime("07:00", "%H:%M")
            time_0800 = datetime.strptime("08:00", "%H:%M")
            time_1300 = datetime.strptime("13:00", "%H:%M")
            time_1800 = datetime.strptime("18:00", "%H:%M")
            if time_0700 <= rt_obj <= datetime.strptime("07:59", "%H:%M"):
                time_ranges.append("0700-0759")
            elif time_0800 <= rt_obj <= datetime.strptime("12:59", "%H:%M"):
                time_ranges.append("0800-1259")
            elif time_1300 <= rt_obj <= datetime.strptime("17:59", "%H:%M"):
                time_ranges.append("1300-1759")
            elif time_1800 <= rt_obj <= datetime.strptime("21:59", "%H:%M"):
                time_ranges.append("1800-2159")
            else:
                time_ranges.append("2200-0659")

        df['Time_Range'] = time_ranges

    st.write('\n')
    if st.button("Show the turnaround / layover flight"):
        for i in range(len(df)):
            flight1 = df.iloc[i]
            valid_connection = False

            # check for valid connections
            for j in range(i+1, len(df)):
                flight2 = df.iloc[j]
                if (flight1['ArrStn'] == flight2['DepStn'] and flight1['DepStn'] == flight2['ArrStn'] and
                    flight1['Flight_No'] != flight2['Flight_No'] and
                    all(flight1['Flight_No'] not in x and flight2['Flight_No'] not in x for x in valid_flights)):
                    valid_connection = True
                    sum_fdt = round(flight1['diff decimal'] + flight2['diff decimal'], 2)
                    connection = (flight1['DepStn'], flight1['ArrStn'], flight2['DepStn'], flight2['ArrStn'])
                    valid_flights.add(f"{flight1['Flight_No']} and {flight2['Flight_No']}")
                    valid_count += 1
                    turn.append(flight2['Flight_No'])
                    fdp = fdp_rules[flight1['Time_Range']][2] if flight2['Flight_No'] in turn else fdp_rules[flight1['Time_Range']][1]
                    remaintime = round(fdp - sum_fdt ,2)
                    st.write(f"\nFlights {flight1['Flight_No']} ({flight1['DepStn']} and {flight1['ArrStn']}) and {flight2['Flight_No']} ({flight2['DepStn']} and {flight2['ArrStn']}) is a turnaround flight. The est.FDT is {sum_fdt} hr. The FDP is {fdp} hr. The remaining time is {remaintime} hr.")

            # check for invalid connections
            if not valid_connection and all(flight1['Flight_No'] not in x for x in valid_flights):
                sum_fdt = round(flight1['diff decimal'], 2)
                invalid_count += 1
                lay.append(flight1['Flight_No'])
                fdp = fdp_rules[flight1['Time_Range']][1] if flight1['Flight_No'] in lay else fdp_rules[flight1['Time_Range']][2]
                st.write(f"\nFlight {flight1['Flight_No']} ({flight1['DepStn']} and {flight1['ArrStn']}) is a layover flight. The est.FDT is {sum_fdt} hr. The FDP is {fdp} hr.")

        st.write(f"\n{valid_count} pairs of flight(s) is/are turnaround.")
        st.write(f"{invalid_count} flight(s) is/are layover.")

        total = valid_count + invalid_count
        st.write(f"There are in total {total} flights within this month.")
