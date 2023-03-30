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
