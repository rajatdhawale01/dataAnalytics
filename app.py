import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# Streamlit App
# -------------------------
st.title("📊 AI Data Analysis Assistant")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="main_uploader")

if uploaded_file:
    # Read dataset
    df = pd.read_csv(uploaded_file)
    st.success("✅ File loaded successfully!")
    st.write(df.head())

    # -------------------------
    # 1. Dataset Summary
    # -------------------------
    st.subheader("📋 Dataset Summary")
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.write("Missing Values:", df.isnull().sum())
    st.write("Duplicate Rows:", df.duplicated().sum())

    # -------------------------
    # 2. Quick Statistics
    # -------------------------
    st.subheader("📊 Quick Statistics")
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        st.write(df[numeric_cols].describe())
        st.write("Correlation Matrix:")
        st.write(df[numeric_cols].corr())
    else:
        st.info("No numeric columns found.")

    # -------------------------
    # 3. Filter / Group (Dynamic)
    # -------------------------
    st.subheader("🔍 Filter Data")
    col = st.selectbox("Select column to filter", df.columns, key="filter_column")
    condition = st.text_input("Enter condition (examples: >50, == 'HR', null, notnull)", key="filter_condition")

    if st.button("Apply Filter"):
        try:
            if condition.lower() == "null":
                filtered = df[df[col].isnull()]
            elif condition.lower() == "notnull":
                filtered = df[df[col].notnull()]
            else:
                filtered = df.query(f"{col}{condition}")
            st.write(filtered.head())
        except Exception as e:
            st.error(f"Error: {e}")

    # -------------------------
    # 4. Simple Plots
    # -------------------------
    st.subheader("📈 Visualizations")
    if len(numeric_cols) >= 2:
        st.write("Scatter Plot (first two numeric cols):")
        st.scatter_chart(df[numeric_cols].iloc[:, :2])

    if len(numeric_cols) > 0:
        st.write("Histograms:")
        st.bar_chart(df[numeric_cols])

    # -------------------------
    # 5. Natural Language Queries (AI-powered)
    # -------------------------
    

    st.subheader("🤖 Ask AI About Your Data")
    query = st.text_input("Enter your question (e.g., Average salary by department?)", key="ai_query")
    
    if query:
        with st.spinner("AI is thinking..."):
            try:
                # Ask OpenAI to generate pandas code
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": f"You are a pandas expert. Available columns: {df.columns.tolist()}"},
                        {"role": "user", "content": f"Write ONLY pandas code (no explanation) to answer: {query}. The dataframe is named df. Always assign the final result to a variable named result."}
                    ]
                )
    
                code = response.choices[0].message.content
    
                # 🔧 Clean response
                code = re.sub(r"```(python)?", "", code)   # remove ```python
                code = code.replace("```", "")            # remove closing ```
                code = code.replace("’", "'").replace("“", '"').replace("”", '"')
                
                # Keep only code lines (drop explanations if AI adds them)
                code_lines = [line for line in code.split("\n") if not line.strip().startswith("#") and "You can find" not in line]
                code = "\n".join(code_lines).strip()
    
                st.code(code, language="python")
    
                # Run the AI-generated code safely
                local_env = {"df": df}
                exec(code, {}, local_env)
    
                if "result" in local_env:
                    st.write("✅ Result:")
                    st.write(local_env["result"])
            except Exception as e:
                st.error(f"⚠️ AI could not run query: {e}")
