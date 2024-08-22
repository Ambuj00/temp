import openai
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, func
from sqlalchemy.sql import select

# Function to generate the database schema as a string with data types
def generate_schema(df):
    schema = ""
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype == 'int64':
            col_type = "INTEGER"
        elif dtype == 'float64':
            col_type = "FLOAT"
        else:
            col_type = "TEXT"
        schema += f"{col} ({col_type}), "
    return schema.rstrip(', ')

# Function to construct the prompt for the GPT model, optimized for complex queries
def construct_prompt(natural_language_query, schema):
    prompt = f"""
You are an advanced AI assistant that converts natural language into SQL queries.

Here is the database schema:
Table: data
Columns:
{schema}

Generate an accurate and efficient SQL query for the following request:
"{natural_language_query}"

Consider complex operations like grouping, filtering, clustering, and analyzing trends.
Only provide the SQL query.
    """
    return prompt

# Function to generate the SQL query using OpenAI's GPT model
def generate_sql_query(natural_language_query, schema, openai_api_key):
    client = openai.OpenAI(api_key=openai_api_key)
    prompt = construct_prompt(natural_language_query, schema)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Use 'gpt-4' if available
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0,
    )
    sql_query = response.choices[0].message.content.strip()
    return sql_query

# Function to create the database table
def create_database_table(df, engine):
    metadata = MetaData()

    columns = []
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == 'int64':
            col_type = Integer
        elif dtype == 'float64':
            col_type = Float
        else:
            col_type = String

        columns.append(Column(col, col_type))

    data_table = Table('data', metadata, *columns)
    metadata.create_all(engine)

    return data_table

# Function to execute the SQL query and handle errors, optimized for complex queries
def execute_sql_query(engine, sql_query):
    try:
        print(f"Executing SQL query: {sql_query}")
        result_df = pd.read_sql_query(sql_query, con=engine)
        return result_df, None  # Return result and no error
    except Exception as e:
        error_message = str(e).split(':')[-1].strip()
        if 'no such table' in error_message.lower():
            error_message = "The query could not find the specified table."
        elif 'syntax error' in error_message.lower():
            error_message = "The query has a syntax error."
        else:
            error_message = "An error occurred while executing the query."

        return None, error_message

# Main function to run the Streamlit app
def main():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "current_query" not in st.session_state:
        st.session_state["current_query"] = ""

    st.title("CSV Data Query Chatbot")

    # Sidebar for OpenAI API key and file upload
    st.sidebar.header("OpenAI API Key")
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password", placeholder="Your OpenAI API Key")
    st.sidebar.header("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = [
            "Page title and screen name", "Country", "Views", 
            "Users", "Views per user", "Average engagement time", 
            "Event count", "Key events"
        ]

        engine = create_engine('sqlite://', echo=False)
        create_database_table(df, engine)
        df.to_sql('data', con=engine, index=False, if_exists='replace')

        st.subheader("Database Preview")
        preview_df = df.iloc[8:]
        st.dataframe(preview_df)

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for entry in st.session_state["history"]:
            st.markdown(f'<div class="chat-bubble user-bubble">{entry["query"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble ai-bubble"><strong>Generated SQL Query:</strong><br><pre><code class="sql">{entry["sql_query"]}</code></pre></div>', unsafe_allow_html=True)
            if entry["table"]:
                st.markdown(f'<div class="chat-bubble ai-bubble"><strong>Result:</strong><br>{entry["table"]}</div>', unsafe_allow_html=True)
            elif entry["response"]:
                st.markdown(f'<div class="chat-bubble ai-bubble"><strong>Result:</strong><br>{entry["response"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        user_query = st.text_area("Type your query here:", st.session_state["current_query"])

        if openai_api_key:
            if st.button("Submit Query"):
                if st.session_state["current_query"] != user_query:
                    st.session_state["current_query"] = user_query
                    with st.spinner('Generating SQL query...'):
                        schema = generate_schema(df)
                        sql_query = generate_sql_query(user_query, schema, openai_api_key)

                        with st.spinner('Executing SQL query...'):
                            result, error = execute_sql_query(engine, sql_query)

                        if result is not None:
                            if result.empty:
                                response_text = "The query executed successfully but returned no results."
                                table_html = "" 
                            else:
                                if 'table' in user_query.lower():
                                    table_html = result.to_html(index=False)
                                    response_text = ""
                                else:
                                    response_text = result.to_string(index=False)
                                    table_html = ""
                        else:
                            response_text = f"Error: {error}"
                            table_html = ""

                        st.session_state["history"].append({
                            "query": user_query,
                            "sql_query": f"{sql_query}",
                            "response": response_text,
                            "table": table_html
                        })
                else:
                    st.warning("Please enter a new query to proceed.")
        else:
            st.warning("Please enter your OpenAI API key to proceed.")

if __name__ == "__main__":
    main()
