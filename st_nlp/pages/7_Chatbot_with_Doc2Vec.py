import streamlit as st
import time
from cache_func import load_company_tagged, load_doc2vec, find_similar_doc

df = load_company_tagged()
d2v = load_doc2vec()

st.title("Chatbot Using Doc2Vec Search")

st.subheader("Using Doc2Vec we can make a pseudo Chatbot that gives the best matches for a certain request")
st.subheader("Explanations on the Performance of the Doc2Vec Search")
st.write('We can obviously see that the Doc2Vec model performance is lacking for many requests. For example, if we type "jewelry", "watch" or "wine", The model gives us results that make sense. For some requests like "shoes" or "animals", there are some results which are not plausible.')
st.write('We can clearly see the limitations of such a model, but we can also see the limitations of our text dataset.')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    for response in message["responses"]:
        with st.chat_message(message["role"]):
            st.markdown(response)
# Accept user input
if prompt := st.chat_input("What are your needs?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "responses": []})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    best_match = find_similar_doc(d2v, prompt, df)
    best_match = [int(x) for x in best_match]
    df_match = df[df['tag'].isin(best_match[:3])]

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = f"Here are the best matches found for {prompt}: "
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)


    responses = []
    for index, row in df_match.iterrows():
        time.sleep(0.5)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response1 = ""
            assistant_response = row["company_name"] + " : " + row["description_en"]
            for chunk in assistant_response.split():
                response1 += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(response1 + "▌")
            message_placeholder.markdown(response1)
            responses.append(response1)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response, "responses": responses})