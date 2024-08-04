import streamlit as st

def main():
    st.set_page_config(page_title="Proposal Toolkit")

    # Main content
    st.title("Proposal Toolkit")

    # Chat History
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("Chat History")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You**: {msg['content']}")
        else:
            st.markdown(f"**Assistant**: {msg['content']}")

    # Chat input
    user_input = st.text_input("Enter your query to search in documents and craft new content", key="user_input")
    if st.button("Send"):
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        st.markdown(f"**You**: {user_input}")

        # Example response (in real use case, replace with actual model response)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "This is a response from the assistant."}
        )
        st.markdown(f"**Assistant**: This is a response from the assistant.")

if __name__ == "__main__":
    main()
