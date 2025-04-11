# chat_interface.py
import streamlit as st
from langgraph.graph import START
from agents.supervisor import create_career_graph 
def chat_interface():
    st.title("Career Path Advisor")
    
    # Initialize the graph
    graph = create_career_graph()
    
    with open("assets/graph.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())

    # Initialize session state for messages if not already done
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Enter your LinkedIn profile URL or job details"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare initial state
        initial_state = {
            "messages": st.session_state.messages,
            "profile_analysis_complete": False,
            "job_fit_complete": False,
            "career_guidance_complete": False,
            "current_step": "start",
            "steps_taken": 0
        }
        
        # Run the graph
        try:
            stream = graph.stream(initial_state)
            
            full_response = ""
            with st.chat_message("assistant"):
                for step in stream:
                    for node, output in step.items():
                        if node in ["profile_analysis", "job_fit", "career_guidance"]:
                            # Assuming the last message contains the response
                            response = output.get('messages', [])
                            if response:
                                response_content = response[-1].content if hasattr(response[-1], 'content') else str(response[-1])
                                full_response += f"**{node.replace('_', ' ').title()}:**\n{response_content}\n\n"
                                st.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response
            })
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            # Log the error for debugging
            import traceback
            traceback.print_exc()

def main():
    chat_interface()

if __name__ == "__main__":
    main()
