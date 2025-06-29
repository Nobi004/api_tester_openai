import streamlit as st
import openai
from openai import OpenAI
import time
from typing import Dict, List, Tuple

# Page configuration
st.set_page_config(
    page_title="OpenAI API Key Tester",
    page_icon="🔑",
    layout="centered"
)

# Initialize session state
if 'test_results' not in st.session_state:
    st.session_state.test_results = {}

def test_single_model(client: OpenAI, model: str) -> Tuple[bool, str]:
    """
    Test a single model with the provided API client.
    Returns (success: bool, message: str)
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
            temperature=0
        )
        return True, "Successfully connected"
    except openai.AuthenticationError:
        return False, "Authentication failed - Invalid API key"
    except openai.PermissionDeniedError:
        return False, "Permission denied - No access to this model"
    except openai.NotFoundError:
        return False, "Model not found"
    except openai.RateLimitError:
        return False, "Rate limit exceeded"
    except openai.APIError as e:
        return False, f"API error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def get_available_models(client: OpenAI) -> List[str]:
    """
    Fetch available models from OpenAI API.
    Returns list of model IDs that are GPT models.
    """
    try:
        models = client.models.list()
        gpt_models = [
            model.id for model in models.data 
            if 'gpt' in model.id.lower() and any(
                variant in model.id for variant in ['gpt-4', 'gpt-3.5-turbo']
            )
        ]
        return sorted(gpt_models)
    except Exception:
        # If we can't fetch models, return default list
        return []

def test_api_key(api_key: str) -> Dict[str, Tuple[bool, str]]:
    """
    Test the API key against multiple models.
    Returns dictionary with model names as keys and (success, message) tuples as values.
    """
    # Default models to test
    default_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4-32k",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k"
    ]
    
    results = {}
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Try to get available models
        available_models = get_available_models(client)
        
        # Combine default and available models, removing duplicates
        models_to_test = list(set(default_models + available_models))
        models_to_test.sort()
        
        # Test each model
        for model in models_to_test:
            success, message = test_single_model(client, model)
            results[model] = (success, message)
            # Small delay to avoid rate limiting
            time.sleep(0.1)
            
    except openai.AuthenticationError:
        # If authentication fails at client creation
        for model in default_models:
            results[model] = (False, "Invalid API key")
    except Exception as e:
        # General error
        for model in default_models:
            results[model] = (False, f"Client error: {str(e)}")
    
    return results

# UI Components
st.title("🔑 OpenAI API Key Tester")
st.markdown("Test your OpenAI API key across multiple GPT models")

# API Key input
api_key = st.text_input(
    "Enter your OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="Your API key will not be stored and is only used for testing"
)

# Test button
if st.button("Test API Key", type="primary", use_container_width=True):
    if not api_key:
        st.error("Please enter an API key")
    elif not api_key.startswith("sk-"):
        st.warning("⚠️ API key should start with 'sk-'")
    else:
        with st.spinner("Testing API key across models..."):
            st.session_state.test_results = test_api_key(api_key)

# Display results
if st.session_state.test_results:
    st.markdown("---")
    st.subheader("Test Results")
    
    # Count successes and failures
    successes = sum(1 for success, _ in st.session_state.test_results.values() if success)
    total = len(st.session_state.test_results)
    
    # Overall summary
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Models Tested", total)
    with col2:
        st.metric("Successful", f"{successes}/{total}")
    
    # Detailed results in an expander
    with st.expander("Detailed Results by Model", expanded=True):
        # Separate successful and failed models
        successful_models = []
        failed_models = []
        
        for model, (success, message) in sorted(st.session_state.test_results.items()):
            if success:
                successful_models.append((model, message))
            else:
                failed_models.append((model, message))
        
        # Display successful models first
        if successful_models:
            st.markdown("### ✅ Accessible Models")
            for model, message in successful_models:
                st.success(f"**{model}**: {message}")
        
        # Display failed models
        if failed_models:
            st.markdown("### ❌ Inaccessible Models")
            for model, message in failed_models:
                st.error(f"**{model}**: {message}")
    
    # Final summary
    st.markdown("---")
    if successes == total:
        st.success("✅ **API Key is valid** - All models accessible!")
    elif successes > 0:
        st.warning(f"⚠️ **API Key is partially valid** - Access to {successes}/{total} models")
    else:
        st.error("❌ **API Key failed** - No models accessible")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <small>
        💡 Note: Some models may not be accessible based on your API tier or permissions.<br>
        🔒 Your API key is not stored and is only used for testing connectivity.
        </small>
    </div>
    """,
    unsafe_allow_html=True
)