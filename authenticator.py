import streamlit as st
import streamlit_authenticator as stauth

# Define user credentials
credentials = {
    "usernames": {
        "admin": {"name": "Admin User", "password": stauth.Hasher(["securepassword"]).generate()[0]}
    }
}

authenticator = stauth.Authenticate(credentials, "app_home", "auth", cookie_expiry_days=30)

# Login widget
name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status:
    st.write(f"Welcome, {name}!")
    authenticator.logout("Logout", "sidebar")
    # Proceed with the dashboard
elif authentication_status is False:
    st.error("Username or password is incorrect")