# ============================================================================
# SECURITY CONFIG
# ============================================================================
# These will look for environment variables, otherwise use defaults
APP_USER = os.getenv("AGRI_USER", "student")
APP_PASS = os.getenv("AGRI_PASS", "vpkbiet2026")



# ============================================================================
# AUTHENTICATION & LAUNCH
# ============================================================================

def verify_user(username, password):
    """
    Function to handle more complex login logic if needed 
    (e.g., checking against a database or specific hashing).
    """
    return username == APP_USER and password == APP_PASS

if __name__ == "__main__":
    print(f"🔐 Starting AgriShield with Authentication...")
    
    # Launch with settings for security and performance
    app.launch(
        auth=verify_user,
        auth_message="🌿 Welcome to AgriShield. Please enter your credentials to access the AI Crop Health Dashboard.",
        
        # Security/Server Settings
        show_error=False,       # Prevents leaking code details on crash
        share=False,            # Set to True only if you need a temporary public URL
        server_name="0.0.0.0",  # Allows access from other devices on the network
        favicon_path=None       # You can point this to a .ico file for a custom tab icon
    )

export AGRI_USER="your_name"
export AGRI_PASS="your_secure_password"
python a.py
