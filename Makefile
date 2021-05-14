# ----------------------------------
#         API COMMANDS
# ----------------------------------

run_api:
	uvicorn api.fast:app --reload  # load web server with code autoreload

# ----------------------------------
#         HEROKU COMMANDS
# ----------------------------------

streamlit:
	-@streamlit run titanic_app.py
