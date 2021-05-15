mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"victor.bonnet.mg@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[theme]\n\
base=\"light\"\n\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
