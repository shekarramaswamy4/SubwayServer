# SubwayServer

SubwayServer is a telegram bot that lets you query the arrival/departure times for subways at specific stations.

<img width="670" height="388" alt="Screenshot 2025-11-04 at 11 09 56 AM" src="https://github.com/user-attachments/assets/700158ef-5d87-4d92-80a0-7291e254acbe" />

# How to use
Go to [t.me/subwayserver_bot](t.me/subwayserver_bot) and start chatting!

# How it works
Users can either input their current location or the name of a subway station. The app then attempts to narrow down the possible set of subway stations based on this input and queries a subway times API to get the relevant train data for each possible station. An LLM then formats and returns the output. 

Users can also set their home location for quick access. They can also repeat their previous query.

I built this because I wanted to play with some of Anthropic's latest APIs. 

