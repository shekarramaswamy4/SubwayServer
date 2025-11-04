# SubwayServer

<img width="768" height="528" alt="ChatGPT Image Nov 4, 2025, 04_56_14 PM" src="https://github.com/user-attachments/assets/b0ac4a7e-b368-49b2-a7fd-fcd84fbc178e" />

SubwayServer is a telegram bot that lets you query the arrival/departure times for subways at specific stations.

<img width="658" height="359" alt="Screenshot 2025-11-04 at 5 43 42 PM" src="https://github.com/user-attachments/assets/5b623078-c08f-44b0-8f6b-910280ec1aa0" />

# How to use

Go to [t.me/subwayserver_bot](https://t.me/subwayserver_bot) and start chatting!

# How it works
Users can either input their current location or the name of a subway station. The app then attempts to narrow down the possible set of subway stations based on this input and queries a subway times API to get the relevant train data for each possible station. An LLM then formats and returns the output. 

Users can also set their home location for quick access. They can also repeat their previous query.

I built this because I wanted to play with some of Anthropic's latest APIs. This was a fun exercise in prompt engineering and formatting data in a user-friendly way. Feel free to contribute or DM me if you want any features to be added.

This was almost entirely vibe-coded, so please treat it accordingly :)

