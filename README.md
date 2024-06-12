# AI Powered Student Assistant

An important service we need to provide for our student body at UWindsor is easy access to information about our programs. Many fantastic efforts have been made towards this goal most notably the creation of a student wiki. This project represents an even more streamlined, adaptive, and time efficient information source: A digital assistant powered by ChatGPT and RAG (retrieval augmented generation). 

At a high level, we take the student’s question, finding important context in our database pertinent to the question, and feeding both the context and the query to the LLM to get a response. 

For example: “What is the course content for COMP-4540?”

Would cause our program to fetch documents such as the course description. This would all get fed to ChatGPT in a template such as:

```
System prompt: You are a helpful AI assistant.

Context: <Fetched Document>

Query: What is the course content for COMP-4540?
```

After the response is generated, we simply relay the response to our interface of choice, such as a dialog on the website or through the discord bot in a special channel!

This allows for personalized assistance to any student query, enhancing availability of information to the students and thereby driving a more intuitive, accessible and straightforward student experience.

**IMPORTANT DISCLAIMER: PLEASE DOUBLE CHECK ANY IMPORTANT INFORMATION USING TRUSTWORTHY SOURCES SUCH AS https://uwindsorcss.github.io/wiki/. THIS SERVICE IS BASED ON GENERATIVE AI, AN INFORMATION SOURCE WHICH IS NOT ALWAYS 100% ACCURATE**

This project is still in development. Contributions are welcome. Please direct any inquiries to `cj_star` on Discord.