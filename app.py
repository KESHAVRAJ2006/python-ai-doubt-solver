import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()

# Create Flask application
app = Flask(__name__)


# Get API key from .env
api_key = os.getenv("GOOGLE_API_KEY")


# Check API key
if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY is missing. Please add it to your .env file."
    )


# Create Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=api_key
)


# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful Python Doubt Solver.

        Your job is to help students understand Python programming.

        Rules:
        1. Explain concepts in simple language.
        2. Give examples whenever useful.
        3. If code is requested, provide clean Python code.
        4. Explain the code briefly.
        5. Be friendly and helpful.
        """
    ),
    ("human", "{question}")
])


# Connect prompt and model
chain = prompt | model


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Chat API
@app.route("/ask", methods=["POST"])
def ask():

    try:
        data = request.get_json()

        question = data.get("question", "").strip()

        # Check empty question
        if not question:
            return jsonify({
                "error": "Please enter a question."
            }), 400

        # Generate response using LangChain
        response = chain.invoke({
            "question": question
        })

        return jsonify({
            "answer": response.content
        })

    except Exception as e:

        print("Error:", e)

        return jsonify({
            "error": "Sorry, something went wrong. Please try again."
        }), 500


# Run application
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )