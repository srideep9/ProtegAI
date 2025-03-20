from flask import Flask, request, jsonify, session
from flask_cors import CORS
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import threading
from langchain.schema.runnable import RunnableLambda
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from pyngrok import ngrok


# Initialize models at the global level when app starts
model_name = "sdornala/bert-uncertainty-classifier_teacher"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
PINECONE_API_KEY = "PINECONE_API_KEY"
embed_model = HuggingFaceEmbedding()
pc = PineconeGRPC(api_key=PINECONE_API_KEY)
index_name = "llama-integration-example"
pinecone_index = pc.Index(index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index, query_engine= None, None

# Initialize LLM pipeline
HF_TOKEN = "HUGGING_FACE_TOKEN"
from langchain_community.llms import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HF_TOKEN,
    task="text-generation",
    max_length=1024
)
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
llm2 = HuggingFaceInferenceAPI(model_name = 'meta-llama/Llama-3.2-3B-Instruct', token = HF_TOKEN)
def initializeRAG(text, llm):
  documents = [Document(text=text)]
  index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)
  query_engine = index.as_query_engine(llm=llm)
  return index, query_engine

# Global variables for state management
is_first_message = True
current_question = ""
topic = ""
unclear_count = 0

# Enhanced prompt templates
first_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""<s>[INST] <<SYS>>
    You are an expert educational tutor using the Socratic method to help students who said "{topic}".
    
    Your goal is to stimulate critical thinking through questions rather than direct instruction.
    Ask questions that encourage the student to discover knowledge through reflection.
    <</SYS>>
    
    Create an engaging first question about "{topic}" that questions their fundamental understanding.
    Respond with only one clear, thought-provoking question. Respond with only the question itself, not any explanation of your approach.[/INST]"""
)

accuracy_prompt = PromptTemplate(
    input_variables=["question", "answer", "topic"],
    template="""<s>[INST] <<SYS>>
    You are a validation system analyzing student responses in a Socratic teaching session about {topic}.
    <</SYS>>

    Evaluate if the response: "{answer}" 
    Is a factually correct and valid answer to the question: "{question}"
    
    Only respond with:
    - "Y" if the answer is factually correct or represents valid reasoning
    - "N" if the answer contains factual errors or flawed reasoning
    
    Response (Y/N only): [/INST]"""
)

uncertain_prompt = PromptTemplate(
    input_variables=["input", "question", "topic"],
    template="""<s>[INST] <<SYS>>
    You are an expert tutor guiding a student through understanding {topic} using the Socratic method.
    The student has responded with uncertainty to your previous question.
    <</SYS>>

    Previous question: "{question}"
    Student's uncertain response: "{input}"
    
    Create a follow-up question that:
    1. Acknowledges their uncertainty
    2. Guides them toward greater specificity
    3. Breaks down the concept into more manageable parts
    
    Respond with only one clear, focused question. Respond with only the question itself, not any explanation of your approach.[/INST]"""
)

certain_prompt = PromptTemplate(
    input_variables=["topic", "previous_question", "previous_answer"],
    template="""<s>[INST] <<SYS>>
    You are an expert tutor using the Socratic method who is helping a user that said "{topic}".
    The student has demonstrated understanding in their previous response.
    <</SYS>>

    Previous question: "{previous_question}"
    Student's answer: "{previous_answer}"
    
    Create a new question that:
    1. Builds upon their demonstrated understanding
    2. Encourages deeper exploration to help the user who said "{topic}"
    3. Introduces a new aspect or challenges their thinking further
    
    Respond with only one clear, thought-provoking question. Respond with only the question itself, not any explanation of your approach.[/INST]"""
)

incorrect_prompt = PromptTemplate(
    input_variables=["topic", "question", "answer"],
    template="""<s>[INST] <<SYS>>
    You are an expert tutor using the Socratic method who is helping a user that said "{topic}".
    The student has provided an incorrect response.
    <</SYS>>

    Question: "{question}"
    Student's incorrect answer: "{answer}"
    
    Create a response that:
    1. Does NOT directly state they are wrong
    2. Asks a simpler, more guided question that helps them reconsider their thinking
    3. Provides a subtle hint to guide them toward the correct understanding
    
    Respond with only one clear, supportive question. Respond with only the question itself, not any explanation of your approach.[/INST]"""
)

# RAG enabled incorrect prompt for future use
rag_incorrect_prompt = PromptTemplate(
    input_variables=["topic", "question", "answer", "relevant_content"],
    template="""<s>[INST] <<SYS>>
    You are an expert tutor using the Socratic method to teach {topic}.
    The student has provided an incorrect response.
    You have access to relevant educational content to help guide the student.
    <</SYS>>

    Question: "{question}"
    Student's incorrect answer: "{answer}"
    
    Relevant content to guide your response:
    {relevant_content}
    
    Create a response that:
    1. Does NOT directly state they are wrong
    2. Uses concepts from the relevant content to create a more guided question
    3. Provides a subtle hint based on the educational content
    
    Respond with only one clear, supportive question. [/INST]"""
)

# Initialize chains
first_chain = LLMChain(llm=llm, prompt=first_prompt)
accuracy_chain = LLMChain(llm=llm, prompt=accuracy_prompt)
uncertain_chain = LLMChain(llm=llm, prompt=uncertain_prompt)
certain_chain = LLMChain(llm=llm, prompt=certain_prompt)
incorrect_chain = LLMChain(llm=llm, prompt=incorrect_prompt)
rag_incorrect_chain = LLMChain(llm=llm, prompt=rag_incorrect_prompt)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Store the document and conversation history in memory (in production, use a database)
stored_document = ""
conversation_history = []


@app.route('/api/upload_document', methods=['POST'])
def upload_document():
    global stored_document
    global conversation_history
    global index
    global query_engine
    data = request.json
    document = data.get('document', '')
    stored_document = document
    conversation_history = []
    index, query_engine = initializeRAG(document, llm2)
    return jsonify({
        'success': True,
        'message': f"Document uploaded successfully. Length: {len(document)} characters.",
        'document_preview': document[:100] + "..." if len(document) > 100 else document
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    global stored_document
    global conversation_history
    global is_first_message
    global unclear_count
    global topic
    global current_question

    data = request.json
    user_message = data.get('message', '')

    if not stored_document:
        return jsonify({
            'response': "No document has been uploaded yet. Please upload a document first.",
            'conversation_history': []
        })

    # Add user message to conversation history
    conversation_history.append({
        'role': 'user',
        'content': user_message
    })

    # First interaction - user specifies topic
    if is_first_message:
        topic = user_message
        response_obj = first_chain({"topic": user_message})
        response = response_obj['text'].strip()
        current_question = response
        flow_status = "first_question"
        is_first_message = False
    else:

        is_uncertain = classify_uncertainty(user_message.lower()) == 0
        
        # Uncertainty handling with counter
        if is_uncertain:
            unclear_count += 1
            if unclear_count >= 3:
                rag = query_engine.query("Find the section of the text that has the answer to the previous question: "+current_question)
                response_obj = first_chain({"topic": topic})
                response = "I notice you're having trouble with this concept. Let's try a different approach. " + response_obj['text'].strip()
                current_question = response_obj['text'].strip()
                unclear_count = 0
                flow_status = "reset_after_repeated_uncertainty"
                response = "Here is some text from your document you could look at:\n\n"+rag.response+"\n"+response
            else:
                # Ask clarifying question
                response_obj = uncertain_chain({
                    "input": user_message,
                    "question": current_question,
                    "topic": topic
                })
                response = response_obj['text'].strip()
                current_question = response
                flow_status = "uncertain_response"
        
        else:
            validity_check = accuracy_chain({
                "question": current_question,
                "answer": user_message,
                "topic": topic
            })
            is_valid = validity_check['text'].strip()

            if "N" not in is_valid:
                response_obj = certain_chain({
                    "topic": topic, 
                    "previous_question": current_question,
                    "previous_answer": user_message
                })
                response = "Well reasoned. " + response_obj['text'].strip()
                current_question = response_obj['text'].strip()
                unclear_count = 0
                flow_status = "valid_detailed_response"
            else:
                rag = query_engine.query("Find the section of the text that has the answer to the following question: "+current_question)
                response_obj = incorrect_chain({
                    "topic": topic,
                    "question": current_question,
                    "answer": user_message
                })
                response = response_obj['text'].strip()
                current_question = response
                response = "Here is some text from your document you could look at:\n\n"+rag.response+"\n"+response
                flow_status = "invalid_answer"

    # Add bot response to conversation history
    conversation_history.append({
        'role': 'assistant',
        'content': response
    })

    return jsonify({
        'flow_status': flow_status,
        'response': response,
        'conversation_history': conversation_history,
    })
@app.route('/api/clear', methods=['POST'])
def cleardoc():
    global is_first_message
    global unclear_count
    global stored_document
    global conversation_history
    global index
    global query_engine
    is_first_message = True
    unclear_count = 0
    stored_document = ""
    conversation_history = []
    index = None
    query_engine = None
    current_question = ""
    topic = ""
    return jsonify({
        "Null" : "Null"
    })
def classify_uncertainty(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # Get predictions
    outputs = model(**inputs)
    probabilities = outputs.logits.softmax(dim=1)

    # Get raw class prediction (0 or 1)
    predicted_class = probabilities.argmax().item()
    confidence = probabilities.max().item()

    return predicted_class  # Returns 0 (Uncertain) or 1 (Certain)

if __name__ == "__main__":
    threading.Thread(target=app.run, kwargs={'host':'0.0.0.0','port':5000}).start()
    public_url = ngrok.connect(5000).public_url
    print(f"Your Flask app is available at: {public_url}")
