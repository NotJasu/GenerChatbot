from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import chromadb
import os
import time

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import CHROMA_SETTINGS

@app.route('/api/chat', methods=['POST'])
def chat_with_bot():
    try:
        data = request.get_json()
        query = data.get('query')

        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS, path=persist_directory)
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings,
                    client_settings=CHROMA_SETTINGS, client=chroma_client)
        retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
        callbacks = [StreamingStdOutCallbackHandler()] if not data.get('mute_stream') else []

        if data.get('model_type') == "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch,
                            callbacks=callbacks, verbose=False)
        elif data.get('model_type') == "GPT4All":
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch,
                          callbacks=callbacks, verbose=False)
        else:
            return jsonify({'error': f"Model type {data.get('model_type')} is not supported. Please choose one of the following: LlamaCpp, GPT4All"}), 400

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                        return_source_documents=not data.get('hide_source'))

        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if data.get('hide_source') else res['source_documents']
        end = time.time()

        response = {
            'query': query,
            'answer': answer,
            'source_documents': [document.page_content for document in docs],
            'response_time_seconds': round(end - start, 2)
        }

        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)