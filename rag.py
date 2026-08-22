import preprocesser
import retriever
import generator
from sentence_transformers import CrossEncoder
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",device="cpu")
embedding_model="BAAI/bge-small-en-v1.5"
pdf_path= "computer_vision.pdf"
chunks_file = "chunks.json"
chunk_size=1500  
chunk_overlap=300
candidate_k=30
top_k=5
chunks,collection=preprocesser.preprocess(chunk_size,chunk_overlap,embedding_model,chunks_file,pdf_path)
query_text1="What is the difference between image classification, object detection, and semantic segmentation?"	
query_text2="Explain how convolution operates on an image and why it's useful for feature extraction."	
query_text3="What is the purpose of the Harris corner detector, and what mathematical property does it rely on?"
query_text4="Compare SIFT and SURF as feature descriptors — what are their main tradeoffs?"	
query_text5="How does the RANSAC algorithm work, and in what computer vision tasks is it commonly applied?"	
query_text6="What is optical flow, and what assumption underlies the Lucas-Kanade method?"	
query_text7="Describe the architecture and purpose of a Convolutional Neural Network (CNN) for image recognition."
query_text8="What problem does non-maximum suppression solve in object detection pipelines?"	
query_text9="Explain the concept of image pyramids and their use in multi-scale feature detection."	
query_text10="How do Generative Adversarial Networks (GANs) apply to computer vision tasks like image synthesis?"
query_list=[query_text1,query_text2,query_text3,query_text4,query_text5,query_text6,query_text7,query_text8,query_text9,query_text10]

for query in query_list:
    results=retriever.retrieve(reranker,chunks,query, collection,candidate_k,top_k)
    context = "\n\n".join(
            f"[{doc_id}]: {text}" for doc_id, text, _ in results
    )
    prompt = f"""Answer the question using the context below. 
    If the context doesn't contain enough information to answer, say "I don't have enough information to answer this."
    Cite the chunk ID(s) you used in your answer, like [chunk_2].
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:"""
    print("-"*20)
    print(query)
    print("\nLLM answer")
    response=generator.generate_answer(query)
    print(response)
    print("\nRAG answer")
    response=generator.generate_answer(prompt)    
    print(response)
    print("-"*20)