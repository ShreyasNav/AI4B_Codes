import os
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import pandas as pd
import time
import trafilatura

class SearchTermDiscovery:
    def __init__(self, huggingface_token):
        self.embeddings = HuggingFaceEmbeddings()
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
        self.llm = HuggingFaceHub(repo_id="google/flan-t5-large")
    
    def get_urls_from_google(self, language, num_results=30):
        """Get URLs from Google search using different queries."""
        search_queries = [
            f"{language} stories audio",
            f"{language} audiobooks",
            f"{language} literature audio",
            f"{language} poetry recitation"
        ]
        
        all_urls = set()
        for query in search_queries:
            try:
                # Get URLs for each query
                urls = search(query, num_results=num_results//len(search_queries), timeout = 30)
                all_urls.update(urls)
                time.sleep(2)  # Be nice to Google
            except Exception as e:
                print(f"Error in Google search for query '{query}': {e}")
                
        return list(all_urls)
    
    def extract_text_from_url(self, url):
        """Extract text content from a URL using trafilatura, with better cleaning."""
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded, 
                                        include_comments=False, 
                                        include_tables=False)
                
                if text:
                    # Clean the text
                    cleaned_text = self.clean_text(text)
                    return cleaned_text if cleaned_text else ""
            return ""
        except Exception as e:
            print(f"Error extracting text from {url}: {e}")
            return ""

    def clean_text(self, text):
        """Clean extracted text to remove boilerplate and keep useful content."""
        # Split into lines and clean
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip useless lines
            if any([
                len(line) < 20,  # Too short
                'download' in line.lower(),
                'zip' in line.lower(),
                'mp3' in line.lower(),
                'mp4' in line.lower(),
                'audio' in line.lower() and 'mb' in line.lower(),
                'language code' in line.lower(),
                'iso' in line.lower(),
                'sample of' in line.lower(),
                line.startswith('©'),
                line.startswith('http'),
                'privacy policy' in line.lower(),
                'terms of use' in line.lower(),
                'cookie' in line.lower()
            ]):
                continue
                
            cleaned_lines.append(line)
        
        # Join non-empty lines
        cleaned_text = ' '.join(line for line in cleaned_lines if line)
        return cleaned_text
    
    def gather_context(self, language, num_results=30):
        """Gather context from Google search results."""
        print(f"Gathering URLs for {language}...")
        urls = self.get_urls_from_google(language, num_results)
        
        print(f"\nFound {len(urls)} URLs. Starting text extraction...")
        documents = []
        for i, url in enumerate(urls, 1):
            text = self.extract_text_from_url(url)
            if text:
                print(f"✓ Document {i}: Got {len(text)} characters from {url}")
                documents.append({
                    "text": text,
                    "source": url
                })
            else:
                print(f"✗ Document {i}: Failed to extract text from {url}")
        
        print(f"\nSuccessfully extracted text from {len(documents)} documents")
        return documents

    def process_documents(self, documents):
        """Process documents with better filtering."""
        if not documents:
            raise ValueError("No documents to process!")
            
        print(f"\nProcessing {len(documents)} documents...")
        
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Prepare texts for vectorization with filtering
        texts = []
        for doc in documents:
                
            chunks = splitter.split_text(doc["text"])
            # Filter chunks
            filtered_chunks = []
            for chunk in chunks:
                # Skip chunks that are likely navigation or boilerplate
                if any([
                    'click here' in chunk.lower(),
                    'copyright' in chunk.lower(),
                    'all rights reserved' in chunk.lower(),
                    'privacy policy' in chunk.lower(),
                    'terms of service' in chunk.lower()
                ]):
                    continue
                filtered_chunks.append(chunk)
                
            texts.extend([{
                "text": chunk,
                "source": doc["source"]
            } for chunk in filtered_chunks])
        
        print(f"\nCreated {len(texts)} meaningful text chunks...")
        
        if not texts:
            raise ValueError("No meaningful text chunks found after filtering!")
        
        # Create vector store
        vectorstore = FAISS.from_texts(
            [t["text"] for t in texts],
            self.embeddings,
            metadatas=[{"source": t["source"]} for t in texts]
        )
        
        return vectorstore

    def generate_search_terms(self, language, vectorstore):
        """Generate YouTube search terms using RAG."""
        print("\nGenerating search terms...")
        
        # First, let's see what we're working with
        sample_docs = vectorstore.similarity_search("", k=3)
        print("\nSample of available context:")
        for i, doc in enumerate(sample_docs, 1):
            print(f"\nDocument {i} preview: {doc.page_content[:200]}...")

        template = f"""
        Based on the provided context about {language} language content, generate '10' specific
        YouTube search terms that might lead to audio content.

        Make sure to:
        - Include actual program, book, audiobook or audio series names if mentioned in the context.
        - Use keywords specific Dogri content.
        - Provide terms that people are likely to use to search for Dogri audio content.

        Format each term on a new line as follows:
        "{language} term" - Description

        Make sure each term is actually in or about {language} language content.
        """
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True  # Add this to see what context was used
        )
        
        result = qa_chain.invoke(template)
        print("\nContext used for generation:", result["source_documents"])
        
        # Split the response into lines and filter out empty ones
        terms = [term.strip() for term in result["result"].split('\n') if term.strip()]
        print(terms)
        
        return terms
        
    def validate_terms(self, search_terms):
        """Validate search terms by checking YouTube result count."""
        results = []
        for term in search_terms:
            try:
                print(term)
                url = "https://www.youtube.com/results?search_query="+f"{term}"
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                video_count = len(soup.find_all('div', {'class': 'style-scope ytd-video-renderer'}))
                
                print(f"Found {video_count} results for term: {term}")
                
                results.append({
                    'search_term': term,
                    'result_count': video_count,
                    'url': url
                })
                time.sleep(2)  # Be nice to YouTube
                
            except Exception as e:
                print(f"Error validating term '{term}': {e}")
                continue
        
        if not results:
            print("Warning: No valid results found!")
            return pd.DataFrame(columns=['search_term', 'result_count', 'url'])
            
        return pd.DataFrame(results).sort_values('result_count', ascending=False)

def main():
    pipeline = SearchTermDiscovery(huggingface_token="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")
    
    language = "Dogri"
    print(f"\nStarting search term discovery for {language}...")
    
    # Gather and process context
    documents = pipeline.gather_context(language)
    if not documents:
        print("Error: No documents found!")
        return
        
    vectorstore = pipeline.process_documents(documents)
    
    # Generate and validate search terms
    search_terms = pipeline.generate_search_terms(language, vectorstore)
    if not search_terms:
        print("Error: No search terms generated!")
        return
        
    print("\nGenerated terms:", search_terms)
    
    ranked_terms = pipeline.validate_terms(search_terms)
    
    print("\nTop recommended YouTube search terms:")
    print(ranked_terms)
    
    # Save results
    ranked_terms.to_csv(f"{language}_search_terms.csv", index=False)

if __name__ == "__main__":
    main()