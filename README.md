# 🚀 RAGitify - Enterprise RAG Platform

[![Python](https://img.shields.io/badge/Python-3.10.5-blue.svg)](https://www.python.org/downloads/release/python-3105/)
[![Django](https://img.shields.io/badge/Django-5.2-green.svg)](https://docs.djangoproject.com/en/5.2/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**RAGitify** is a powerful Enterprise-grade Retrieval-Augmented Generation (RAG) platform that enables organizations to build intelligent document search and question-answering systems. Built with Django and powered by advanced AI models, RAGitify transforms your documents into searchable knowledge bases with conversational AI capabilities.

## ✨ Key Features

###  **Advanced Document Processing**
- **Multi-format Support**: PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, HTML, XML, JSON, YAML, images, videos, and more
- **Intelligent Text Extraction**: OCR for images, speech-to-text for audio/video files
- **Large File Handling**: Streaming processing for files with 50K+ rows
- **Automatic Chunking**: Smart text segmentation with configurable strategies

###  **AI-Powered Search & Q&A**
- **Hybrid Search**: Combines BM25 lexical search with vector similarity search
- **Multi-Modal RAG**: Document, web, and conversational modes
- **Context-Aware Responses**: Maintains conversation history and context
- **Tool Integration**: Support for custom function calling and tool usage

### 🏢 **Enterprise Features**
- **Multi-Tenant Architecture**: Isolated data and configurations per organization
- **User Management**: Role-based access control and authentication
- **Vector Store Management**: Multiple vector stores per tenant
- **Document Access Control**: Granular permissions and sharing

### 🔧 **Advanced Capabilities**
- **Auto-Generated Thread Titles**: Intelligent conversation naming
- **Document Enrichment**: Automatic summarization and keyword extraction
- **Alert System**: Keyword-based document monitoring
- **API-First Design**: RESTful APIs with Swagger documentation
- **Real-time Processing**: Asynchronous document ingestion

### 🧭 Assistant Prompt Guidance
- Default assistants treat the tenant user as a read-only attribute. Prompts and responses must not attempt to change or imply modifications to the tenant user assignment.

## 🏗️ System Architecture

### Complete RAGitify Workflow

```mermaid
graph TB
    %% User Actions
    A["👤 User Login"] --> B["🔑 Token Authentication"]
    B --> C["📁 Create Vector Store"]
    
    %% Document Processing Pipeline
    C --> D["📄 Upload Document"]
    D --> E["🔍 File Type Detection"]
    E --> F["📝 Text Extraction"]
    F --> G["✂️ Smart Chunking"]
    G --> H["🧠 Generate Embeddings"]
    H --> I["💾 Store in Qdrant"]
    I --> J["✅ Document Status: Completed"]
    
    %% QA & Conversation Flow
    J --> K["🤖 Create Thread"]
    K --> L["🤖 Create Assistant"]
    L --> M["❓ User Question"]
    
    %% Mode Selection & Processing
    M --> N{"🎯 Mode Selection"}
    N -->|Document| O["📚 Document RAG"]
    N -->|Web| P["🌐 Web Search"]
    N -->|Normal| Q["💭 Direct Chat"]
    
    %% Document RAG Flow
    O --> R["🔍 Hybrid Search"]
    R --> S["📋 Context Retrieval"]
    S --> T["🧠 LLM Processing"]
    
    %% Web Search Flow
    P --> U["🌍 Web Search API"]
    U --> V["📰 Web Results"]
    V --> T
    
    %% Direct Chat Flow
    Q --> T
    
    %% Response Generation
    T --> W{"🛠️ Tools Required?"}
    W -->|Yes| X["🔧 Tool Execution"]
    W -->|No| Y["📝 Generate Response"]
    X --> Y
    
    %% Final Response
    Y --> Z["💬 Return Answer"]
    Z --> AA["📝 Save to Thread"]
    AA --> BB["🔄 Continue Conversation"]
    
    %% Styling
    classDef userAction fill:#e1f5fe
    classDef processing fill:#f3e5f5
    classDef storage fill:#e8f5e8
    classDef ai fill:#fff3e0
    classDef decision fill:#fce4ec
    
    class A,B userAction
    class D,E,F,G,H processing
    class I,J storage
    class R,S,T,U,V,W,X,Y ai
    class N,O,P,Q decision
```

### Document Processing Pipeline

```mermaid
graph LR
    A["📄 Document Upload"] --> B["🔍 File Type Detection"]
    B --> C{"📁 File Type"}
    C -->|PDF/DOCX| D["📝 Text Extraction"]
    C -->|Images| E["🖼️ OCR Processing"]
    C -->|Audio/Video| F["🎵 Speech-to-Text"]
    C -->|Excel/CSV| G["📊 Structured Data"]
    
    D --> H["✂️ Intelligent Chunking"]
    E --> H
    F --> H
    G --> H
    
    H --> I["🧠 Embedding Generation"]
    I --> J["💾 Qdrant Storage"]
    J --> K["✅ Processing Complete"]
    
    classDef upload fill:#e3f2fd
    classDef processing fill:#f3e5f5
    classDef storage fill:#e8f5e8
    
    class A,B upload
    class C,D,E,F,G,H,I processing
    class J,K storage
```

### Query Processing Pipeline

```mermaid
graph TB
    A["❓ User Question"] --> B{"🎯 Mode Selection"}
    B -->|Document| C["📚 Document Mode"]
    B -->|Web| D["🌐 Web Mode"]
    B -->|Normal| E["💭 Normal Mode"]
    
    C --> F["🔍 Vector Search"]
    C --> G["📝 BM25 Search"]
    F --> H["🔄 Hybrid Search"]
    G --> H
    
    D --> I["🌍 Web Search API"]
    I --> J["📰 Web Results"]
    
    E --> K["💭 Direct Processing"]
    
    H --> L["📋 Context Assembly"]
    J --> L
    K --> L
    
    L --> M["🧠 LLM Processing"]
    M --> N{"🛠️ Tools Required?"}
    N -->|Yes| O["🔧 Tool Execution"]
    N -->|No| P["📝 Generate Response"]
    O --> P
    
    P --> Q["💬 Final Response"]
    
    classDef input fill:#e1f5fe
    classDef processing fill:#f3e5f5
    classDef decision fill:#fce4ec
    classDef output fill:#e8f5e8
    
    class A,B,C,D,E,F,G,H,I,J,K,L,M processing
    class N decision
    class O,P,Q output
```

### Data Flow Architecture

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant API as 🌐 Django API
    participant DB as 🗄️ Database
    participant AI as 🤖 AI Service
    participant QD as 🔍 Qdrant
    
    %% Document Upload Flow
    U->>API: 📄 Upload Document
    API->>DB: 💾 Create Document Record
    API->>API: 🔍 Extract Text
    API->>API: ✂️ Chunk Text
    API->>AI: 🧠 Generate Embeddings
    AI-->>API: 📊 Embeddings
    API->>QD: 💾 Store Vectors
    API->>DB: ✅ Update Status
    API-->>U: ✅ Upload Complete
    
    %% Query Processing Flow
    U->>API: ❓ Ask Question
    API->>DB: 📋 Get Thread History
    API->>QD: 🔍 Search Vectors
    QD-->>API: 📄 Relevant Docs
    API->>AI: 🧠 Process Query
    AI-->>API: 📝 Generated Response
    API->>DB: 💾 Save Message
    API-->>U: 💬 Return Answer
```

#### **⚡ Async Processing**
```mermaid
graph LR
    A["📄 Document Upload"] --> B["🔄 Async Background Task"]
    B --> C["📝 Text Extraction"]
    B --> D["✂️ Chunking"]
    B --> E["🧠 Embedding Generation"]
    B --> F["💾 Vector Storage"]
    C --> G["✅ Status Updates"]
    D --> G
    E --> G
    F --> G
    
    classDef async fill:#fff3e0
    classDef processing fill:#f3e5f5
    classDef status fill:#e8f5e8
    
    class A,B async
    class C,D,E,F processing
    class G status
```

#### **🚀 Caching Strategy**
```mermaid
graph TB
    A["🧠 LLM Responses"] --> B["💾 Response Cache"]
    C["🔤 Embeddings"] --> D["🔄 Embedding Cache"]
    E["📝 Search Results"] --> F["📊 Search Cache"]
    
    B --> G["⚡ Faster Responses"]
    D --> H["🚀 Reduced API Calls"]
    F --> I["📈 Better Performance"]
    
    classDef cache fill:#e8f5e8
    classDef benefit fill:#e1f5fe
    
    class A,C,E,B,D,F cache
    class G,H,I benefit
```

## ️ Quick Start

### Prerequisites

- **Python 3.10.5+**
- **Pip 22.0.4+**
- **PostgreSQL** (for production) or **SQLite** (for development)
- **Qdrant** vector database
- **OpenAI API Key** or **Ollama** setup

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/RAGitify.git
   cd RAGitify
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # Linux/macOS
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set up environment variables**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit .env with your configuration
   OPENAI_API_KEY=sk-your-openai-api-key-here
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   ```

6. **Configure settings**
   ```python
   # In project/settings.py
   OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
   QDRANT_HOST = 'localhost'
   QDRANT_PORT = '6333'
   ```

7. **Run database migrations**
   ```bash
   python manage.py migrate
   ```

8. **Create superuser**
   ```bash
   python manage.py createsuperuser
   ```

9. **Start the development server**
   ```bash
   python manage.py runserver
   ```

10. **Access the application**
    - **API**: http://127.0.0.1:8000/
    - **Admin**: http://127.0.0.1:8000/admin/
    - **API Doc**: http://127.0.0.1:8000/rag/api-documentation/

## ⚙️ Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional
QDRANT_HOST=localhost
QDRANT_PORT=6333
DEFAULT_LLM_PROVIDER=OpenAI
DEFAULT_OPENAI_MODEL=gpt-4.1
DEBUG=True
```

### Settings Configuration

```python
# project/settings.py

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Vector Store
QDRANT_HOST = 'localhost'
QDRANT_PORT = '6333'
QDRANT_VECTOR_DIMENSION = 1536

# LLM Configuration
DEFAULT_LLM_PROVIDER = "OpenAI"
DEFAULT_OPENAI_MODEL = "gpt-4.1"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"

# File Processing
LARGE_FILE_THRESHOLD = 50000
ROWS_PER_CHUNK = 5000
CHUNK_STRATEGY = "auto"
```

## 🏢 Production Deployment

### Docker Setup

```dockerfile
# Dockerfile
FROM python:3.10.5-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
```

### Environment Configuration

```bash
# Production environment variables
DEBUG=False
ALLOWED_HOSTS=your-domain.com
DATABASE_URL=postgresql://user:password@host:port/dbname
OPENAI_API_KEY=sk-your-production-key
QDRANT_HOST=your-qdrant-host
QDRANT_PORT=6333
```

### Database Setup

```bash
# PostgreSQL setup
python manage.py migrate
python manage.py collectstatic
python manage.py createsuperuser
```

## 🔒 Security Features

- **Token-based Authentication**: Secure API access
- **Multi-tenant Isolation**: Data separation between organizations
- **Input Validation**: Comprehensive request validation
- **Rate Limiting**: Protection against abuse
- **CORS Configuration**: Secure cross-origin requests

## 📊 Monitoring & Logging

- **Comprehensive Logging**: Detailed operation logs
- **Performance Monitoring**: Memory usage tracking
- **Error Handling**: Graceful error recovery
- **Tracing**: Request tracing for debugging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ️ Support

- **Documentation**: [Wiki](https://github.com/your-username/RAGitify/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-username/RAGitify/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/RAGitify/discussions)

## 🙏 Acknowledgments

- **OpenAI** for GPT models
- **Qdrant** for vector database
- **Django** for the web framework
- **LangChain** for LLM integration
- **spaCy** for NLP processing

---

**Made with ❤️ by the RAGitify Team**

