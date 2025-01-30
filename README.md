# Shakespeare Text Generator with SmolLM2-135M

A microservices-based application that generates Shakespeare-style text using the SmolLM2-135M language model. The project uses FastAPI for the backend, Streamlit for the frontend, and Docker for containerization.

## Architecture

The application consists of two services:
- **Frontend (Streamlit)**: Provides the user interface for text generation at `http://localhost:8501`
- **Backend (FastAPI)**: Runs the SmolLM2-135M model and displays results at `http://localhost:8000`

## Prerequisites

- Docker and Docker Compose
- Python 3.9 or later (if running locally)
- The SmolLM2-135M model checkpoint (`step_5050.pt`)

## Project Structure
```
text-generator-docker/
├── model-server/
│   ├── app.py           # FastAPI server
│   ├── model.py         # SmolLM2 model implementation
│   ├── config.py        # Model configuration
│   ├── requirements.txt # Python dependencies
│   ├── Dockerfile      
│   └── checkpoints/     # Model weights directory
│       └── step_5050.pt # Model checkpoint
├── client/
│   ├── app.py           # Streamlit interface
│   ├── requirements.txt # Python dependencies
│   └── Dockerfile
└── docker-compose.yml   # Service orchestration
```

## Setup and Running

1. Clone the repository:
```bash
git clone [your-repo-url]
cd text-generator-docker
```

2. Place the model checkpoint:
```bash
# Copy step_5050.pt to model-server/checkpoints/
cp path/to/step_5050.pt model-server/checkpoints/
```

3. Build and run with Docker Compose:
```bash
docker-compose up --build
```

4. Access the applications:
- Text Generation UI: http://localhost:8501
- Generated Results: http://localhost:8000

## Usage

1. Open http://localhost:8501 in your browser
2. Enter a prompt or select an example prompt
3. Adjust generation parameters:
   - Max Tokens (10-200)
   - Temperature (0.1-2.0)
4. Click "Generate Text"
5. View results at http://localhost:8000

## Features

- Real-time text generation
- Adjustable generation parameters
- Auto-refreshing results page
- Example Shakespeare prompts
- Containerized deployment
- Health monitoring
- Separate input and output interfaces

## Technical Details

### Model Server
- FastAPI backend
- SmolLM2-135M model
- CPU-optimized PyTorch
- CORS middleware
- Health checks

### Client
- Streamlit interface
- Real-time communication with backend
- Responsive design
- Error handling

### Docker Configuration
- Multi-container setup
- Volume mounts for model files
- Health check integration
- Container orchestration

## Dependencies

### Model Server
- PyTorch (CPU version)
- Transformers
- FastAPI
- Uvicorn
- NumPy

### Client
- Streamlit
- Requests
- Python-dotenv

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SmolLM2-135M model architecture
- Shakespeare's works for training data
- FastAPI and Streamlit communities