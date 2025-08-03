# Deep Image Refinement using GANs

A Flask web application that uses Super-Resolution Generative Adversarial Networks (SRGAN) and Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN) to enhance image quality and resolution.

## Features

- 🎨 **Dual GAN Processing**: Uses both SRGAN and ESRGAN models for image enhancement
- 📱 **Web Interface**: User-friendly web interface for image upload and processing
- 🚀 **Real-time Processing**: Instant image enhancement with AI models
- 📊 **Side-by-side Comparison**: View original and enhanced images together

## Quick Deployment

### Option 1: Deploy on Render (Recommended - Easiest)

1. **Fork/Clone this repository** to your GitHub account

2. **Sign up for Render** at [render.com](https://render.com)

3. **Create a new Web Service**:
   - Connect your GitHub repository
   - Set the following:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app`
     - **Environment**: Python 3

4. **Deploy!** Render will automatically build and deploy your app

### Option 2: Deploy on Railway

1. **Sign up for Railway** at [railway.app](https://railway.app)

2. **Connect your GitHub repository**

3. **Railway will automatically detect** it's a Python app and deploy it

### Option 3: Deploy on Heroku

1. **Sign up for Heroku** at [heroku.com](https://heroku.com)

2. **Install Heroku CLI** and login

3. **Create a new Heroku app**:
   ```bash
   heroku create your-app-name
   ```

4. **Deploy**:
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push heroku main
   ```

## Local Development

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Deep-Image-Refinement-using-GANs
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and go to `http://localhost:5000`

## Project Structure

```
Deep-Image-Refinement-using-GANs/
├── app.py                 # Main Flask application
├── srgan.py              # SRGAN model implementation
├── esrgan.py             # ESRGAN model implementation
├── generator.pt          # ESRGAN model weights
├── last_G.pt            # SRGAN model weights
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html       # Web interface template
├── static/
│   ├── uploads/         # Uploaded images
│   └── outputs/         # Processed images
└── README.md            # This file
```

## How It Works

1. **Image Upload**: Users upload an image through the web interface
2. **Downsampling**: The image is downsampled to create a low-resolution version
3. **GAN Processing**: Both SRGAN and ESRGAN models process the low-resolution image
4. **Super-Resolution**: The models generate high-resolution enhanced versions
5. **Display Results**: Original and enhanced images are displayed side-by-side

## Model Information

- **SRGAN**: Super-Resolution Generative Adversarial Network
- **ESRGAN**: Enhanced Super-Resolution Generative Adversarial Network
- Both models are pre-trained and ready to use
- Processing time depends on image size and server performance

## API Endpoints

- `GET /`: Main web interface
- `POST /`: Upload and process images

## Environment Variables

No environment variables are required for basic deployment.

## Troubleshooting

### Common Issues

1. **Model files missing**: Ensure `generator.pt` and `last_G.pt` are in the root directory
2. **Memory issues**: The models require significant RAM. Consider using a service with at least 1GB RAM
3. **Processing time**: Large images may take longer to process

### Performance Tips

- Use images under 5MB for faster processing
- Consider image compression before upload
- For production, consider using GPU-enabled servers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions, please open an issue on GitHub. 